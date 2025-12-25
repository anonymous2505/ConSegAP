import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    print("Warning: Torchaudio library not found.")
    print("         Frequency-based activity detector will not be available.")
    print("         Falling back to the simple Conv1d activity detector.")
    T = None # Define T as None if import fails
    TORCHAUDIO_AVAILABLE = False

class AdaptivePatchEmbedding(nn.Module):
    """
    Adaptive Patch Embedding module for EEG sequences.

    This module extracts features from EEG signals using a multi-scale CNN
    (inspired by Multi-Scale CNN/DeepConvNet) and then adaptively segments
    the sequence into patches based on frequency-domain activity (Mu/Beta bands).
    It outputs a fixed number of patch embeddings suitable for sequence models.

    Key features:
    1. Multi-scale temporal feature extraction.
    2. Activity detection based on STFT power in Mu (8-13Hz) and Beta (13-30Hz) bands.
    3. Adaptive patch partitioning: Higher activity -> shorter patches (higher resolution).
    4. Combined Average + Max Pooling for patch representation.
    5. Returns patch embeddings, patch durations (grid sizes), and activity map.
    """
    def __init__(self,
                 num_channels: int = 22,
                 emb_size: int = 64,
                 base_patch_size: int = 25, # Also used as hop_length for STFT
                 scales: list = [(1, 9), (1, 25), (1, 49)], # Kernel sizes for Multi-Scale CNN Conv2d time dim
                 num_patches: int = 100, # Desired number of output patches
                 sampling_rate: int = 250, # Hz, crucial for frequency band calculation
                 n_fft: int = 128, # FFT window size for STFT
                 min_patch_size: int = 10, # Minimum allowed patch duration in samples
                 max_patch_size: int = 100, # Maximum allowed patch duration in samples
                 shallow_cnn_out_channels: int = 40, # Output channels per scale in Multi-Scale CNN part
                 dropout_rate: float = 0.5
                 ):
        """
        Initializes the AdaptivePatchEmbedding module.

        Args:
            num_channels (int): Number of EEG channels.
            emb_size (int): Dimension of the output patch embeddings.
            base_patch_size (int): Base granularity and STFT hop length.
            scales (list): List of tuples for Conv2d kernel sizes (time dim) in Multi-Scale CNN.
            num_patches (int): Fixed number of output patches.
            sampling_rate (int): Sampling rate of the input EEG signal (Hz).
            n_fft (int): Window size for STFT. Affects frequency resolution.
                         Choose based on sampling rate and desired resolution (e.g., 128 or 256 for 250Hz).
            min_patch_size (int): Minimum duration (samples) for an adaptive patch.
            max_patch_size (int): Maximum duration (samples) for an adaptive patch.
            shallow_cnn_out_channels (int): Number of output filters for each scale's Conv2d.
            dropout_rate (float): Dropout probability used in Multi-Scale CNN part.
        """
        super().__init__()
        self.num_channels = num_channels
        self.emb_size = emb_size
        self.base_patch_size = base_patch_size
        self.scales = scales
        self.num_patches = num_patches
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = base_patch_size # Link STFT time resolution to base patch size
        self.min_size = min_patch_size
        self.max_size = max_patch_size
        self.shallow_cnn_out_channels = shallow_cnn_out_channels
        self.dropout_rate = dropout_rate

        # --- 1. Multi-Scale CNN-like Multi-Scale Feature Extractor ---
        # Uses Conv2d for temporal and spatial filtering across different time scales
        self.Multi_scale_CNN = nn.ModuleList([
            nn.Sequential(
                # Temporal Convolution (different kernel sizes)
                nn.Conv2d(1, self.shallow_cnn_out_channels, kernel_size=(1, scale[1]), stride=(1, 1),
                          padding=(0, (scale[1] - 1) // 2)),
                # Spatial Convolution (learns spatial filters across channels)
                nn.Conv2d(self.shallow_cnn_out_channels, self.shallow_cnn_out_channels,
                          kernel_size=(num_channels, 1), stride=(1, 1),
                          groups=1), # Use groups=1 for standard convolution here
                          # bias=False), # Often used with BatchNorm
                nn.BatchNorm2d(self.shallow_cnn_out_channels),
                nn.ELU(),
                # nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)), # Optional pooling? Keep length 5000 for now.
                nn.Dropout(self.dropout_rate),
            ) for scale in scales
        ])
        self.num_feature_channels = self.shallow_cnn_out_channels * len(scales) # e.g., 40 * 3 = 120

        # --- 2. Frequency-Based Activity Detector Components ---
        self.activity_detector_type = "none"
        if TORCHAUDIO_AVAILABLE:
            try:
                self.stft = T.Spectrogram(
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.n_fft, # Can be same as n_fft
                    window_fn=torch.hann_window,
                    power=2.0, # Output power spectrogram |STFT|^2
                    # batch_first=True # Assume input [B, C, T] -> output [B, C, Freq, T_frames]
                )

                # Calculate frequency bins and indices for Mu (8-13Hz) and Beta (13-30Hz)
                freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/self.sampling_rate)
                self.mu_indices = (freqs >= 8) & (freqs <= 13)
                self.beta_indices = (freqs >= 13) & (freqs <= 30)

                if not self.mu_indices.any() or not self.beta_indices.any():
                    print(f"Warning: Mu ({self.mu_indices.sum()} bins) or Beta ({self.beta_indices.sum()} bins) band is empty.")
                    print(f"         Check n_fft ({self.n_fft}) and sampling_rate ({self.sampling_rate}).")
                    print(f"         Falling back to simple activity detector.")
                    # Fallback flag needed inside __init__ if check fails
                    USE_SIMPLE_ACTIVITY = True
                else:
                    # Define CNN to process combined band power [B, C, T_frames] -> [B, 1, T_frames]
                    self.activity_conv1 = nn.Conv1d(num_channels, num_channels // 2, kernel_size=3, padding=1)
                    self.activity_bn1 = nn.BatchNorm1d(num_channels // 2)
                    self.activity_relu = nn.ReLU()
                    self.activity_conv2 = nn.Conv1d(num_channels // 2, 1, kernel_size=1) # Aggregate channels
                    self.activity_sigmoid = nn.Sigmoid()
                    self.activity_detector_type = "frequency"
                    USE_SIMPLE_ACTIVITY = False

            except Exception as e:
                print(f"Error initializing STFT or frequency bands: {e}")
                print("Falling back to simple activity detector.")
                USE_SIMPLE_ACTIVITY = True
        else:
            USE_SIMPLE_ACTIVITY = True # Torchaudio not available

        # Fallback simple activity detector if needed
        if USE_SIMPLE_ACTIVITY and not hasattr(self, 'activity_detector'):
             print("Using simple Conv1d activity detector.")
             self.activity_detector = nn.Sequential(
                 nn.Conv1d(num_channels, 1, kernel_size=base_patch_size, stride=base_patch_size),
                 nn.Sigmoid(),
             )
             self.activity_detector_type = "simple_conv"


        # --- 3. Final Projection Layer ---
        # Input dimension depends on pooling strategy (Avg + Max = * 2)
        self.pooling_factor = 1 # 1 for Avg only, 2 for Avg+Max
        self.projection_in_dim = self.num_feature_channels * self.pooling_factor
        self.projection = nn.Conv1d(self.projection_in_dim, emb_size, kernel_size=1)

    def _adaptive_grid(self, activity, total_length):
        """
        Generates adaptive grid sizes based on activity map.
        Higher activity leads to smaller grid sizes (patches).
        Ensures the sum of grid sizes equals total_length.
        """
        batch_size, num_patches = activity.shape
        device = activity.device

        # Inverse activity defines weights: high activity -> small weight -> small patch
        weights = 1.0 / (activity + 1e-6) # Add epsilon for numerical stability
        weights = weights / torch.sum(weights, dim=1, keepdim=True) # Normalize weights per batch item

        # Calculate initial grid sizes
        grid_sizes = weights * total_length
        grid_sizes = torch.clamp(grid_sizes, self.min_size, self.max_size) # Clamp to min/max

        # Adjust grid sizes to sum exactly to total_length
        # Iterative adjustment or proportional scaling can be used. Proportional scaling:
        current_sum = torch.sum(grid_sizes, dim=1, keepdim=True)
        scale_factor = total_length / current_sum
        grid_sizes = grid_sizes * scale_factor

        # Round and ensure sum is *exactly* total_length using integer arithmetic
        grid_sizes_rounded = grid_sizes.round().long()
        current_sum_int = torch.sum(grid_sizes_rounded, dim=1)
        diff = total_length - current_sum_int

        # Distribute rounding difference (add to the last patch for simplicity)
        for b in range(batch_size):
            grid_sizes_rounded[b, -1] += diff[b]

        # Final clamp after adjustments
        grid_sizes_final = torch.clamp(grid_sizes_rounded, self.min_size, self.max_size)

        # Need one more check/adjustment if clamping changes the sum again (less likely with clamp first, scale later)
        final_sum_check = torch.sum(grid_sizes_final, dim=1)
        final_diff = total_length - final_sum_check
        for b in range(batch_size):
            if final_diff[b] != 0:
                # Simple fix: Adjust last element again if clamping caused issues
                grid_sizes_final[b, -1] += final_diff[b]

        # Ensure no zero sizes if min_size > 0
        grid_sizes_final[grid_sizes_final < self.min_size] = self.min_size

        return grid_sizes_final # Return as tensor [B, num_patches]

    def forward(self, x):
        """
        Forward pass of the AdaptivePatchEmbedding module.

        Args:
            x (torch.Tensor): Input EEG signal. Shape: [batch, channels, time_steps]
                              Example: [B, 22, 5000]

        Returns:
            torch.Tensor: Output patch embeddings. Shape: [batch, num_patches, emb_size]
                          Example: [B, num_patches, 64]
            torch.Tensor: Tensor of adaptive grid sizes (patch durations). Shape: [batch, num_patches]
                          Example: [B, num_patches]
            torch.Tensor: Interpolated activity map used for grid generation. Shape: [batch, num_patches]
                          Example: [B, num_patches]
        """
        b, c, t = x.shape
        device = x.device

        # --- 1. Calculate Activity Map ---
        activity = None
        if self.activity_detector_type == "frequency":
            # Calculate STFT
            x_stft = self.stft(x) # [B, C, Freq, T_frames]
            expected_time_frames = t // self.hop_length

            # Adjust T_frames dimension if needed (due to n_fft padding/windowing)
            if x_stft.shape[-1] != expected_time_frames:
                diff = expected_time_frames - x_stft.shape[-1]
                if diff > 0: x_stft = F.pad(x_stft, (0, diff))
                else: x_stft = x_stft[..., :expected_time_frames]

            # Get band power
            # Move indices to correct device inside forward pass
            mu_power = torch.mean(x_stft[..., self.mu_indices.to(device), :], dim=-2)
            beta_power = torch.mean(x_stft[..., self.beta_indices.to(device), :], dim=-2)
            combined_power = mu_power + beta_power # [B, C, T_frames]
            
            act = self.activity_conv1(combined_power)
            act = self.activity_bn1(act)
            act = self.activity_relu(act)
            act = self.activity_conv2(act).squeeze(1) # [B, 1, T_frames]
            activity = self.activity_sigmoid(act) # [B, T_frames] (~200 length)

        elif self.activity_detector_type == "simple_conv":
             # Calculate expected output length for simple conv detector
             expected_time_frames = (t - self.base_patch_size) // self.base_patch_size + 1 # Manual calculation
             act_raw = self.activity_detector(x).squeeze(1) # [B, ~200]
             # Adjust length if needed (less likely here if stride=kernel)
             if act_raw.shape[-1] != expected_time_frames:
                 activity = F.interpolate(act_raw.unsqueeze(1), size=expected_time_frames, mode='linear', align_corners=False).squeeze(1)
             else:
                 activity = act_raw
        else:
            # Should not happen if __init__ is correct
            raise RuntimeError("Activity detector not properly initialized.")

        # Interpolate activity map to the target number of patches
        activity_interpolated = F.interpolate(activity.unsqueeze(1), size=self.num_patches, mode='linear', align_corners=False).squeeze(1) # [B, num_patches]

        # --- 2. Generate Adaptive Grid Sizes ---
        grid_sizes_tensor = self._adaptive_grid(activity_interpolated, total_length=t) # [B, num_patches]

        # --- 3. Extract Multi-Scale CNN Features ---
        # Input shape for Conv2d: [B, C_in, H, W], here [B, 1, Channels, Time]
        x_unsqueezed = x.unsqueeze(1)
        features = torch.cat([scale_CNN(x_unsqueezed).squeeze(2) for scale_CNN in self.Multi_scale_CNN], dim=1) # [B, 120, T]

        # --- 4. Adaptive Patching and Pooling ---
        boundaries = torch.cumsum(grid_sizes_tensor, dim=1)
        starts = torch.cat([torch.zeros(b, 1, device=device), boundaries[:, :-1]], dim=1).long()
        ends = boundaries.long()

        patches_pooled = []
        for b_idx in range(b):
            batch_item_patches = []
            for i in range(self.num_patches):
                start, end = starts[b_idx, i], ends[b_idx, i]

                # Handle zero-length or invalid patches after adjustments
                if start >= end or start < 0 or end > t:
                    # Use zero vector if patch is invalid
                    # Dimension should match projection input dim / pooling_factor
                    pooled_features = torch.zeros(self.num_feature_channels, device=device)
                else:
                    patch_slice = features[b_idx, :, start:end].unsqueeze(0) # [1, 120, PatchLen]

                    # Apply combined Avg + Max Pooling
                    avg_pooled = F.adaptive_avg_pool1d(patch_slice, 1).squeeze() # [120]
                    # max_pooled = F.adaptive_max_pool1d(patch_slice, 1).squeeze() # [120]
                    pooled_features = avg_pooled # [240]

                batch_item_patches.append(pooled_features)
            patches_pooled.append(torch.stack(batch_item_patches)) # Stack patches for this batch item: [num_patches, 240]

        x_pooled = torch.stack(patches_pooled) # [B, num_patches, 240]

        # --- 5. Final Projection ---
    
        x_projected = self.projection(x_pooled.transpose(1, 2)) # [B, emb_size, num_patches]

        # Output shape: [B, Length, Channels] -> [B, num_patches, emb_size]
        x_out = x_projected.transpose(1, 2)

        # --- 6. Return Results ---
        return x_out, grid_sizes_tensor, activity_interpolated


# # --- Example Usage ---
# if __name__ == "__main__":
#     # Parameters
#     batch_size = 4
#     num_channels = 22
#     time_steps = 5000 
#     emb_size = 64
#     num_patches = num_patches
#     sampling_rate = 250

#     # Create dummy input tensor
#     dummy_eeg = torch.randn(batch_size, num_channels, time_steps)

#     # Instantiate the model
#     model = AdaptivePatchEmbedding(
#         num_channels=num_channels,
#         emb_size=emb_size,
#         num_patches=num_patches,
#         sampling_rate=sampling_rate,
#     )

#     # Print model info
#     print("Model Architecture:")
#     print(model)
#     print(f"\nActivity Detector Type: {model.activity_detector_type}")

#     # Perform forward pass
#     if model.activity_detector_type != "none":
#       try:
#           with torch.no_grad(): # Disable gradient calculation for inference example
#               x_out, grid_sizes, activity = model(dummy_eeg)

#           # Print output shapes
#           print("\n--- Output Shapes ---")
#           print(f"Input EEG shape:       {dummy_eeg.shape}")
#           print(f"Output Embeddings shape: {x_out.shape}")     # Expected: [B, num_patches, emb_size]
#           print(f"Grid Sizes shape:      {grid_sizes.shape}")  # Expected: [B, num_patches]
#           print(f"Activity Map shape:    {activity.shape}")    # Expected: [B, num_patches]

#           # Verify grid sizes sum for the first batch item
#           print("\n--- Grid Size Verification ---")
#           sum_grid_sizes_b0 = torch.sum(grid_sizes[0])
#           print(f"Sum of grid sizes for batch 0: {sum_grid_sizes_b0}") # Should be == time_steps (5000)
#           if sum_grid_sizes_b0 == time_steps:
#               print("Grid sizes sum verification PASSED.")
#           else:
#               print(f"Grid sizes sum verification FAILED! Expected {time_steps}, got {sum_grid_sizes_b0}.")

#           print(f"\nMin grid size found: {grid_sizes.min()}")
#           print(f"Max grid size found: {grid_sizes.max()}")

#       except Exception as e:
#           print(f"\nAn error occurred during the forward pass: {e}")
#           import traceback
#           traceback.print_exc()
#     else:
#       print("\nCannot run forward pass example as activity detector failed to initialize.")