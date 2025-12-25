import numpy as np
import mne
import scipy.signal as signal
import scipy.io as sio
import os

def get_data(subject_index):
    """
    Process BCI Competition IV 2a dataset for a given subject.
    Args:
        subject_index (int): Subject number (1-9)
    Returns:
        None (saves processed data and labels to .mat files)
    """
    subject_index = 6  # Fixed to 6 as in original code; can be modified

    # Process both session types: Training (T) and Evaluation (E)
    for session_type in ['T', 'E']:
        # File paths (adjust as needed)
        gdf_file = f'/home/jun-li/文档/datatat/BCICIV_2a_gdf/A0{subject_index}{session_type}.gdf'
        label_file = f'/home/jun-li/文档/datatat/true_labels/A0{subject_index}{session_type}.mat'

        # Load GDF file using MNE
        raw = mne.io.read_raw_gdf(gdf_file, preload=True)
        events, event_id = mne.events_from_annotations(raw)

        # Load labels from .mat file
        label_data = sio.loadmat(label_file)
        labels = label_data['classlabel'].ravel()  # Assuming 'classlabel' is the key

        # Extract trial data (22 channels x 1000 samples x 288 trials)
        data = np.zeros((22, 1000, 288))
        k = 0
        for j, event in enumerate(events):
            if event[2] == event_id.get('768', None):  # Event type 768
                start = event[0] + 500
                end = event[0] + 1500
                data[:, :, k] = raw.get_data(picks='eeg', start=start, stop=end)[:22, :]
                k += 1

        # Replace NaN with 0
        data = np.nan_to_num(data, nan=0)

        # Bandpass filter (4-40 Hz, sampling rate 250 Hz)
        fs = 250  # Sampling rate
        low = 4
        high = 40
        wn = [low * 2 / fs, high * 2 / fs]  # Normalized frequencies
        b, a = signal.cheby2(6, 60, wn, btype='bandpass')
        for j in range(data.shape[2]):
            data[:, :, j] = signal.filtfilt(b, a, data[:, :, j], axis=1)  # Filter along time axis

        # Standardization (mean and std across trials)
        eeg_mean = np.mean(data, axis=2, keepdims=True)  # Shape: (22, 1000, 1)
        eeg_std = np.std(data, axis=2, keepdims=True)    # Shape: (22, 1000, 1)
        data = (data - eeg_mean) / (eeg_std + 1e-10)     # Avoid division by zero

        # Transpose data to match MATLAB format (1000 samples x 22 channels x 288 trials)
        data = np.transpose(data, (1, 0, 2))

        # Save to .mat file
        save_dir = f'/home/jun-li/文档/datatat/A0{subject_index}{session_type}.mat'
        sio.savemat(save_dir, {'data': data, 'label': labels})

if __name__ == "__main__":
    get_data(6)  # Example call with subject_index=6