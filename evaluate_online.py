import argparse
import torch
import time
from torch.utils.data import TensorDataset, DataLoader
from sliding_utils import load_h5_data, segment_signal, merge_predicted_events, visualize_global_events
from eval_utils import post_process_segmentation, extract_events_from_map
from models.ours import ConSegAP as OursConSegAP
from models.ctnet import ConSegAP as CTNetConSegAP
from models.deepconvnet import ConSegAP as DeepConvNetConSegAP
from models.conformer import ConSegAP as ConformerConSegAP
from models.eegnet import ConSegAP as EEGNetConSegAP
from utils import device

def main():
    parser = argparse.ArgumentParser(description="Evaluate ConSegAP with sliding window on BCI IV-2a sample")
    parser.add_argument('--data-file', type=str, required=True, help='Path to train_data.h5 file')
    parser.add_argument('--model', type=str, required=True, choices=['ours', 'CTNet', 'DeepConvNet', 'conformer', 'EEGNet'], help='Model type')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index to evaluate (0-based)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--stride', type=int, default=50, help='Sliding window stride')
    parser.add_argument('--segment-length', type=int, default=5000, help='Segment length')
    parser.add_argument('--min-duration-threshold', type=int, default=200, help='Minimum event duration in samples')
    parser.add_argument('--output-dir', type=str, default='plots', help='Directory to save visualization plots')

    args = parser.parse_args()

    # Load data
    test_segments, test_events, metadata = load_h5_data(args.data_file)
    print("Metadata:")
    print(f"  Description: {metadata['description']}")
    print(f"  Test segments shape: {metadata['test_segments_shape']}")
    print(f"  Test events shape: {metadata['test_events_shape']}")
    print(f"Loaded test_segments shape: {test_segments.shape}")
    print(f"Loaded test_events shape: {test_events.shape}")

    # Segment signal
    segments, seg_events = segment_signal(
        sample_idx=args.sample_idx,
        stride=args.stride,
        test_segments=test_segments,
        test_events=test_events,
        segment_length=args.segment_length
    )
    print(f"Segments shape: {segments.shape}")
    print(f"Number of segments: {len(seg_events)}")

    # Create dataset and loader
    test_dataset = TensorDataset(segments)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model_classes = {
        'ours': OursConSegAP,
        'CTNet': CTNetConSegAP,
        'DeepConvNet': DeepConvNetConSegAP,
        'conformer': ConformerConSegAP,
        'EEGNet': EEGNetConSegAP
    }
    model_params = {
        'ours': {'num_channels': 22, 'emb_size': 64, 'num_classes': 5},
        'CTNet': {'num_channels': 22, 'emb_size': 64, 'num_classes': 5},
        'DeepConvNet': {'num_channels': 22, 'emb_size': 64, 'feature_dim': 1400, 'num_classes': 5},
        'conformer': {'num_channels': 22, 'emb_size': 64, 'num_classes': 5},
        'EEGNet': {'num_channels': 22, 'emb_size': 64, 'feature_dim': 16, 'num_classes': 5}
    }

    model_class = model_classes[args.model]
    model = model_class(**model_params[args.model]).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- Model Parameters ---")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Total Parameters (in millions): {total_params / 1e6:.2f} M")
    print('================================================')

    # Evaluate
    print(f"Evaluating {args.model} model on sample {args.sample_idx + 1}...")
    total_samples = segments.shape[0]
    total_inference_time = 0.0
    pred_events_list = []

    with torch.no_grad():
        for batch_idx, (batch_x,) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            short_win_logits, long_win_logits, grid_sizes, _ = model(batch_x)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            total_inference_time += end_time - start_time

            predicted_map = post_process_segmentation(
                long_win_logits,
                short_win_logits,
                grid_sizes,
                min_duration_threshold=args.min_duration_threshold,
                label_method='majority_vote'
            )
            pred_events_list.extend(extract_events_from_map(predicted_map))

    # Merge predicted events
    initial_events, votes = merge_predicted_events(
        pred_events_list,
        stride=args.stride,
        segment_length=args.segment_length,
        total_length=test_segments.shape[2]
    )
    print(f"Global events shape: {initial_events.shape}")

    # Visualize results
    true_events = test_events[args.sample_idx].cpu().numpy()
    visualize_global_events(
        sample_idx=args.sample_idx,
        signal=test_segments[args.sample_idx],
        global_events=initial_events,
        true_events=true_events,
        output_dir=args.output_dir
    )

    # Report inference performance
    per_sample_time = (total_inference_time / total_samples) if total_samples > 0 else 0.0
    throughput = (total_samples / total_inference_time) if total_inference_time > 0 else 0.0

    print("\n--- Inference Performance ---")
    print(f"  Total samples processed: {total_samples}")
    print(f"  Average inference time per sample: {per_sample_time:.2f} s")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    print(f"  Total inference time: {total_inference_time:.3f} seconds")
    if total_inference_time < 60:
        print(f"  Efficiency note: Processed {total_samples} segments in {total_inference_time:.3f} seconds, "
            f"achieving real-time performance for EEG analysis.")
    else:
        print(f"  Efficiency note: Processed {total_samples} segments in {total_inference_time:.3f} seconds, "
            f"which may not be suitable for real-time EEG analysis.")    
    print('================================================')

if __name__ == "__main__":
    main()