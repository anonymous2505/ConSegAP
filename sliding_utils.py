import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import device

def load_h5_data(file_path):
    """Load test segments and events from an HDF5 file."""
    with h5py.File(file_path, "r") as f:
        test_segments = torch.from_numpy(f["test_segments"][:])
        test_events = torch.from_numpy(f["test_events"][:])
        metadata = {
            "description": f.attrs.get("description", ""),
            "test_segments_shape": f.attrs.get("test_segments_shape", test_segments.shape),
            "test_events_shape": f.attrs.get("test_events_shape", test_events.shape)
        }
    return test_segments, test_events, metadata

def segment_signal(sample_idx, stride, test_segments, test_events, segment_length=5000):
    """
    Segment a signal and its events using a sliding window.

    Args:
        sample_idx (int): Sample index (0-based).
        stride (int): Sliding window stride.
        test_segments (torch.Tensor): Shape [num_samples, 22, total_length].
        test_events (torch.Tensor): Shape [num_samples, num_events, 3].
        segment_length (int): Length of each segment.

    Returns:
        segments (torch.Tensor): Shape [num_segments, 22, segment_length].
        segment_events (list): List of event arrays per segment.
    """
    signal = test_segments[sample_idx].cpu()  # [22, total_length]
    events = test_events[sample_idx].cpu().numpy()  # [num_events, 3]

    total_length = signal.shape[1]
    num_segments = (total_length - segment_length) // stride + 1

    segments = []
    segment_events = []

    for seg_idx in range(num_segments):
        start = seg_idx * stride
        end = start + segment_length
        segment = signal[:, start:end]
        segments.append(segment)

        seg_events = []
        for event in events:
            start_time, end_time, label = event
            label = int(label)
            if label == 0 or start_time == 0:
                continue
            if start_time < end and end_time > start:
                rel_start = max(start_time - start, 0)
                rel_end = min(end_time - start, segment_length)
                if rel_end > rel_start:
                    seg_events.append([rel_start, rel_end, label])
        segment_events.append(np.array(seg_events) if seg_events else np.empty((0, 3)))

    segments = torch.stack(segments)
    return segments, segment_events

def merge_predicted_events(pred_events_list, stride=50, segment_length=5000, total_length=15000):
    """
    Merge predicted events to a global timeline.

    Args:
        pred_events_list (list): List of [num_events, 3] event arrays.
        stride (int): Sliding window stride.
        segment_length (int): Segment length.
        total_length (int): Global timeline length.

    Returns:
        global_events (np.ndarray): Merged events [num_global_events, 3].
        votes (list): List of predicted labels per time point.
    """
    num_segments = len(pred_events_list)
    votes = [[] for _ in range(total_length)]

    for seg_idx, events in enumerate(pred_events_list):
        seg_start = seg_idx * stride
        if len(events) > 0:
            for event in events:
                rel_start, rel_end, label = event
                label = int(label)
                if label == 0:
                    continue
                global_start = int(seg_start + rel_start)
                global_end = int(seg_start + rel_end)
                global_start = min(global_start, total_length - 1)
                global_end = min(global_end, total_length)
                for t in range(global_start, global_end):
                    votes[t].append(label)

    merged_labels = np.zeros(total_length, dtype=np.int32)
    for t in range(total_length):
        if votes[t]:
            vote_counts = np.bincount(votes[t], minlength=5)[1:]
            merged_labels[t] = np.argmax(vote_counts) + 1 if vote_counts.sum() > 0 else 0

    global_events = []
    current_label = 0
    start_time = 0
    for t in range(total_length):
        if merged_labels[t] != current_label:
            if current_label != 0:
                global_events.append([start_time, t, current_label])
            current_label = merged_labels[t]
            start_time = t
    if current_label != 0:
        global_events.append([start_time, total_length, current_label])

    return np.array(global_events) if global_events else np.empty((0, 3)), votes

def visualize_global_events(sample_idx, signal, global_events, true_events, output_dir="plots"):
    """
    Visualize predicted and true events on the global timeline.

    Args:
        sample_idx (int): Sample index (0-based).
        signal (torch.Tensor): Signal [22, total_length].
        global_events (np.ndarray): Predicted events [num_events, 3].
        true_events (np.ndarray): True events [num_events, 3].
        output_dir (str): Directory to save plots.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}
    labels = {1: 'Event 1', 2: 'Event 2', 3: 'Event 3', 4: 'Event 4'}

    plt.figure(figsize=(15, 8))
    time_points = np.arange(signal.shape[1])

    plt.subplot(2, 1, 1)
    temp_labels = labels.copy()
    for event in global_events:
        start, end, label = event
        label = int(label)
        if label != 0:
            plt.axvspan(start, end, alpha=0.3, color=colors.get(label, 'gray'),
                        label=temp_labels.get(label, f'Pred Event {label}'))
            temp_labels[label] = '_' + temp_labels[label]
    plt.title(f'Sample {sample_idx + 1} Predicted Events')
    plt.xlabel('Time Points')
    plt.ylabel('Signal Amplitude')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    temp_labels = labels.copy()
    for event in true_events:
        start, end, label = event
        label = int(label)
        if label != 0:
            plt.axvspan(start, end, alpha=0.3, color=colors.get(label, 'gray'),
                        label=temp_labels.get(label, f'True Event {label}'))
            temp_labels[label] = '_' + temp_labels[label]
    plt.title(f'Sample {sample_idx + 1} True Events')
    plt.xlabel('Time Points')
    plt.ylabel('Signal Amplitude')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'global_events_sample_{sample_idx + 1}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()