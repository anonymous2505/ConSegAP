import torch
import numpy as np
import time
from utils import device, align_labels

def find_contiguous_segments(labels):
    """Find start, end indices, and value of contiguous segments in a 1D array."""
    if len(labels) == 0:
        return []
    segments = []
    start_idx = 0
    current_val = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current_val:
            segments.append((start_idx, i - 1, current_val))
            start_idx = i
            current_val = labels[i]
    segments.append((start_idx, len(labels) - 1, current_val))
    return segments

def extract_events_from_map(pred_map_batch):
    """Convert a batch of predicted time maps [B, T] to event lists."""
    batch_events = []
    for b in range(pred_map_batch.shape[0]):
        labels = pred_map_batch[b].cpu().numpy()
        T = len(labels)
        events = []
        if T == 0:
            batch_events.append(events)
            continue
        start_idx = 0
        current_class = labels[0]
        for i in range(1, T):
            if labels[i] != current_class:
                if current_class != 0:
                    events.append([start_idx, i, current_class])
                start_idx = i
                current_class = labels[i]
        if current_class != 0:
            events.append([start_idx, T, current_class])
        batch_events.append(events)
    return batch_events

def compute_event_metrics(pred_events_list, true_events_list):
    """
    Compute event-level evaluation metrics.

    Args:
        pred_events_list: List of predicted event lists [[start, end, class], ...].
        true_events_list: List of ground truth event lists [[start, end, class], ...].

    Returns:
        dict: Metrics including iou_class_agnostic, event_accuracy.
    """
    batch_size = len(pred_events_list)
    total_specific_iou_sum = 0.0       
    total_agnostic_iou_sum = 0.0       
    total_true_events_count = 0       
    total_agnostic_matches_count = 0   
    correctly_classified_agnostic_matches = 0

    for b in range(batch_size):
        preds = pred_events_list[b]
        trues = true_events_list[b]

        try:
            trues = [[int(s), int(e), int(c)] for s, e, c in trues if int(c) != 0]
            preds = [[int(s), int(e), int(c)] for s, e, c in preds]
        except (ValueError, TypeError, IndexError):
             print(f"Warning: Invalid event format found in batch {b}. Skipping sample.")
             continue 
        if not trues:
            continue

        for true in trues:
            true_start, true_end, true_class = true
            true_len = true_end - true_start
            if true_len <= 0: continue

            total_true_events_count += 1 

            best_specific_iou_for_true = 0.0 
            best_agnostic_iou_for_true = 0.0 
            best_agnostic_match_pred_class = -1 

            if not preds: 
                continue

            for pred in preds:
                pred_start, pred_end, pred_class = pred
                pred_len = pred_end - pred_start
                if pred_len <= 0: continue

                inter_start = max(true_start, pred_start)
                inter_end = min(true_end, pred_end)
                intersection = max(0, inter_end - inter_start)
                union = true_len + pred_len - intersection

                if union > 0:
                    iou = intersection / union

                    if iou > best_agnostic_iou_for_true:
                        best_agnostic_iou_for_true = iou
                        best_agnostic_match_pred_class = pred_class

                    if pred_class == true_class: 
                        best_specific_iou_for_true = max(best_specific_iou_for_true, iou)

            total_specific_iou_sum += best_specific_iou_for_true 

            if best_agnostic_iou_for_true > 0.8: 
                total_agnostic_iou_sum += best_agnostic_iou_for_true
                total_agnostic_matches_count += 1
                if best_agnostic_match_pred_class == true_class:
                    correctly_classified_agnostic_matches += 1


    avg_iou_specific = total_specific_iou_sum / total_true_events_count if total_true_events_count > 0 else 0.0

    avg_iou_agnostic = total_agnostic_iou_sum / total_agnostic_matches_count if total_agnostic_matches_count > 0 else 0.0

    event_accuracy = correctly_classified_agnostic_matches / total_agnostic_matches_count if total_agnostic_matches_count > 0 else 0.0

    if total_agnostic_matches_count == 0 and total_true_events_count > 0:
        # avg_iou_agnostic = 0.0
        event_accuracy = 0.0 

    return {
        'iou_class_specific': avg_iou_specific,
        'iou_class_agnostic': avg_iou_agnostic, 
        'event_accuracy': event_accuracy         
    }

def compute_grid_accuracy(normal_context_logits, batch_events, grid_sizes):
    """Compute patch-wise accuracy for a batch."""
    aligned_labels = align_labels(batch_events, grid_sizes).to(device)
    preds = torch.argmax(normal_context_logits, dim=-1)
    correct = (preds == aligned_labels).sum().item()
    total = aligned_labels.numel()
    return correct, total

def post_process_segmentation(long_win_logits, short_win_logits, grid_sizes, min_duration_threshold=700, label_method='sum_logits'):
    """Post-process model outputs to generate final segmentation map."""
    device = long_win_logits.device
    long_win_logits_cpu = long_win_logits.cpu().detach()
    short_win_logits_cpu = short_win_logits.cpu().detach()

    if isinstance(grid_sizes, list):
        batch_size = len(grid_sizes)
        num_patches = len(grid_sizes[0])
        grid_sizes_tensor = torch.stack([torch.as_tensor(g, dtype=torch.long) for g in grid_sizes]).cpu()
    elif torch.is_tensor(grid_sizes):
        batch_size = grid_sizes.shape[0]
        num_patches = grid_sizes.shape[1]
        grid_sizes_tensor = grid_sizes.cpu().detach().long()
    else:
        raise TypeError("grid_sizes must be a tensor or list of tensors/lists")

    total_time_steps = grid_sizes_tensor[0].sum().item()
    final_segmentation = torch.zeros((batch_size, total_time_steps), dtype=torch.long, device=device)

    long_labels = torch.argmax(long_win_logits_cpu, dim=2)

    for b in range(batch_size):
        time_boundaries = torch.cumsum(torch.cat([torch.tensor([0]), grid_sizes_tensor[b]]), dim=0)
        detected_segments = find_contiguous_segments(long_labels[b].numpy())
        valid_segments = []
        for start_patch_idx, end_patch_idx, detected_class_long in detected_segments:
            if detected_class_long == 0:
                continue
            duration_samples = grid_sizes_tensor[b, start_patch_idx:end_patch_idx + 1].sum().item()
            if duration_samples >= min_duration_threshold:
                valid_segments.append({
                    "start_patch": start_patch_idx,
                    "end_patch": end_patch_idx,
                    "start_time": time_boundaries[start_patch_idx].item(),
                    "end_time": time_boundaries[end_patch_idx + 1].item()
                })

        for segment in valid_segments:
            start_p, end_p = segment["start_patch"], segment["end_patch"]
            start_t, end_t = segment["start_time"], segment["end_time"]
            segment_short_logits = short_win_logits_cpu[b, start_p:end_p + 1, :]
            segment_class = 0
            if label_method == 'sum_logits':
                if segment_short_logits.numel() > 0:
                    summed_logits = torch.sum(segment_short_logits, dim=0)
                    segment_class = torch.argmax(summed_logits).item()
            elif label_method == 'majority_vote':
                if segment_short_logits.numel() > 0:
                    short_labels_segment = torch.argmax(segment_short_logits, dim=1)
                    non_zero_labels = short_labels_segment[short_labels_segment != 0]
                    if non_zero_labels.numel() > 0:
                        counts = torch.bincount(non_zero_labels)
                        if len(counts) > 0:
                            segment_class = torch.argmax(counts).item()
            else:
                raise ValueError(f"Unknown label_method: {label_method}")

            if segment_class != 0:
                start_t_idx, end_t_idx = max(0, start_t), min(total_time_steps, end_t)
                if start_t_idx < end_t_idx:
                    final_segmentation[b, start_t_idx:end_t_idx] = segment_class

    return final_segmentation

def evaluate_model(model, test_loader, min_duration_threshold=700):
    """Evaluate the model on the test dataset."""
    model.eval()
    total_samples = 0
    total_correct = 0
    total_preds = 0
    all_event_metrics = []
    total_inference_time = 0.0

    with torch.no_grad():
        for batch_idx, (batch_x, batch_events_list) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            current_batch_size = batch_x.shape[0]
            total_samples += current_batch_size

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            short_win_logits, long_win_logits, grid_sizes, _ = model(batch_x)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            batch_duration = end_time - start_time
            total_inference_time += batch_duration

            # Compute Grid accuracy
            correct, total = compute_grid_accuracy(short_win_logits, batch_events_list, grid_sizes)
            total_correct += correct
            total_preds += total

            # Compute event-based metrics
            predicted_map = post_process_segmentation(
                long_win_logits,
                short_win_logits,
                grid_sizes,
                min_duration_threshold=min_duration_threshold,
                label_method='sum_logits'
            )

            pred_events_list = extract_events_from_map(predicted_map)
            metrics = compute_event_metrics(pred_events_list, batch_events_list)
            all_event_metrics.append(metrics)

    grid_accuracy = (total_correct / total_preds * 100) if total_preds > 0 else 0.0
    per_sample_time = (total_inference_time / total_samples) if total_samples > 0 else 0.0  # s/sample
    throughput = (total_samples / total_inference_time) if total_inference_time > 0 else 0.0  # samples/sec

    if all_event_metrics:
        avg_iou_agnostic = np.mean([m['iou_class_agnostic'] for m in all_event_metrics])
        avg_event_accuracy = np.mean([m['event_accuracy'] for m in all_event_metrics])* 100
    else:
        avg_iou_agnostic = 0.0
        avg_event_accuracy = 0.0

    return {
        'grid_accuracy': grid_accuracy,
        'avg_iou_agnostic': avg_iou_agnostic,
        'avg_event_accuracy': avg_event_accuracy,
        'per_sample_time': per_sample_time,
        'total_samples': total_samples,
        'total_inference_time': total_inference_time
    }