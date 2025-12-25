import torch
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def align_labels(events, grid_sizes):
    """
    Align event-based labels to model patches based on grid_sizes.

    Args:
        events: [batch, num_events, 3], event labels [start_time, end_time, class]
        grid_sizes: list of [batch, 100], dynamic grid sizes (ms)

    Returns:
        aligned_labels: [batch, 100], aligned labels
    """
    batch_size = events.shape[0]
    num_patches = len(grid_sizes[0])  # 100
    aligned_labels = torch.zeros(batch_size, num_patches, dtype=torch.long)

    for b in range(batch_size):
        model_times = np.cumsum([0] + grid_sizes[b])
        event_list = events[b]
        event_list = event_list[event_list[:, 2] != 0]

        for i in range(num_patches):
            model_start = model_times[i]
            model_end = model_times[i + 1]
            max_overlap = 0
            selected_class = 0

            for e in range(event_list.shape[0]):
                event_start = event_list[e, 0].item()
                event_end = event_list[e, 1].item()
                event_class = int(event_list[e, 2].item())
                overlap_start = max(model_start, event_start)
                overlap_end = min(model_end, event_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    selected_class = event_class

            aligned_labels[b, i] = selected_class

    return aligned_labels

def extract_events(labels, grid_sizes):
    """
    Extract event blocks from labels and convert to time ranges.

    Args:
        labels: [batch, num_patches], class labels (0-4)
        grid_sizes: list, each element is [num_patches] window sizes

    Returns:
        events: list, each element is [[start, end, class], ...]
    """
    batch_size, num_patches = labels.shape
    events = []

    for b in range(batch_size):
        batch_labels = labels[b].cpu().numpy()
        batch_grid_sizes = grid_sizes[b]
        time_points = np.cumsum([0] + batch_grid_sizes)
        batch_events = []
        i = 0
        while i < num_patches:
            if batch_labels[i] == 0:
                i += 1
                continue
            start_idx = i
            event_class = batch_labels[i]
            while i < num_patches and batch_labels[i] == event_class:
                i += 1
            end_idx = i
            start_time = time_points[start_idx]
            end_time = time_points[end_idx]
            batch_events.append([start_time, end_time, event_class])
        events.append(batch_events)

    return events

def focal_loss(logits, labels, alpha=None, gamma=2.0, reduction='mean'):
    """
    Focal Loss implementation.

    Args:
        logits: [N, C], unnormalized model outputs
        labels: [N], target labels
        alpha: [C], class weights (optional)
        gamma: modulation factor for hard/easy samples
        reduction: 'mean' or 'sum'

    Returns:
        Loss value
    """
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    probs = F.softmax(logits, dim=-1)
    p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    focal_weight = (1 - p_t) ** gamma
    loss = focal_weight * ce_loss

    if alpha is not None:
        alpha_t = alpha[labels]
        loss = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def compute_iou_loss(pred_events, true_events):
    """
    Compute IoU Loss between predicted and true event blocks.

    Args:
        pred_events: list, each element is [[start, end, class], ...]
        true_events: list, each element is [[start, end, class], ...]

    Returns:
        iou_loss: scalar, IoU Loss
    """
    batch_size = len(pred_events)
    total_iou = 0.0
    total_pairs = 0

    for b in range(batch_size):
        preds = pred_events[b]
        trues = true_events[b]

        if not trues or not preds:
            continue

        for true in trues:
            true_start, true_end, true_class = true
            best_iou = 0.0
            for pred in preds:
                pred_start, pred_end, pred_class = pred
                if pred_class != true_class:
                    continue
                inter_start = max(true_start, pred_start)
                inter_end = min(true_end, pred_end)
                intersection = max(0, inter_end - inter_start)
                union = (true_end - true_start) + (pred_end - pred_start) - intersection
                iou = intersection / union if union > 0 else 0.0
                best_iou = max(best_iou, iou)
            total_iou += best_iou
            total_pairs += 1

    if total_pairs == 0:
        return torch.tensor(0.0, device=device)

    avg_iou = total_iou / total_pairs
    iou_loss = 1.0 - avg_iou
    return torch.tensor(iou_loss, device=device)
