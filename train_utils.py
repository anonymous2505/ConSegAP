import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from utils import align_labels, extract_events, focal_loss, compute_iou_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, optimizer, num_epochs=10, sub=1, data_type='2a'):
    model.train()
    if data_type == '2a':
        class_weights = torch.tensor([0.7, 1.0, 1.0, 1.0, 1.0]).to(device)
        num_classes = 5
    elif data_type == '2b':
        class_weights = torch.tensor([0.7, 1.0, 1.0]).to(device)
        num_classes = 3
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_events in train_loader:
            batch_x = batch_x.squeeze(1).to(device)
            batch_events = batch_events.to(device)
            optimizer.zero_grad()
            
            normal_context_logits, extended_context_logits, grid_sizes, activity = model(batch_x)
            
            aligned_labels = align_labels(batch_events, grid_sizes).to(device)
            
            short_cls_loss = focal_loss(normal_context_logits.view(-1, num_classes), aligned_labels.view(-1), alpha=class_weights, gamma=2.0)
            long_cls_loss = focal_loss(extended_context_logits.view(-1, num_classes), aligned_labels.view(-1), alpha=class_weights, gamma=2.0)
            cls_loss = 0.5 * short_cls_loss + 0.5 * long_cls_loss
            
            short_preds = torch.argmax(normal_context_logits, dim=-1)
            long_preds = torch.argmax(extended_context_logits, dim=-1)
            consistency_loss = F.cross_entropy(normal_context_logits.view(-1, num_classes), long_preds.view(-1), reduction='mean') + \
                              F.cross_entropy(extended_context_logits.view(-1, num_classes), short_preds.view(-1), reduction='mean')
            
            preds = torch.argmax(extended_context_logits, dim=-1)
            pred_events = extract_events(preds, grid_sizes)
            true_events = extract_events(aligned_labels, grid_sizes)
            iou_loss = compute_iou_loss(pred_events, true_events)
            
            loss = 1.5 * cls_loss + 0.5 * consistency_loss + 0.1 * iou_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'best_model_{sub}.pth')
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
                  f"Cls Loss: {cls_loss.item():.4f}, "
                  f"Consistency Loss: {consistency_loss.item():.4f}, IoU Loss: {iou_loss.item():.4f}")
            print(f"Saved best model_{sub} at epoch {epoch+1} with loss {best_loss:.4f}")
            print('=====================================')
    
    torch.save(model.state_dict(), f'final_model_{sub}.pth')
    print("Training completed. Final model saved as 'final_model.pth'")

def evaluate_model(model, test_loader, report_metrics=False):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_grid_sizes = []
    all_signals = []
    all_events = []
    
    
    with torch.no_grad():
        for batch_x, batch_events in test_loader:
            batch_x = batch_x.squeeze(1).to(device)
            batch_events = batch_events.to(device)
            normal_context_logits, extended_context_logits, grid_sizes_batch, _ = model(batch_x)
            
            aligned_labels = align_labels(batch_events, grid_sizes_batch).to(device)
            preds = torch.argmax(normal_context_logits, dim=-1)
            correct += (preds == aligned_labels).sum().item()
            total += aligned_labels.numel()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(aligned_labels.cpu().numpy())
            all_grid_sizes.extend(grid_sizes_batch)
            all_signals.append(batch_x.cpu().numpy())
            all_events.extend(batch_events)
    
    accuracy = correct / total * 100
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    
    if report_metrics:
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2, 3, 4])
        for cls in range(5):
            print(f"Class {cls}: Precision={precision[cls]:.2f}, Recall={recall[cls]:.2f}, F1={f1[cls]:.2f}")
    
    return accuracy
