import argparse
import torch
from dataloader import get_dataloaders
from eval_utils import evaluate_model
from models.ours import ConSegAP as OursConSegAP
from models.ctnet import ConSegAP as CTNetConSegAP
from models.deepconvnet import ConSegAP as DeepConvNetConSegAP
from models.conformer import ConSegAP as ConformerConSegAP
from models.eegnet import ConSegAP as EEGNetConSegAP

def main():
    parser = argparse.ArgumentParser(description="Evaluate ConSegAP model on BCI IV-2a dataset")
    parser.add_argument('--data', type=str, required=True, choices=['2a', '2b'], help='need to specify the data type')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to standard_2a_data directory')
    parser.add_argument('--model', type=str, required=True, choices=['ours', 'CTNet', 'DeepConvNet', 'conformer', 'EEGNet'], help='Model type')
    parser.add_argument('--subject', type=int, required=True, help='Subject ID (1-9)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--min-duration-threshold', type=int, default=400, help='Minimum event duration in samples')

    args = parser.parse_args()

    # Determine break_dir based on data_dir
    if args.data == "2a":
        break_dir = args.data_dir.replace("standard_2a_data", "break_time_2a_data")
        model_params = {
        'ours': {'num_channels': 22, 'emb_size': 64, 'num_classes': 5},
        'CTNet': {'num_channels': 22, 'emb_size': 64, 'num_classes': 5},
        'DeepConvNet': {'num_channels': 22, 'emb_size': 64, 'feature_dim': 1400, 'num_classes': 5},
        'conformer': {'num_channels': 22, 'emb_size': 64, 'num_classes': 5},
        'EEGNet': {'num_channels': 22, 'emb_size': 64, 'feature_dim': 16, 'num_classes': 5}
        }
    elif args.data == "2b":
        break_dir = args.data_dir
        model_params = {
        'ours': {'num_channels': 3, 'emb_size': 64, 'num_classes': 3},
        'CTNet': {'num_channels': 3, 'emb_size': 64, 'num_classes': 3},
        'DeepConvNet': {'num_channels': 3, 'emb_size': 64, 'feature_dim': 1400, 'num_classes': 3},
        'conformer': {'num_channels': 3, 'emb_size': 64, 'num_classes': 3},
        'EEGNet': {'num_channels': 3, 'emb_size': 64, 'feature_dim': 16, 'num_classes': 3}
        }

    _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        break_dir=break_dir,
        subject=args.subject,
        batch_size=args.batch_size,
        data_type= args.data
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_classes = {
        'ours': OursConSegAP,
        'CTNet': CTNetConSegAP,
        'DeepConvNet': DeepConvNetConSegAP,
        'conformer': ConformerConSegAP,
        'EEGNet': EEGNetConSegAP
    }


    model_class = model_classes[args.model]
    model = model_class(**model_params[args.model]).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- Model Parameters for Subject {args.subject} ---")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Total Parameters (in millions): {total_params / 1e6:.2f} M")
    print('================================================')

    print(f"Evaluating {args.model} model for subject {args.subject}...")
    results = evaluate_model(model, test_loader, min_duration_threshold=args.min_duration_threshold)

    print("\n--- Overall Event Metrics ---")
    print(f"  Grid Accuracy: {results['grid_accuracy']:.2f}%")
    print(f"  IoU (Class-Agnostic): {results['avg_iou_agnostic']:.4f}")
    print(f"  Event Accuracy: {results['avg_event_accuracy']:.2f}%")
    print('================================================')

    print("\n--- Inference Performance ---")
    print(f"  Total samples processed: {results['total_samples']}")
    print(f"  Average inference time per sample: {results['per_sample_time']:.3f} s")
    print(f"  Total inference time: {results['total_inference_time']:.3f} seconds")
    print('================================================')

if __name__ == "__main__":
    main()