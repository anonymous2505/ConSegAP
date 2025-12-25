import argparse
import torch
import torch.optim as optim
from dataloader import get_dataloaders
from train_utils import train_model, evaluate_model

# Import model classes
from models.ours import ConSegAP as OursConSegAP
from models.ctnet import ConSegAP as CTNetConSegAP
from models.deepconvnet import ConSegAP as DeepConvNetConSegAP
from models.conformer import ConSegAP as ConformerConSegAP
from models.eegnet import ConSegAP as EEGNetConSegAP

def main():
    parser = argparse.ArgumentParser(description="Train ConSegAP model on BCI IV-2a/b dataset")
    parser.add_argument('--data', type=str, required=True, choices=['2a', '2b'], help='need to specify the data type')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to standard_2a_data directory')
    parser.add_argument('--model', type=str, required=True, choices=['ours', 'CTNet', 'DeepConvNet', 'conformer', 'EEGNet'], help='Model type')
    parser.add_argument('--subject', type=int, required=True, help='Subject ID (1-9)')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
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
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        break_dir=break_dir,
        subject=args.subject,
        batch_size=args.batch_size,
        data_type= args.data
    )
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model based on --model argument
    model_classes = {
        'ours': OursConSegAP,
        'CTNet': CTNetConSegAP,
        'DeepConvNet': DeepConvNetConSegAP,
        'conformer': ConformerConSegAP,
        'EEGNet': EEGNetConSegAP
    }
    
    
    model_class = model_classes[args.model]
    model = model_class(**model_params[args.model]).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # Train and evaluate
    print(f"Training model for subject {args.subject}...(use {args.model})")
    train_model(model, train_loader, optimizer, num_epochs=args.num_epochs, sub=args.subject, data_type=args.data)
    
    print("Evaluating model...")
    report_metrics = args.model in ['CTNet', 'DeepConvNet', 'conformer', 'EEGNet']
    evaluate_model(model, test_loader, report_metrics=report_metrics)

if __name__ == "__main__":
    main()
