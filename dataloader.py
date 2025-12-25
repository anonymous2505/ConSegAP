from torch.utils.data import DataLoader, TensorDataset
from dataloader_test import BCIDataLoader

def get_dataloaders(data_dir, break_dir, subject, batch_size, train_ratio=0.8, data_type='2a'):
    if data_type == '2a':
        loader = BCIDataLoader(
            dataset_type='IV-2a',
            nsub=subject,
            data_dir=data_dir,
            break_dir=break_dir,
            train_ratio=train_ratio,
            num_train_segments_factor=3,
            num_test_segments_factor=0.5,
            number_events_range=(1, 3),
            number_aug=2,
            number_seg=8
        )
    else:
        loader = BCIDataLoader(
            dataset_type='IV-2b',
            nsub=subject,
            data_dir=data_dir,
            break_dir=break_dir,
            train_ratio=train_ratio,
            num_train_segments_factor=3,
            num_test_segments_factor=0.5,
            number_events_range=(1, 3),
            number_aug=2,
            number_seg=8
        )
    train_segments, train_labels, test_segments, test_labels = loader.load_data()
    
    train_dataset = TensorDataset(train_segments, train_labels)
    test_dataset = TensorDataset(test_segments, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
