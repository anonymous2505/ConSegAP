# Example usage
import h5py
from dataloader_test import BCIDataLoader

if __name__ == "__main__":

    loader = BCIDataLoader(
        dataset_type='IV-2a',
        nsub=1,
        data_dir="./data/4_2a_data/standard_2a_data",
        break_dir="./data/4_2a_data/break_time_2a_data",
        train_ratio=0.8,
        num_train_segments_factor=3,
        num_test_segments_factor=0.5,
        number_events_range=(3, 9),
        number_aug=2,
        number_seg=8,
        segment_length=15000
    )
    train_segments, train_events, test_segments, test_events = loader.load_data()
    print(f"Train segments: {train_segments.shape}")
    print(f"Train events: {train_events.shape}")
    print(f"Test segments: {test_segments.shape}")
    print(f"Test events: {test_events.shape}")


    test_segments_np = test_segments.cpu().numpy()  
    test_events_np = test_events.cpu().numpy()

    output_file = "./data/Online_exp_data/test_data.h5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("test_segments", data=test_segments_np, compression="gzip", compression_opts=4)
        f.create_dataset("test_events", data=test_events_np, compression="gzip", compression_opts=4)
        f.attrs["description"] = "Test data for EEG experiment"
        f.attrs["test_segments_shape"] = test_segments_np.shape
        f.attrs["test_events_shape"] = test_events_np.shape

    print(f"Data saved to {output_file}")
