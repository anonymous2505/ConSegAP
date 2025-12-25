import h5py
import numpy as np
import os
import torch
import scipy.io

class DataReconstructor:
    def __init__(self, raw_data, raw_labels, num_segments, break_data, 
                 segment_length, event_length, num_channels, num_events_range=(1, 3)):
        """
        Data reconstructor to create 20-second continuous signals from event samples,
        using subject's break time data as background.
        
        Args:
            raw_data: Original data, (trials, 1, n_channels, event_length)
            raw_labels: Original labels, (trials,)
            num_segments: Number of 20-second segments to generate
            break_data: Break time data, (total_break_length, n_channels)
            segment_length: Target segment length (e.g., 5000 for 250 Hz, 20000 for 1000 Hz)
            event_length: Event length (e.g., 1000 for IV-2a/2b)
            num_channels: Number of channels (e.g., 3 for IV-2b, 22 for IV-2a)
            num_events_range: Range of number of events per segment
        """
        self.raw_data = raw_data
        self.raw_labels = raw_labels
        self.num_segments = num_segments
        self.break_data = break_data
        self.segment_length = segment_length
        self.event_length = event_length
        self.num_channels = num_channels
        self.num_events_range = num_events_range

    def sample_break_noise(self):
        """Randomly sample a 20-second background signal from break data"""
        total_samples = self.break_data.shape[0]
        if total_samples < self.segment_length:
            repeats = (self.segment_length // total_samples) + 1
            break_data_extended = np.tile(self.break_data, (repeats, 1))
        else:
            break_data_extended = self.break_data
        
        start_idx = np.random.randint(0, break_data_extended.shape[0] - self.segment_length + 1)
        noise = break_data_extended[start_idx:start_idx + self.segment_length, :].T
        return np.expand_dims(noise, axis=0)  # (1, n_channels, segment_length)

    def generate_segment(self):
        """Generate a 20-second signal with event annotations"""
        segment = np.zeros((1, self.num_channels, self.segment_length))
        event_list = []

        num_events = np.random.randint(self.num_events_range[0], self.num_events_range[1] + 1)
        available_positions = list(range(0, self.segment_length - self.event_length + 1))

        for _ in range(num_events):
            if not available_positions:
                break
            bs = np.random.choice(available_positions)
            event_idx = np.random.randint(0, self.raw_data.shape[0])
            event_data = self.raw_data[event_idx]  # (1, n_channels, event_length)
            event_class = self.raw_labels[event_idx]

            segment[:, :, bs:bs+self.event_length] = event_data.cpu()
            event_list.append({"class": event_class, "bs": bs, "bd": self.event_length})

            # Avoid overlap
            overlap_margin = self.event_length
            used_range = range(max(0, bs - overlap_margin), min(self.segment_length, bs + self.event_length + overlap_margin))
            available_positions = [p for p in available_positions if p not in used_range]

        noise = self.sample_break_noise()
        mask = np.ones_like(segment)
        for event in event_list:
            mask[:, :, event["bs"]:event["bs"]+event["bd"]] = 0
        segment = segment + mask * noise

        return segment, event_list

    def reconstruct(self):
        """Generate all 20-second segments"""
        segments = []
        annotations = []
        for _ in range(self.num_segments):
            segment, events = self.generate_segment()
            segments.append(segment)
            annotations.append(events)
        segments = np.concatenate(segments)  # (num_segments, 1, n_channels, segment_length)
        return segments, annotations

def annotations_to_events(annotations, sampling_rate):
    """Convert annotations to event labels"""
    total = len(annotations)
    max_events = max(len(events) for events in annotations)
    events_tensor = torch.zeros(total, max_events, 3, dtype=torch.float)

    for i, events in enumerate(annotations):
        for e, event in enumerate(events):
            start_sample = event["bs"]
            duration = event["bd"]
            event_class = event["class"]

            start_time = (start_sample / sampling_rate) * sampling_rate
            end_time = ((start_sample + duration) / sampling_rate) * sampling_rate
            events_tensor[i, e, 0] = start_time
            events_tensor[i, e, 1] = end_time
            events_tensor[i, e, 2] = event_class + 1  # 1-based indexing

    return events_tensor

class BCIDataLoader:
    def __init__(self, dataset_type, nsub, data_dir, break_dir=None, 
                 train_ratio=0.8, num_train_segments_factor=3, num_test_segments_factor=0.5,
                 number_events_range=(1, 3), number_aug=2, number_seg=8, segment_length = 5000):
        """
        DataLoader for BCI IV-2a, IV-2b datasets, generating 20-second signals.
        
        Args:
            dataset_type (str): 'IV-2a', 'IV-2b'
            nsub (int or str): Subject number (1-9 for IV-2a/2b)
            data_dir (str): Directory containing data files
            break_dir (str, optional): Directory containing break time data
            train_ratio (float): Ratio of training data
            num_train_segments_factor (float): Factor for number of training segments
            num_test_segments_factor (float): Factor for number of test segments
            number_events_range (tuple): Range of events per segment
            number_aug (int): Number of augmentations per class
            number_seg (int): Number of segments for augmentation
        """
        self.dataset_type = dataset_type
        self.nSub = nsub
        self.root = data_dir
        self.break_dir = break_dir or data_dir
        self.train_ratio = train_ratio
        self.num_train_segments_factor = num_train_segments_factor
        self.num_test_segments_factor = num_test_segments_factor
        self.number_events_range = number_events_range
        self.number_augmentation = number_aug
        self.number_seg = number_seg
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.segment_length = segment_length

        # Set dataset-specific parameters
        if dataset_type == 'IV-2a':
            self.number_class = 4
            self.number_channel = 22
            self.sampling_rate = 250
            # self.segment_length = 5000  # 20s at 250 Hz
            self.event_length = 1000    # 4s at 250 Hz
        elif dataset_type == 'IV-2b':
            self.number_class = 2
            self.number_channel = 3
            self.sampling_rate = 250
            # self.segment_length = 5000  # 20s at 250 Hz
            self.event_length = 1000    # 4s at 250 Hz
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def load_break_data(self, session_type=None):
        """Load break time data"""
        if self.dataset_type in ['IV-2a', 'IV-2b']:
            file_path = os.path.join(
                self.break_dir,
                f"{'A0' if self.dataset_type == 'IV-2a' else 'B0'}{self.nSub}{f'{session_type}_break_preprocessed.mat' if self.dataset_type == 'IV-2a' else f'{session_type}.mat'}"
            )
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Break time file {file_path} not found.")
            if self.dataset_type == 'IV-2a':
                with h5py.File(file_path, 'r') as f:
                    break_data_ref = f['break_data']
                    refs = break_data_ref[0] 
                    break_segments = [f[ref][()].T for ref in refs]  
                    break_data = np.concatenate(break_segments, axis=0)
                    print(f"Final break_data shape for A0{self.nSub}{session_type}: {break_data.shape}")
            else:
                mat = scipy.io.loadmat(file_path)
                break_data = mat['break']  # (n_break_trials, n_channels, 1000)
                break_data = break_data.transpose(0, 2, 1).reshape(-1, self.number_channel)  # (n_break_trials*1000, n_channels)

                print(f"Break data shape for {'A0' if self.dataset_type == 'IV-2a' else 'B0'}{self.nSub}{session_type}: {break_data.shape}")
        return break_data

    def interaug(self, timg, label):
        """Data augmentation by combining segments from same class"""
        aug_data = []
        aug_label = []
        number_records_by_augmentation = self.number_augmentation * int(72 / self.number_class)
        number_segmentation_points = self.sampling_rate // self.number_seg
        
        for clsAug in range(self.number_class):
            cls_idx = np.where(label == clsAug)[0]
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            
            if tmp_data.shape[0] == 0:
                print(f"Warning: No samples for class {clsAug} in augmentation, skipping.")
                continue
            
            tmp_aug_data = np.zeros((number_records_by_augmentation, 1, self.number_channel, 
                                   self.event_length))
            for ri in range(number_records_by_augmentation):
                for rj in range(self.number_seg):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 1)[0]
                    tmp_aug_data[ri, :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points] = \
                        tmp_data[rand_idx, :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:number_records_by_augmentation])
        
        if not aug_data:
            print("Warning: No augmented data generated, returning empty arrays.")
            return torch.zeros((0, 1, self.number_channel, self.event_length)).cuda().float(), \
                   torch.zeros(0).cuda().long()
        
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda().float()
        aug_label = torch.from_numpy(aug_label).cuda().long()
        return aug_data, aug_label

    def load_source_data(self):
        """Load and preprocess source data"""
        if self.dataset_type in ['IV-2a', 'IV-2b']:
            # Load training data
            train_file = os.path.join(self.root, f"{'A0' if self.dataset_type == 'IV-2a' else 'B0'}{self.nSub}T.mat")
            train_mat = scipy.io.loadmat(train_file)
            train_data = train_mat['data']  # (trials, n_channels, timepoints)
            train_label = train_mat['label'].flatten() - 1  # (trials,)
            if self.dataset_type == 'IV-2a':
                train_data = np.transpose(train_data, (2, 1, 0))
            # Load test data
            test_file = os.path.join(self.root, f"{'A0' if self.dataset_type == 'IV-2a' else 'B0'}{self.nSub}E.mat")
            test_mat = scipy.io.loadmat(test_file)
            test_data = test_mat['data']  # (trials, n_channels, timepoints)
            if self.dataset_type == 'IV-2a':
                test_data = np.transpose(test_data, (2, 1, 0))
            test_label = test_mat['label'].flatten() - 1  # (trials,)
            
            train_data = np.expand_dims(train_data, axis=1)    # (trials, 1, timepoints, n_channels)
            test_data = np.expand_dims(test_data, axis=1)

            train_break_data = self.load_break_data('T')
            test_break_data = self.load_break_data('E')

        return train_data, train_label, test_data, test_label, train_break_data, test_break_data

    def make_data(self):
        """Prepare training and test data with augmentation"""
        train_data, train_label, test_data, test_label, train_break_data, test_break_data = self.load_source_data()
        
        train_data = torch.from_numpy(train_data).type(torch.float)
        train_label = torch.from_numpy(train_label).type(torch.long)
        test_data = torch.from_numpy(test_data).type(torch.float)
        test_label = torch.from_numpy(test_label).type(torch.long)
        
        # Data augmentation
        aug_data, aug_label = self.interaug(train_data, train_label)
        
        train_data = torch.cat((train_data.cuda(), aug_data)).type(self.Tensor).cuda()
        train_label = torch.cat((train_label.cuda(), aug_label)).type(self.LongTensor).cuda()
        
        test_data = test_data.type(self.Tensor).cuda()
        test_label = test_label.type(self.LongTensor).cuda()
        
        return train_data, train_label, test_data, test_label, train_break_data, test_break_data

    def load_data(self):
        """Load and reconstruct data for training and testing"""
        train_data, train_label, test_data, test_label, train_break_data, test_break_data = self.make_data()
        
        # Reconstruct training data
        train_reconstructor = DataReconstructor(
            raw_data=train_data,
            raw_labels=train_label,
            num_segments=int(len(train_data) * self.num_train_segments_factor),
            break_data=train_break_data,
            segment_length=self.segment_length,
            event_length=self.event_length,
            num_channels=self.number_channel,
            num_events_range=self.number_events_range
        )
        train_segments, train_annotations = train_reconstructor.reconstruct()
        train_segments = torch.from_numpy(train_segments).type(self.Tensor).cuda()
        
        # Reconstruct test data
        test_reconstructor = DataReconstructor(
            raw_data=test_data,
            raw_labels=test_label,
            num_segments=int(len(test_data) * self.num_test_segments_factor),
            break_data=test_break_data,
            segment_length=self.segment_length,
            event_length=self.event_length,
            num_channels=self.number_channel,
            num_events_range=self.number_events_range
        )
        test_segments, test_annotations = test_reconstructor.reconstruct()
        test_segments = torch.from_numpy(test_segments).type(self.Tensor).cuda()
        
        # Convert annotations to events
        train_events = annotations_to_events(train_annotations, self.sampling_rate)
        test_events = annotations_to_events(test_annotations, self.sampling_rate)
        
        train_events = train_events.type(self.Tensor).cuda()
        test_events = test_events.type(self.Tensor).cuda()
        
        return train_segments, train_events, test_segments, test_events

# # Example usage
# if __name__ == "__main__":

#     loader = BCIDataLoader(
#         dataset_type='IV-2a',
#         nsub=1,
#         data_dir="./data/4_2a_data/standard_2a_data",
#         break_dir="./data/4_2a_data/break_time_2a_data",
#         train_ratio=0.8,
#         num_train_segments_factor=3,
#         num_test_segments_factor=0.5,
#         number_events_range=(1, 3),
#         number_aug=2,
#         number_seg=8,
#         segment_length=5000
#     )
#     train_segments, train_events, test_segments, test_events = loader.load_data()
#     print(f"Train segments: {train_segments.shape}")
#     print(f"Train events: {train_events.shape}")
#     print(f"Test segments: {test_segments.shape}")
#     print(f"Test events: {test_events.shape}")
