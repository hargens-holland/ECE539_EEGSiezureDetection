
import os
import numpy as np
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class CHBMITDataset(Dataset):
    
    def __init__(self, data_dir, window_size=4, overlap=0.5, target_fs=256):
        """
        Args
            data_dir: Path to data directory
            window_size: Window length in seconds
            overlap: Overlap fraction ( 0 to 1)
            target_fs: Target sampling frequency
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size #section of conintuous eeg file (4 seconds)
        self.overlap = overlap # over lap between windows (0-4, 2-6, 4-8 ...)
        self.target_fs = target_fs #256 Hz (256 times per second)
        self.samples_per_window = int(window_size * target_fs) # 4 * 256 
        self.stride = int(self.samples_per_window * (1 - overlap))
        
        # Load data and create index
        self.windows, self.labels = self._load_all_data()
    


    def _load_edf(self, edf_file, seizure_times):
        """takes edf file and sezizure times 
           returns 4 second windows labeled with seizure or not
        """
        # Read edf file
        #loads entire recording into memory
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        
        # preprocessing
        raw.filter(0.5, 40., fir_design='firwin', verbose=False) # filters out noise that aren't brain wave frequencies
        
        # resample to 256 Hz (some recording sessions use 512 Hz)
        raw.resample(self.target_fs, verbose=False) 
        
        #get data
        data = raw.get_data() # 23 x num samples (921600 for 1 hour)
        n_channels, n_samples = data.shape
        
        windows = []
        labels = []
        
        # Extract windows with overlap 
        for start in range(0, n_samples - self.samples_per_window, self.stride):
            end = start + self.samples_per_window
            window = data[:, start:end]
            
            # Normalize window
            window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)
            
            # Label window
            window_time = start / self.target_fs
            is_seizure = any(
                seizure_time <= window_time < seizure_time + self.window_size 
                for seizure_time in seizure_times
            )
            
            windows.append(window)
            labels.append(1 if is_seizure else 0)
        
        return windows, labels




    #loads edf files and extracts windows 
    def _load_all_data(self):
        windows = []
        labels = []
        
        # all folders in data directory (whatever is downloaded in drive)
        patient_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for patient_dir in patient_dirs:
            print(f"Loading {patient_dir.name}...")
            
            # Parse seizure annotations from summary file
            seizure_info = self._parse_summary(patient_dir)
            
            # Load each EDF (European data format) file
            # each edf has continuoes EEG recordings (1 hour)
            edf_files = sorted(patient_dir.glob("*.edf"))
            for edf_file in edf_files:
                file_windows, file_labels = self._load_edf(edf_file, seizure_info.get(edf_file.name, []))
                windows.extend(file_windows)
                labels.extend(file_labels)
        
        print(f"Total windows: {len(windows)}, Seizure: {sum(labels)}, Non-seizure: {len(labels)-sum(labels)}")
        
        #windows is a tensor (number of 4 second windows, channels (23), number of samples (1024))
        # label tells whether that window index contains seizure or not
        # windows [0][0][0] would be the first window of the first of the first sample (some number that represents the wave)
        return np.array(windows), np.array(labels)
    
    
    def _parse_summary(self, patient_dir):
        """Parse seizure times from summary file for labels """
        summary_file = patient_dir / f"{patient_dir.name}-summary.txt"
        seizure_info = {}
        
        if not summary_file.exists():
            return seizure_info
        
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        
        current_file = None
        for line in lines:
            if 'File Name:' in line:
                current_file = line.split(':')[1].strip()
                seizure_info[current_file] = []
            elif 'Seizure Start Time:' in line and current_file:
                start = int(line.split(':')[1].strip().split()[0])
                seizure_info[current_file].append(start)
        
        return seizure_info
    
    
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]), torch.LongTensor([self.labels[idx]])[0]


def get_dataloaders(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test dataloaders"""
    dataset = CHBMITDataset(data_dir)
    
    n = len(dataset)
    indices = np.random.permutation(n)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

