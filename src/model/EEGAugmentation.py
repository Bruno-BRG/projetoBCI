# Standard library imports
import os
import numpy as np
import pandas as pd

# Third-party imports
import torch
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

# Local imports
from .EEGClassificationModel import EEGClassificationModel

# Constants for electrode configurations
WANTED_CHANNELS = ['C3','C4','Fp1','Fp2','F7','F3','F4','F8',
                  'T7','T8','P7','P3','P4','P8','O1','O2']

class EEGAugmentation:
    """Advanced EEG data augmentation techniques"""
    augmentation_count = 5  # Number of augmented versions to create per sample

    @staticmethod
    def time_shift(data, max_shift=10):
        """Apply random time shift to the signal"""
        shifted_data = np.roll(data, np.random.randint(-max_shift, max_shift), axis=-1)
        return shifted_data
    
    @staticmethod
    def add_gaussian_noise(data, mean=0, std=0.1):
        """Add random Gaussian noise to the signal"""
        noise = np.random.normal(mean, std * np.std(data), data.shape)
        return data + noise
    
    @staticmethod
    def scale_amplitude(data, scale_range=(0.8, 1.2)):
        """Scale signal amplitude by random factor"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale
    
    @staticmethod
    def augment_data(data):
        """Apply all augmentations to create multiple versions of each sample"""
        augmented_samples = []
        # Keep original data
        augmented_samples.append(data)
        
        # For each original sample
        for i in range(len(data)):
            sample = data[i:i+1]  # Keep dimensions (1, channels, time)
            # Create augmented versions
            for _ in range(EEGAugmentation.augmentation_count):
                aug_sample = sample.copy()
                # Apply augmentations in sequence with different random values
                if np.random.random() > 0.3:  # 70% chance of applying each augmentation
                    aug_sample = EEGAugmentation.time_shift(aug_sample)
                if np.random.random() > 0.3:
                    aug_sample = EEGAugmentation.add_gaussian_noise(aug_sample)
                if np.random.random() > 0.3:
                    aug_sample = EEGAugmentation.scale_amplitude(aug_sample)
                augmented_samples.append(aug_sample)
        
        # Stack all samples together
        return np.concatenate(augmented_samples, axis=0)

def load_and_process_data(subject_id=1, augment=False):  # Changed default to False
    """Load EEG data from CSV files for imagined left/right hand movement and apply epoching"""
    # Find relevant run files (4,8,12 contain imagined left/right hand movement)
    runs = [4, 8, 12]
    subj_dir = os.path.join('eeg_data', 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0', f'S{subject_id:03d}')
    X_list, y_list = [], []

    for run in runs:
        csv_path = os.path.join(subj_dir, f'S{subject_id:03d}R{run:02d}_csv_openbci.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Required CSV file not found: {csv_path}")

        # Read CSV data
        df = pd.read_csv(csv_path, comment='%', engine='python', on_bad_lines='skip')
        eeg_cols = [col for col in df.columns if col.startswith('EXG Channel')]
        data = df[eeg_cols].values.T  # channels x samples

        # Find events from Annotations column
        # T1 marks the start of left hand trials
        # T2 marks the start of right hand trials
        annotations = df['Annotations'].fillna('')
        event_indices = []
        event_types = []

        for idx, annotation in enumerate(annotations):
            if (annotation in ['T1', 'T2']):
                event_indices.append(idx)
                # Convert T1->0 (left), T2->1 (right)
                event_types.append(0 if annotation == 'T1' else 1)

        # Extract epochs: 1s to 4.1s after each event
        sfreq = 125  # OpenBCI sample rate
        samples_per_epoch = int(3.1 * sfreq)  # 3.1s window (4.1s - 1s)
        start_offset = int(sfreq)  # 1s offset

        for evt_idx, evt_type in zip(event_indices, event_types):
            # Extract epoch
            start_idx = evt_idx + start_offset
            end_idx = start_idx + samples_per_epoch
            if end_idx <= data.shape[1]:  # Only use if we have enough samples
                epoch = data[:, start_idx:end_idx]
                X_list.append(epoch)
                y_list.append(evt_type)

    if not X_list:
        raise ValueError(f"No valid epochs found for subject {subject_id}")

    X = np.stack(X_list)
    y = np.array(y_list)
    
    if augment:
        # apply augmentation like previous version
        X = EEGAugmentation.augment_data(X)
        y = np.repeat(y, EEGAugmentation.augmentation_count+1)
    
    return X, y, X.shape[1]  # Return data, labels, and channel count

def load_local_eeg_data(subject_id=1, augment=False):
    """Load EEG data from all CSV files in project root (alias of load_and_process_data)"""
    return load_and_process_data(subject_id, augment)

def load_model(eeg_channel):
    model = EEGClassificationModel(eeg_channel=eeg_channel, dropout=0.125)
    # You can load pretrained weights here if needed
    return model
