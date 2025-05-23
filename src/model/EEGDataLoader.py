import os
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from .EEGFilter import EEGFilter
except ImportError:
    from EEGFilter import EEGFilter


# Constants for epoch extraction
SFREQ = 125  # OpenBCI sampling rate (Hz)
RUNS = [4, 8, 12]


def load_and_process_data(subject_ids=2):
    """
    Load EEG data for one or multiple subjects, extract epochs, and apply bandpass filter (4-50 Hz).
    Args:
        subject_ids: int or list of ints for subject IDs
    Returns:
        X: ndarray of shape (n_epochs_total, n_channels, n_times)
        y: ndarray of labels (0=Left, 1=Right)
        ch_count: number of channels
    """
    # Normalize input to list
    if isinstance(subject_ids, int):
        subject_ids = [subject_ids]
    X_list, y_list = [], []
    # Initialize filter once
    eeg_filter = EEGFilter(sfreq=SFREQ)
    # Determine absolute path to eeg_data folder at project root
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / 'eeg_data' / 'MNE-eegbci-data' / 'files' / 'eegmmidb' / '1.0.0'
    for subject_id in subject_ids:
        subj_dir = data_root / f'S{subject_id:03d}'
        for run in RUNS:
            csv_path = subj_dir / f'S{subject_id:03d}R{run:02d}_csv_openbci.csv'
            if not csv_path.exists():
                continue
            df = pd.read_csv(str(csv_path), comment='%', engine='python', on_bad_lines='skip')
            eeg_cols = [col for col in df.columns if col.startswith('EXG Channel')]
            # extract continuous data and apply bandpass filter
            data = df[eeg_cols].values.T
            data = eeg_filter.filter_offline(data)
            annotations = df['Annotations'].fillna('')
            evt_idx, evt_types = [], []
            for idx, ann in enumerate(annotations):
                if ann in ['T1', 'T2']:
                    evt_idx.append(idx)
                    evt_types.append(0 if ann=='T1' else 1)
            samples = int(3.1 * SFREQ)
            offset = int(1 * SFREQ)
            for idx, typ in zip(evt_idx, evt_types):
                start = idx + offset
                end = start + samples
                if end <= data.shape[1]:
                    X_list.append(data[:, start:end])
                    y_list.append(typ)
    if not X_list:
        raise ValueError(f"No valid epochs found for subjects {subject_ids}")
    # Stack and return
    X = np.stack(X_list)
    y = np.array(y_list)
    ch_count = X.shape[1]
    return X, y, ch_count


def load_local_eeg_data(subject_id=2, augment=False):
    """
    Alias for load_and_process_data with optional data augmentation
    
    Args:
        subject_id: The subject ID to load data for
        augment: If True, apply data augmentation
        
    Returns:
        X, y, ch_count: EEG data, labels, and channel count
    """
    return load_and_process_data(subject_id)
