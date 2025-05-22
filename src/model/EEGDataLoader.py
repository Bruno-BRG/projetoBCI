import os
import numpy as np
import pandas as pd
from .EEGFilter import EEGFilter


# Constants for epoch extraction
SFREQ = 125  # OpenBCI sampling rate (Hz)
RUNS = [4, 8, 12]


def load_and_process_data(subject_id=2):
    """
    Load EEG data for a subject, extract epochs, and apply bandpass filter (4-50 Hz).
    Returns:
        X: ndarray of shape (n_epochs, n_channels, n_times)
        y: ndarray of labels (0=Left, 1=Right)
        ch_count: number of channels
    """
    subj_dir = os.path.join('eeg_data', 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0', f'S{subject_id:03d}')
    X_list, y_list = [], []

    for run in RUNS:
        csv_path = os.path.join(subj_dir, f'S{subject_id:03d}R{run:02d}_csv_openbci.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, comment='%', engine='python', on_bad_lines='skip')
        eeg_cols = [col for col in df.columns if col.startswith('EXG Channel')]
        data = df[eeg_cols].values.T  # shape (channels, samples)

        annotations = df['Annotations'].fillna('')
        event_indices, event_types = [], []
        for idx, ann in enumerate(annotations):
            if ann in ['T1', 'T2']:
                event_indices.append(idx)
                event_types.append(0 if ann=='T1' else 1)

        samples_per_epoch = int(3.1 * SFREQ)
        start_offset = int(1 * SFREQ)
        for evt_idx, evt_type in zip(event_indices, event_types):
            start = evt_idx + start_offset
            end = start + samples_per_epoch
            if end <= data.shape[1]:
                epoch = data[:, start:end]
                X_list.append(epoch)
                y_list.append(evt_type)

    if not X_list:
        raise ValueError(f"No valid epochs found for subject {subject_id}")

    X = np.stack(X_list)
    y = np.array(y_list)
    # apply bandpass filter to each epoch
    eeg_filter = EEGFilter(sfreq=SFREQ)
    X = np.array([eeg_filter.filter_offline(epoch) for epoch in X])
    ch_count = X.shape[1]
    return X, y, ch_count


def load_local_eeg_data(subject_id=1, augment=False):
    """
    Alias for load_and_process_data with optional data augmentation
    
    Args:
        subject_id: The subject ID to load data for
        augment: If True, apply data augmentation
        
    Returns:
        X, y, ch_count: EEG data, labels, and channel count
    """
    return load_and_process_data(subject_id)
