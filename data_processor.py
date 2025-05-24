"""
EEG Data Processing Module
Handles loading and preprocessing of EEG data from CSV files (OpenBCI format)
"""

import pandas as pd
import numpy as np
import warnings
import os
import glob
from pathlib import Path

warnings.filterwarnings("ignore")


class EEGDataProcessor:
    """Class for processing EEG data from CSV files (OpenBCI format)"""
    
    def __init__(self):
        """Initialize the EEG data processor"""
        self.data_path = None
        
        # Dataset configuration
        self.N_SUBJECT = 109
        self.BASELINE_EYE_OPEN = [1]
        self.BASELINE_EYE_CLOSED = [2]
        self.OPEN_CLOSE_LEFT_RIGHT_FIST = [3, 7, 11]
        self.IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]
        self.OPEN_CLOSE_BOTH_FIST = [5, 9, 13]
        self.IMAGINE_OPEN_CLOSE_BOTH_FIST = [6, 10, 14]
        
    def set_data_path(self, path: str) -> None:
        """
        Set the data directory path
        
        Args:
            path: Path to the data directory containing CSV files
        """
        self.data_path = path
        
    def _load_csv_file(self, csv_file: str):
        """
        Load and process a single CSV file
        
        Args:
            csv_file: Path to the CSV file
            
        Returns:
            data: List of epoch data arrays
            labels: List of corresponding labels
        """
        try:
            # Read CSV file, skipping header lines
            df = pd.read_csv(csv_file, skiprows=4)  # Skip OpenBCI header
            
            # Extract EEG channels (columns 1-16)
            eeg_columns = [f'EXG Channel {i}' for i in range(16)]
            eeg_data = df[eeg_columns].values
            
            # Extract annotations
            annotations = df['Annotations'].fillna('')
            
            # Find event markers and their sample indices
            events = []
            for idx, annotation in enumerate(annotations):
                if annotation.startswith('T') and len(annotation) > 1:
                    try:
                        event_code = int(annotation[1:])
                        events.append((idx, event_code))
                    except ValueError:
                        continue
            
            # Create epochs based on events
            epoch_data = []
            epoch_labels = []
            
            sampling_rate = 125  # Hz
            epoch_duration = 4.0  # seconds
            epoch_samples = int(epoch_duration * sampling_rate)  # 500 samples
            
            for event_idx, event_code in events:
                # Define epoch start (1 second after event to avoid movement preparation)
                epoch_start = event_idx + sampling_rate  # 1 second after event
                epoch_end = epoch_start + epoch_samples
                
                # Check if we have enough data for the epoch
                if epoch_end <= len(eeg_data):
                    epoch = eeg_data[epoch_start:epoch_end]
                    
                    # Map event codes to motor imagery tasks
                    # T1 = left hand imagery, T2 = right hand imagery
                    # T0 = rest/baseline (we'll skip these for classification)
                    if event_code == 1:  # Left hand imagery
                        epoch_data.append(epoch.T)  # Transpose to (channels, time)
                        epoch_labels.append(0)  # 0 = left
                    elif event_code == 2:  # Right hand imagery
                        epoch_data.append(epoch.T)  # Transpose to (channels, time)
                        epoch_labels.append(1)  # 1 = right
                    # Skip T0 (rest) events
            
            return epoch_data, epoch_labels
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            return [], []    
    def load_data(self, n_subjects: int = 79):
        """
        Load and preprocess EEG data from CSV files
        
        Args:
            n_subjects: Number of subjects to load (1 to 79 or available)
            
        Returns:
            X: EEG data array (samples, channels, time_points)
            y: Labels (0=left, 1=right)
            info: Dictionary with dataset information
        """
        if self.data_path is None:
            # Default path to CSV files
            self.data_path = "eeg_data/MNE-eegbci-data/files/eegmmidb/1.0.0"
        
        # Motor imagery runs (R04, R08, R12 correspond to left vs right hand imagery)
        runs = ['R04', 'R08', 'R12']
        
        all_data = []
        all_labels = []
        
        for subject_id in range(1, min(n_subjects + 1, self.N_SUBJECT + 1)):
            subject_str = f"S{subject_id:03d}"
            
            for run in runs:
                csv_pattern = f"{self.data_path}/{subject_str}/{subject_str}{run}_csv_openbci.csv"
                csv_files = glob.glob(csv_pattern)
                
                if not csv_files:
                    print(f"Warning: No CSV file found for {subject_str}{run}")
                    continue
                
                csv_file = csv_files[0]
                
                try:
                    # Load CSV data
                    data, labels = self._load_csv_file(csv_file)
                    if data is not None and len(data) > 0:
                        all_data.extend(data)
                        all_labels.extend(labels)
                        
                except Exception as e:
                    print(f"Warning: Could not load {csv_file}: {e}")
                    continue
        
        if not all_data:
            raise ValueError("No data files could be loaded!")
        
        # Convert to numpy arrays
        X = np.array(all_data, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)
        
        # Prepare info dictionary
        info = {
            'n_subjects': n_subjects,
            'n_channels': 16,  # OpenBCI EXG channels
            'n_samples': X.shape[0],
            'n_timepoints': X.shape[2] if len(X.shape) > 2 else X.shape[1],
            'classes': ['left', 'right'],
            'sampling_rate': 125  # OpenBCI sampling rate
        }
        
        return X, y, info
    
    def preprocess_data(self, X: np.ndarray, filter_low: float = 8.0, filter_high: float = 30.0) -> np.ndarray:
        """
        Apply additional preprocessing to the data
        
        Args:
            X: EEG data array
            filter_low: Low-pass filter frequency
            filter_high: High-pass filter frequency
            
        Returns:
            Preprocessed data
        """
        # For CSV data, we could implement basic filtering here
        # For now, return as-is since the data is already in the right format
        return X
    
    def get_sample_data(self, X: np.ndarray, y: np.ndarray, n_samples: int = 100):
        """
        Get a subset of samples for quick testing
        
        Args:
            X: Full dataset
            y: Full labels
            n_samples: Number of samples to return
            
        Returns:
            Subset of X and y
        """
        if n_samples >= X.shape[0]:
            return X, y
            
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        return X[indices], y[indices]
