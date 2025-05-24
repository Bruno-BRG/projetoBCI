"""
EEG Data Processing Module
Handles loading and preprocessing of EEG data from the PhysioNet dataset
"""

import mne
from mne.io import concatenate_raws
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")


class EEGDataProcessor:
    """Class for processing EEG data from PhysioNet dataset"""
    
    def __init__(self):
        self.data_path = None
        
        # Dataset configuration
        self.N_SUBJECT = 109
        self.BASELINE_EYE_OPEN = [1]
        self.BASELINE_EYE_CLOSED = [2]
        self.OPEN_CLOSE_LEFT_RIGHT_FIST = [3, 7, 11]
        self.IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]
        self.OPEN_CLOSE_BOTH_FIST = [5, 9, 13]
        self.IMAGINE_OPEN_CLOSE_BOTH_FIST = [6, 10, 14]
        
    def set_data_path(self, path):
        """Set the data directory path"""
        self.data_path = path
        
    def load_data(self, n_subjects=79):
        """
        Load and preprocess EEG data
        
        Args:
            n_subjects: Number of subjects to load (1 to 109)
            
        Returns:
            X: EEG data array (samples, channels, time_points)
            y: Labels (0=left, 1=right)
            info: Dictionary with dataset information
        """
        if self.data_path is None:
            # Use default MNE data path if not specified
            physionet_paths = [
                mne.datasets.eegbci.load_data(
                    subject_id,
                    runs=self.IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,
                )
                for subject_id in range(1, n_subjects + 1)
            ]
        else:
            # Use specified data path
            physionet_paths = [
                mne.datasets.eegbci.load_data(
                    subject_id,
                    runs=self.IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,
                    path=self.data_path,
                )
                for subject_id in range(1, n_subjects + 1)
            ]
            
        # Flatten the list of file paths
        physionet_paths = np.concatenate(physionet_paths)
        
        # Load raw data from all files
        parts = []
        for path in physionet_paths:
            try:
                raw = mne.io.read_raw_edf(
                    path,
                    preload=True,
                    stim_channel='auto',
                    verbose='WARNING',
                )
                parts.append(raw)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                continue
                
        if not parts:
            raise ValueError("No data files could be loaded!")
            
        # Concatenate all raw data
        raw = concatenate_raws(parts)
        
        # Extract events from annotations
        events, _ = mne.events_from_annotations(raw)
        
        # Get EEG channel indices
        eeg_channel_inds = mne.pick_types(
            raw.info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            exclude='bads',
        )
        
        EEG_CHANNEL = int(len(eeg_channel_inds))
        
        # Create epochs for motor imagery tasks
        epoched = mne.Epochs(
            raw,
            events,
            dict(left=2, right=3),  # Event codes for left and right hand imagery
            tmin=1,  # Start 1 second after event
            tmax=4.1,  # End 4.1 seconds after event
            proj=False,
            picks=eeg_channel_inds,
            baseline=None,
            preload=True
        )
        
        # Convert to numpy arrays
        X = (epoched.get_data() * 1e3).astype(np.float32)  # Convert to mV
        y = (epoched.events[:, 2] - 2).astype(np.int64)  # 0=left, 1=right
        
        # Clean up memory
        del raw, events, epoched, physionet_paths, parts
        
        # Prepare info dictionary
        info = {
            'n_subjects': n_subjects,
            'n_channels': EEG_CHANNEL,
            'n_samples': X.shape[0],
            'n_timepoints': X.shape[2],
            'classes': ['left', 'right'],
            'sampling_rate': 160  # PhysioNet EEG dataset sampling rate
        }
        
        return X, y, info
    
    def preprocess_data(self, X, filter_low=8.0, filter_high=30.0):
        """
        Apply additional preprocessing to the data
        
        Args:
            X: EEG data array
            filter_low: Low-pass filter frequency
            filter_high: High-pass filter frequency
            
        Returns:
            Preprocessed data
        """
        # This could include additional filtering, artifact removal, etc.
        # For now, return as-is since MNE handles basic preprocessing
        return X
    
    def get_sample_data(self, X, y, n_samples=100):
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
