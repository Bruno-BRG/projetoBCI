import numpy as np
from mne.filter import filter_data


class EEGFilter:
    """
    EEGFilter provides bandpass filtering methods for streaming (LSL) and offline (calibration/real-use) EEG data.
    """
    def __init__(self, sfreq=250.0, l_freq=4.0, h_freq=50.0, method='iir', iir_params=None):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method
        # default to 4th-order Butterworth if not provided
        self.iir_params = iir_params or {'order': 4, 'ftype': 'butter'}

    def filter_stream(self, data):
        """
        Bandpass filter incoming LSL stream data.
        Expects data of shape (n_channels, n_samples).
        Returns filtered array of same shape.
        """
        return filter_data(
            data,
            sfreq=self.sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method=self.method,
            iir_params=self.iir_params,
            verbose=False
        )

    def filter_offline(self, data):
        """
        Bandpass filter data used for calibration and real-use widgets.
        Accepts arrays of shape (n_channels, n_samples) or (n_samples, n_channels).
        Returns filtered data with original shape.
        """
        arr = np.asarray(data)
        transpose_back = False
        # if shape is (n_samples, n_channels), transpose for filtering
        if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.T
            transpose_back = True

        filtered = filter_data(
            arr,
            sfreq=self.sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method=self.method,
            iir_params=self.iir_params,
            verbose=False
        )

        if transpose_back:
            filtered = filtered.T
        return filtered
