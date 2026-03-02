import numpy as np


def test_butter_filter_shapes_and_rt():
    from brainbridge_v2.infrastructure.signal_processing.butter_filter import ButterworthFilter

    fs = 125.0
    filt = ButterworthFilter(lowcut=0.5, highcut=50.0, fs=fs, order=6)

    # 1D signal
    x = np.random.randn(1000)
    y = filt.apply_filter(x)
    assert y.shape == x.shape

    # 2D multichannel (channels, samples)
    X = np.random.randn(16, 1000)
    Y = filt.apply_filter(X)
    assert Y.shape == X.shape

    # Realtime for single sample (16 channels)
    filt.reset_filter_state()
    sample = np.random.randn(16)
    out = filt.apply_realtime_filter(sample)
    assert out.shape == sample.shape
