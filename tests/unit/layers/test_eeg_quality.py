import numpy as np
import pytest

from brainbridge_v2.application.eeg_quality import EEGWindowQualityValidator


def test_eeg_quality_accepts_plausible_window():
    validator = EEGWindowQualityValidator(max_abs_amplitude=100.0)
    window = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]

    result = validator.validate(window)

    assert result.accepted is True
    assert result.reason == ""


@pytest.mark.parametrize(
    ("window", "reason"),
    [
        ([1.0, 2.0], "invalid_shape"),
        ([], "empty_window"),
        ([[1.0, float("nan")], [2.0, 3.0]], "non_finite"),
        ([[1.0, 2.0], [9999.0, 3.0]], "amplitude_out_of_range"),
        ([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]], "flat_channel"),
    ],
)
def test_eeg_quality_rejects_bad_windows(window, reason):
    validator = EEGWindowQualityValidator(max_abs_amplitude=100.0)

    result = validator.validate(window)

    assert result.accepted is False
    assert result.reason == reason


def test_eeg_quality_accepts_numpy_arrays():
    validator = EEGWindowQualityValidator(max_abs_amplitude=100.0)

    result = validator.validate(np.array([[1.0, 2.0], [2.0, 3.0]]))

    assert result.accepted is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_abs_amplitude": 0.0},
        {"min_channel_std": -1.0},
    ],
)
def test_eeg_quality_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        EEGWindowQualityValidator(**kwargs)
