"""
EEG window quality checks used before runtime inference.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EEGQualityResult:
    accepted: bool
    reason: str = ""


class EEGWindowQualityValidator:
    def __init__(
        self,
        *,
        max_abs_amplitude: float = 5000.0,
        min_channel_std: float = 1e-6,
    ):
        if max_abs_amplitude <= 0:
            raise ValueError("max_abs_amplitude deve ser maior que zero.")
        if min_channel_std < 0:
            raise ValueError("min_channel_std nao pode ser negativo.")
        self.max_abs_amplitude = float(max_abs_amplitude)
        self.min_channel_std = float(min_channel_std)

    def validate(self, window: Sequence[Sequence[float]]) -> EEGQualityResult:
        data = np.asarray(window, dtype="float32")
        if data.size == 0:
            return EEGQualityResult(False, "empty_window")
        if data.ndim != 2:
            return EEGQualityResult(False, "invalid_shape")
        if not np.all(np.isfinite(data)):
            return EEGQualityResult(False, "non_finite")
        if float(np.max(np.abs(data))) > self.max_abs_amplitude:
            return EEGQualityResult(False, "amplitude_out_of_range")

        channel_std = np.std(data, axis=0)
        if np.any(channel_std <= self.min_channel_std):
            return EEGQualityResult(False, "flat_channel")

        return EEGQualityResult(True)
