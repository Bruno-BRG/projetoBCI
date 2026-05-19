"""
Game inference window coordination.

This module keeps the timing/buffering rule independent from the PyQt UI:
after a task marker is sent to Unity, collect exactly a fresh EEG window before
allowing one inference response.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence


@dataclass(frozen=True)
class InferenceWindowResult:
    status: str
    samples_collected: int
    window: Optional[Sequence[Sequence[float]]] = None


class GameInferenceCoordinator:
    """
    Coordinates game-mode inference windows.

    A prediction is allowed only after `window_size` new samples are collected
    after `start_window()` is called.
    """

    STATUS_INACTIVE = "inactive"
    STATUS_LOCKED = "locked"
    STATUS_COLLECTING = "collecting"
    STATUS_READY = "ready"
    STATUS_EXPIRED = "expired"

    def __init__(
        self,
        *,
        window_size: int = 250,
        channels: int = 16,
        window_duration_ms: int = 2000,
    ):
        if window_size <= 0:
            raise ValueError("window_size deve ser maior que zero.")
        if channels <= 0:
            raise ValueError("channels deve ser maior que zero.")
        if window_duration_ms <= 0:
            raise ValueError("window_duration_ms deve ser maior que zero.")

        self.window_size = int(window_size)
        self.channels = int(channels)
        self.window_duration_ms = int(window_duration_ms)
        self.eeg_buffer: Deque[list[float]] = deque(maxlen=self.window_size)
        self.samples_since_window_start = 0
        self.window_started_at_ms: Optional[float] = None
        self.is_window_open = False
        self.prediction_locked = False

    def reset(self) -> None:
        self.eeg_buffer.clear()
        self.samples_since_window_start = 0
        self.window_started_at_ms = None
        self.is_window_open = False
        self.prediction_locked = False

    def configure(
        self,
        *,
        window_size: Optional[int] = None,
        channels: Optional[int] = None,
        window_duration_ms: Optional[int] = None,
    ) -> None:
        if window_size is not None and int(window_size) != self.window_size:
            if int(window_size) <= 0:
                raise ValueError("window_size deve ser maior que zero.")
            self.window_size = int(window_size)
            self.eeg_buffer = deque(maxlen=self.window_size)
        if channels is not None:
            if int(channels) <= 0:
                raise ValueError("channels deve ser maior que zero.")
            self.channels = int(channels)
        if window_duration_ms is not None:
            if int(window_duration_ms) <= 0:
                raise ValueError("window_duration_ms deve ser maior que zero.")
            self.window_duration_ms = int(window_duration_ms)
        self.reset()

    def start_window(self, started_at_ms: Optional[float] = None) -> None:
        self.eeg_buffer.clear()
        self.samples_since_window_start = 0
        self.window_started_at_ms = float(started_at_ms) if started_at_ms is not None else None
        self.is_window_open = True
        self.prediction_locked = False

    def close_window(self) -> None:
        self.is_window_open = False

    def mark_prediction_used(self) -> None:
        self.prediction_locked = True

    def add_sample(
        self,
        sample: Sequence[float],
        *,
        now_ms: Optional[float] = None,
    ) -> InferenceWindowResult:
        if not self.is_window_open:
            return self._result(self.STATUS_INACTIVE)
        if self.prediction_locked:
            return self._result(self.STATUS_LOCKED)
        if self._is_expired(now_ms):
            self.close_window()
            return self._result(self.STATUS_EXPIRED)

        self.eeg_buffer.append(self._normalize_sample(sample))
        self.samples_since_window_start += 1

        if self.samples_since_window_start >= self.window_size:
            window = list(self.eeg_buffer)[-self.window_size :]
            self.samples_since_window_start = 0
            return self._result(self.STATUS_READY, window=window)

        return self._result(self.STATUS_COLLECTING)

    def _is_expired(self, now_ms: Optional[float]) -> bool:
        if self.window_started_at_ms is None or now_ms is None:
            return False
        return float(now_ms) - self.window_started_at_ms > self.window_duration_ms

    def _normalize_sample(self, sample: Sequence[float]) -> list[float]:
        values = sample.tolist() if hasattr(sample, "tolist") else list(sample)
        if len(values) >= self.channels:
            return [float(value) for value in values[: self.channels]]
        return [float(value) for value in values] + [0.0] * (self.channels - len(values))

    def _result(
        self,
        status: str,
        *,
        window: Optional[Sequence[Sequence[float]]] = None,
    ) -> InferenceWindowResult:
        return InferenceWindowResult(
            status=status,
            samples_collected=self.samples_since_window_start,
            window=window,
        )
