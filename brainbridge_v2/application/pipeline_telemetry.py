"""
Telemetry helpers for the EEG -> inference -> Unity pipeline.
"""

from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import Any, Deque, Optional


logger = logging.getLogger("brainbridge.pipeline")


@dataclass(frozen=True)
class PipelineEvent:
    name: str
    timestamp_ms: float
    details: dict[str, Any]


class SampleRateMonitor:
    """Tracks sample rate over fixed time windows."""

    def __init__(self, *, interval_seconds: float = 1.0):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds deve ser maior que zero.")
        self.interval_seconds = float(interval_seconds)
        self.window_start_seconds: Optional[float] = None
        self.samples_in_window = 0
        self.latest_rate_hz = 0.0

    def observe(self, *, now_seconds: Optional[float] = None) -> Optional[float]:
        now = time.time() if now_seconds is None else float(now_seconds)
        if self.window_start_seconds is None:
            self.window_start_seconds = now
            self.samples_in_window = 1
            return None

        self.samples_in_window += 1
        elapsed = now - self.window_start_seconds
        if elapsed < self.interval_seconds:
            return None

        self.latest_rate_hz = self.samples_in_window / elapsed
        self.window_start_seconds = now
        self.samples_in_window = 0
        return self.latest_rate_hz

    def reset(self) -> None:
        self.window_start_seconds = None
        self.samples_in_window = 0
        self.latest_rate_hz = 0.0


class PipelineTelemetry:
    """Records recent pipeline events and logs them consistently."""

    def __init__(
        self,
        *,
        max_events: int = 250,
        sample_rate_interval_seconds: float = 1.0,
        enabled: bool = True,
    ):
        if max_events <= 0:
            raise ValueError("max_events deve ser maior que zero.")
        self.enabled = bool(enabled)
        self.events: Deque[PipelineEvent] = deque(maxlen=max_events)
        self.sample_rate = SampleRateMonitor(
            interval_seconds=sample_rate_interval_seconds
        )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.reset()

    def record(self, name: str, **details: Any) -> PipelineEvent:
        event = PipelineEvent(
            name=name,
            timestamp_ms=time.time() * 1000,
            details=details,
        )
        if not self.enabled:
            return event
        self.events.append(event)
        logger.info("%s %s", name, details)
        return event

    def observe_eeg_sample(
        self,
        *,
        now_seconds: Optional[float] = None,
    ) -> Optional[float]:
        if not self.enabled:
            return None
        rate = self.sample_rate.observe(now_seconds=now_seconds)
        if rate is not None:
            self.record("EEG_RATE", rate_hz=round(rate, 2))
        return rate

    def reset(self) -> None:
        self.events.clear()
        self.sample_rate.reset()

    def latest_events(self, count: int = 20) -> list[PipelineEvent]:
        if count <= 0:
            return []
        return list(self.events)[-count:]
