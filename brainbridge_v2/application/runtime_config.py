"""
Runtime defaults for the EEG game pipeline.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    sample_rate_hz: int = 125
    window_size: int = 250
    channels: int = 16
    ai_window_duration_ms: int = 2000
    game_action_interval_ms: int = 10000
    eeg_max_abs_amplitude: float = 5000.0
    eeg_min_channel_std: float = 1e-6
    tensorflow_warmup_enabled: bool = True


DEFAULT_RUNTIME_CONFIG = RuntimeConfig()
