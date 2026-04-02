"""
Marker and baseline runtime state for the domain layer.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MarkerState:
    """
    Runtime state for action markers and baseline countdown.
    """

    t1_count: int = 0
    t2_count: int = 0
    baseline_remaining_seconds: int = 0

    def validate(self) -> None:
        if self.t1_count < 0 or self.t2_count < 0:
            raise ValueError("Contadores de marcadores nao podem ser negativos.")
        if self.baseline_remaining_seconds < 0:
            raise ValueError("Tempo restante de baseline nao pode ser negativo.")

    @property
    def baseline_active(self) -> bool:
        return self.baseline_remaining_seconds > 0
