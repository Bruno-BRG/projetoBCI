"""
Port definition for storing marker and baseline runtime state.
"""

from brainbridge_v2.domain.entities.marker_state import MarkerState
from typing import Protocol


class MarkerStateStore(Protocol):
    def get(self) -> MarkerState:
        """
        Returns the current marker state.
        """
        ...

    def save(self, state: MarkerState) -> MarkerState:
        """
        Persists and returns the current marker state.
        """
        ...
