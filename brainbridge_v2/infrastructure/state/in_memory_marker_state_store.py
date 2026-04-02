"""
In-memory store for marker and baseline runtime state.
"""

from brainbridge_v2.domain.entities.marker_state import MarkerState


class InMemoryMarkerStateStore:
    """
    Keeps marker counters and baseline state in memory.
    """

    def __init__(self):
        self._state = MarkerState()

    def get(self) -> MarkerState:
        return self._state

    def save(self, state: MarkerState) -> MarkerState:
        self._state = state
        return state
