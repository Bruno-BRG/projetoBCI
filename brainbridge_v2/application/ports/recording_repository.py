"""
Port definition for recording persistence.
"""

from typing import List, Protocol

from brainbridge_v2.domain.entities.recording import Recording


class RecordingRepository(Protocol):
    def add(self, recording: Recording) -> int:
        """
        Persists a recording and returns the generated identifier.
        """
        ...

    def finish(self, recording_id: int, duration_seconds: int) -> None:
        """
        Persists the end of a recording.
        """
        ...

    def list_by_patient(self, patient_id: int) -> List[Recording]:
        """
        Returns all recordings for a patient.
        """
        ...
