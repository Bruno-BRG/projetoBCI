"""
Recording-related use cases.
"""

from typing import List

from brainbridge_v2.application.ports.recording_repository import RecordingRepository
from brainbridge_v2.domain.entities.recording import Recording


class StartRecordingUseCase:
    """
    Registers a new recording after domain validation.
    """

    def __init__(self, repository: RecordingRepository):
        self._repository = repository

    def execute(self, recording: Recording) -> int:
        recording.validate()
        return self._repository.add(recording)


class StopRecordingUseCase:
    """
    Marks an existing recording as finished.
    """

    def __init__(self, repository: RecordingRepository):
        self._repository = repository

    def execute(self, recording_id: int, duration_seconds: int) -> None:
        if recording_id <= 0:
            raise ValueError("Gravacao invalida.")
        if duration_seconds < 0:
            raise ValueError("Duracao da gravacao nao pode ser negativa.")
        self._repository.finish(recording_id, duration_seconds)


class ListPatientRecordingsUseCase:
    """
    Returns all recordings for a patient.
    """

    def __init__(self, repository: RecordingRepository):
        self._repository = repository

    def execute(self, patient_id: int) -> List[Recording]:
        if patient_id <= 0:
            raise ValueError("Paciente invalido.")
        return self._repository.list_by_patient(patient_id)
