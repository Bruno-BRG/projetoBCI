"""
Controller that adapts presentation requests to recording use cases.
"""

from typing import Any, Dict, List

from brainbridge_v2.application.ports.recording_repository import RecordingRepository
from brainbridge_v2.application.use_cases.recording_use_cases import (
    ListPatientRecordingsUseCase,
    StartRecordingUseCase,
    StopRecordingUseCase,
)
from brainbridge_v2.domain.entities.recording import Recording
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    RecordingPresenter,
    RecordingViewModel,
    StartRecordingRequest,
)


class RecordingController:
    """
    Presentation-facing controller for recording workflows.
    """

    def __init__(
        self,
        start_recording_use_case: StartRecordingUseCase,
        stop_recording_use_case: StopRecordingUseCase,
        list_patient_recordings_use_case: ListPatientRecordingsUseCase,
    ):
        self._start_recording_use_case = start_recording_use_case
        self._stop_recording_use_case = stop_recording_use_case
        self._list_patient_recordings_use_case = list_patient_recordings_use_case

    @classmethod
    def from_repository(cls, repository: RecordingRepository) -> "RecordingController":
        return cls(
            start_recording_use_case=StartRecordingUseCase(repository),
            stop_recording_use_case=StopRecordingUseCase(repository),
            list_patient_recordings_use_case=ListPatientRecordingsUseCase(repository),
        )

    def start_recording(
        self,
        recording_data: StartRecordingRequest | Dict[str, Any],
    ) -> int:
        request = self._normalize_request(recording_data)
        recording = Recording(
            patient_id=request.patient_id,
            filename=request.filename.strip(),
            task_type=request.task_type.strip(),
            notes=request.notes.strip(),
        )
        return self._start_recording_use_case.execute(recording)

    def stop_recording(self, recording_id: int, duration_seconds: int) -> None:
        self._stop_recording_use_case.execute(recording_id, duration_seconds)

    def list_patient_recordings(self, patient_id: int) -> List[RecordingViewModel]:
        recordings = self._list_patient_recordings_use_case.execute(patient_id)
        return [RecordingPresenter.present(recording) for recording in recordings]

    @staticmethod
    def _normalize_request(
        recording_data: StartRecordingRequest | Dict[str, Any],
    ) -> StartRecordingRequest:
        if isinstance(recording_data, StartRecordingRequest):
            return recording_data

        return StartRecordingRequest(
            patient_id=int(recording_data.get("patient_id", 0)),
            filename=str(recording_data.get("filename", "")).strip(),
            task_type=str(recording_data.get("task_type", "")).strip(),
            notes=str(recording_data.get("notes", "")).strip(),
        )
