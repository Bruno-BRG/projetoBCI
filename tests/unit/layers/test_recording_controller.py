from typing import List

from brainbridge_v2.domain.entities.recording import Recording
from brainbridge_v2.interface_adapters.controllers.recording_controller import RecordingController
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    StartRecordingRequest,
)


class FakeRecordingRepository:
    def __init__(self):
        self._recordings: List[Recording] = []
        self._next_id = 1
        self.finished_calls = []

    def add(self, recording: Recording) -> int:
        new_id = self._next_id
        self._next_id += 1
        stored = Recording(
            id=new_id,
            patient_id=recording.patient_id,
            filename=recording.filename,
            task_type=recording.task_type,
            notes=recording.notes,
            start_time="2026-04-01T10:30:00",
        )
        self._recordings.append(stored)
        return new_id

    def finish(self, recording_id: int, duration_seconds: int) -> None:
        self.finished_calls.append((recording_id, duration_seconds))

    def list_by_patient(self, patient_id: int) -> List[Recording]:
        return [recording for recording in self._recordings if recording.patient_id == patient_id]


def test_recording_controller_start_list_and_stop():
    controller = RecordingController.from_repository(FakeRecordingRepository())

    recording_id = controller.start_recording(
        StartRecordingRequest(
            patient_id=9,
            filename="P009/teste.csv",
            task_type="treino",
            notes="Primeira coleta",
        )
    )
    recordings = controller.list_patient_recordings(9)
    controller.stop_recording(recording_id, 65)

    assert recording_id == 1
    assert len(recordings) == 1
    assert recordings[0].filename == "P009/teste.csv"
    assert recordings[0].start_time == "2026-04-01T10:30:00"
