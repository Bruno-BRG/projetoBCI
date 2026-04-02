from typing import List

import pytest

from brainbridge_v2.application.use_cases.recording_use_cases import (
    ListPatientRecordingsUseCase,
    StartRecordingUseCase,
    StopRecordingUseCase,
)
from brainbridge_v2.domain.entities.recording import Recording


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
            start_time="2026-04-01T10:00:00",
        )
        self._recordings.append(stored)
        return new_id

    def finish(self, recording_id: int, duration_seconds: int) -> None:
        self.finished_calls.append((recording_id, duration_seconds))

    def list_by_patient(self, patient_id: int) -> List[Recording]:
        return [recording for recording in self._recordings if recording.patient_id == patient_id]


def test_start_recording_use_case_valid_input():
    repo = FakeRecordingRepository()
    use_case = StartRecordingUseCase(repo)

    recording_id = use_case.execute(
        Recording(
            patient_id=7,
            filename="P007/gravacao.csv",
            task_type="treino",
            notes="Sessao inicial",
        )
    )

    assert recording_id == 1
    assert len(repo.list_by_patient(7)) == 1


def test_start_recording_use_case_invalid_patient():
    repo = FakeRecordingRepository()
    use_case = StartRecordingUseCase(repo)

    with pytest.raises(ValueError):
        use_case.execute(
            Recording(
                patient_id=0,
                filename="x.csv",
                task_type="baseline",
            )
        )


def test_stop_recording_use_case_marks_recording_as_finished():
    repo = FakeRecordingRepository()
    use_case = StopRecordingUseCase(repo)

    use_case.execute(10, 42)

    assert repo.finished_calls == [(10, 42)]


def test_list_patient_recordings_use_case_filters_by_patient():
    repo = FakeRecordingRepository()
    repo.add(Recording(patient_id=1, filename="a.csv", task_type="treino"))
    repo.add(Recording(patient_id=2, filename="b.csv", task_type="jogo"))
    use_case = ListPatientRecordingsUseCase(repo)

    recordings = use_case.execute(1)

    assert len(recordings) == 1
    assert recordings[0].filename == "a.csv"
