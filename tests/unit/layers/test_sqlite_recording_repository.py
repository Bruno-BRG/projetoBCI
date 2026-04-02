from pathlib import Path
import shutil
import uuid

from brainbridge_v2.domain.entities.recording import Recording
from brainbridge_v2.infrastructure.database.manager import DatabaseManager
from brainbridge_v2.infrastructure.repositories.sqlite_recording_repository import (
    SQLiteRecordingRepository,
)


def test_sqlite_recording_repository_add_finish_and_list():
    temp_dir = Path("brainbridge_v2") / "data" / f"tmp_test_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        db_path = temp_dir / "recordings_test.db"
        manager = DatabaseManager(db_path=db_path)
        repository = SQLiteRecordingRepository(manager)

        recording_id = repository.add(
            Recording(
                patient_id=3,
                filename="P003/treino.csv",
                task_type="treino",
                notes="Caso de teste",
            )
        )
        repository.finish(recording_id, 33)
        recordings = repository.list_by_patient(3)

        assert recording_id > 0
        assert len(recordings) == 1
        assert recordings[0].id == recording_id
        assert recordings[0].filename == "P003/treino.csv"
        assert recordings[0].task_type == "treino"
        assert recordings[0].duration == 33
        assert recordings[0].end_time is not None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
