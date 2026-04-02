"""
SQLite repository adapter for recording operations.
"""

from typing import List

from brainbridge_v2.domain.entities.recording import Recording
from brainbridge_v2.infrastructure.database.manager import DatabaseManager


class SQLiteRecordingRepository:
    """
    Infrastructure adapter that maps RecordingRepository port to DatabaseManager.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db_manager = db_manager

    def add(self, recording: Recording) -> int:
        return self._db_manager.add_recording(
            recording.patient_id,
            recording.filename,
            recording.task_type,
            recording.notes,
        )

    def finish(self, recording_id: int, duration_seconds: int) -> None:
        self._db_manager.update_recording_end_time(recording_id, duration_seconds)

    def list_by_patient(self, patient_id: int) -> List[Recording]:
        rows = self._db_manager.get_patient_recordings(patient_id)
        recordings: List[Recording] = []
        for row in rows:
            recordings.append(
                Recording(
                    id=row.get("id"),
                    patient_id=int(row.get("patient_id", 0)),
                    filename=row.get("filename", ""),
                    task_type=row.get("task_type", ""),
                    start_time=row.get("start_time"),
                    end_time=row.get("end_time"),
                    duration=row.get("duration"),
                    notes=row.get("notes") or "",
                )
            )
        return recordings
