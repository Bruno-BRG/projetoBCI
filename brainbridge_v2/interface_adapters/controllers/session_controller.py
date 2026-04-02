"""
Controller that adapts presentation requests to session use cases.
"""

from typing import Any, Dict, Optional

from brainbridge_v2.application.ports.session_store import SessionStore
from brainbridge_v2.application.use_cases.session_use_cases import (
    EndSessionUseCase,
    GetCurrentSessionUseCase,
    StartSessionUseCase,
)
from brainbridge_v2.domain.entities.session import Session
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    SessionPresenter,
    SessionViewModel,
    StartSessionRequest,
)


class SessionController:
    """
    Presentation-facing controller for runtime session workflows.
    """

    def __init__(
        self,
        start_session_use_case: StartSessionUseCase,
        get_current_session_use_case: GetCurrentSessionUseCase,
        end_session_use_case: EndSessionUseCase,
    ):
        self._start_session_use_case = start_session_use_case
        self._get_current_session_use_case = get_current_session_use_case
        self._end_session_use_case = end_session_use_case

    @classmethod
    def from_store(cls, store: SessionStore) -> "SessionController":
        return cls(
            start_session_use_case=StartSessionUseCase(store),
            get_current_session_use_case=GetCurrentSessionUseCase(store),
            end_session_use_case=EndSessionUseCase(store),
        )

    def start_session(
        self,
        session_data: StartSessionRequest | Dict[str, Any],
    ) -> SessionViewModel:
        request = self._normalize_request(session_data)
        session = Session(
            patient_id=request.patient_id,
            task_type=request.task_type.strip(),
            recording_id=request.recording_id,
            started_at_epoch=request.started_at_epoch,
        )
        stored = self._start_session_use_case.execute(session)
        return SessionPresenter.present(stored)

    def get_current_session(self) -> Optional[SessionViewModel]:
        current = self._get_current_session_use_case.execute()
        if current is None:
            return None
        return SessionPresenter.present(current)

    def end_session(self) -> Optional[SessionViewModel]:
        current = self._end_session_use_case.execute()
        if current is None:
            return None
        return SessionPresenter.present(current)

    @staticmethod
    def _normalize_request(
        session_data: StartSessionRequest | Dict[str, Any],
    ) -> StartSessionRequest:
        if isinstance(session_data, StartSessionRequest):
            return session_data

        return StartSessionRequest(
            patient_id=int(session_data.get("patient_id", 0)),
            task_type=str(session_data.get("task_type", "")).strip(),
            recording_id=int(session_data.get("recording_id", 0)),
            started_at_epoch=float(session_data.get("started_at_epoch", 0)),
        )
