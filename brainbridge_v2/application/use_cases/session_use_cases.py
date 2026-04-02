"""
Session-related use cases.
"""

from typing import Optional

from brainbridge_v2.application.ports.session_store import SessionStore
from brainbridge_v2.domain.entities.session import Session


class StartSessionUseCase:
    """
    Starts a runtime session after validation.
    """

    def __init__(self, store: SessionStore):
        self._store = store

    def execute(self, session: Session) -> Session:
        if self._store.get_current() is not None:
            raise ValueError("Ja existe uma sessao ativa.")
        session.validate()
        return self._store.start(session)


class GetCurrentSessionUseCase:
    """
    Returns the current runtime session.
    """

    def __init__(self, store: SessionStore):
        self._store = store

    def execute(self) -> Optional[Session]:
        return self._store.get_current()


class EndSessionUseCase:
    """
    Ends the current runtime session.
    """

    def __init__(self, store: SessionStore):
        self._store = store

    def execute(self) -> Optional[Session]:
        current = self._store.get_current()
        self._store.clear()
        return current
