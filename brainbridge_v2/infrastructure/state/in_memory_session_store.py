"""
In-memory store for the active runtime session.
"""

from typing import Optional

from brainbridge_v2.domain.entities.session import Session


class InMemorySessionStore:
    """
    Keeps the current session in memory for the running application process.
    """

    def __init__(self):
        self._current: Optional[Session] = None

    def start(self, session: Session) -> Session:
        self._current = session
        return session

    def get_current(self) -> Optional[Session]:
        return self._current

    def clear(self) -> None:
        self._current = None
