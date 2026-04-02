"""
Port definition for storing the active runtime session.
"""

from typing import Optional, Protocol

from brainbridge_v2.domain.entities.session import Session


class SessionStore(Protocol):
    def start(self, session: Session) -> Session:
        """
        Stores the current active session.
        """
        ...

    def get_current(self) -> Optional[Session]:
        """
        Returns the current active session, if any.
        """
        ...

    def clear(self) -> None:
        """
        Clears the current session.
        """
        ...
