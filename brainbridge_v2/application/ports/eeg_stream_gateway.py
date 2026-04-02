"""
Port definition for EEG stream operations.
"""

from typing import Callable, Protocol


class EEGStreamGateway(Protocol):
    def connect(self, host: str, port: int) -> bool:
        ...

    def disconnect(self) -> None:
        ...

    def is_running(self) -> bool:
        ...

    def is_mock_mode(self) -> bool:
        ...

    def set_data_callback(self, callback: Callable[[object], None]) -> None:
        ...

    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        ...
