"""
Port definition for ESP32 communication operations.
"""

from typing import Callable, Protocol


class ESP32Gateway(Protocol):
    def connect(self) -> bool:
        ...

    def disconnect(self) -> None:
        ...

    def send_direction(self, direction: str) -> bool:
        ...

    def is_connected(self) -> bool:
        ...

    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        ...
