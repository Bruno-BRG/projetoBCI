"""
ESP32 gateway adapter backed by the concrete ESP32 communicator.
"""

from typing import Callable

from brainbridge_v2.infrastructure.communication.esp32 import (
    ESP32SerialCommunicator,
    get_esp32_communicator,
)


class ESP32GatewayAdapter:
    """
    Infrastructure adapter that maps ESP32Gateway to ESP32SerialCommunicator.
    """

    def __init__(self, communicator: ESP32SerialCommunicator | None = None):
        self._communicator = communicator or get_esp32_communicator()

    def connect(self) -> bool:
        return self._communicator.connect()

    def disconnect(self) -> None:
        self._communicator.disconnect()

    def send_direction(self, direction: str) -> bool:
        if direction == "esquerda":
            return self._communicator.send_trigger_left()
        if direction == "direita":
            return self._communicator.send_trigger_right()
        raise ValueError(f"Direcao ESP32 invalida: {direction}")

    def is_connected(self) -> bool:
        return bool(self._communicator.is_connected)

    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        self._communicator.set_connection_callback(callback)
