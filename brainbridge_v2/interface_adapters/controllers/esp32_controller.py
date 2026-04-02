"""
Controller that adapts presentation requests to ESP32 communication use cases.
"""

from typing import Callable

from brainbridge_v2.application.ports.esp32_gateway import ESP32Gateway
from brainbridge_v2.application.use_cases.esp32_use_cases import (
    ConnectESP32UseCase,
    DisconnectESP32UseCase,
    SendESP32SignalUseCase,
)


class ESP32Controller:
    """
    Presentation-facing controller for ESP32 workflows.
    """

    def __init__(
        self,
        gateway: ESP32Gateway,
        connect_use_case: ConnectESP32UseCase,
        disconnect_use_case: DisconnectESP32UseCase,
        send_signal_use_case: SendESP32SignalUseCase,
    ):
        self._gateway = gateway
        self._connect_use_case = connect_use_case
        self._disconnect_use_case = disconnect_use_case
        self._send_signal_use_case = send_signal_use_case

    @classmethod
    def from_gateway(cls, gateway: ESP32Gateway) -> "ESP32Controller":
        return cls(
            gateway=gateway,
            connect_use_case=ConnectESP32UseCase(gateway),
            disconnect_use_case=DisconnectESP32UseCase(gateway),
            send_signal_use_case=SendESP32SignalUseCase(gateway),
        )

    def connect(self) -> bool:
        return self._connect_use_case.execute()

    def disconnect(self) -> None:
        self._disconnect_use_case.execute()

    def send_direction(self, direction: str) -> bool:
        return self._send_signal_use_case.execute(direction)

    def is_connected(self) -> bool:
        return self._gateway.is_connected()

    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        self._gateway.set_connection_callback(callback)
