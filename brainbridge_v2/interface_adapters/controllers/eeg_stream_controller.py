"""
Controller that adapts presentation requests to EEG stream use cases.
"""

from typing import Callable

from brainbridge_v2.application.ports.eeg_stream_gateway import EEGStreamGateway
from brainbridge_v2.application.use_cases.eeg_stream_use_cases import (
    ConnectEEGStreamUseCase,
    DisconnectEEGStreamUseCase,
)


class EEGStreamController:
    """
    Presentation-facing controller for EEG stream workflows.
    """

    def __init__(
        self,
        gateway: EEGStreamGateway,
        connect_use_case: ConnectEEGStreamUseCase,
        disconnect_use_case: DisconnectEEGStreamUseCase,
    ):
        self._gateway = gateway
        self._connect_use_case = connect_use_case
        self._disconnect_use_case = disconnect_use_case

    @classmethod
    def from_gateway(cls, gateway: EEGStreamGateway) -> "EEGStreamController":
        return cls(
            gateway=gateway,
            connect_use_case=ConnectEEGStreamUseCase(gateway),
            disconnect_use_case=DisconnectEEGStreamUseCase(gateway),
        )

    def connect(self, host: str, port: int) -> bool:
        return self._connect_use_case.execute(host, port)

    def disconnect(self) -> None:
        self._disconnect_use_case.execute()

    def is_running(self) -> bool:
        return self._gateway.is_running()

    def is_mock_mode(self) -> bool:
        return self._gateway.is_mock_mode()

    def set_data_callback(self, callback: Callable[[object], None]) -> None:
        self._gateway.set_data_callback(callback)

    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        self._gateway.set_connection_callback(callback)
