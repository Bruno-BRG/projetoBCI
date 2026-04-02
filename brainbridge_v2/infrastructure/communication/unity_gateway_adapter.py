"""
Unity gateway adapter backed by the concrete Unity communicator.
"""

from typing import Callable

from brainbridge_v2.infrastructure.communication.unity import UDP_sender, UnityCommunicator


class UnityGatewayAdapter:
    """
    Infrastructure adapter that maps UnityGateway to UnityCommunicator.
    """

    def __init__(self, communicator: UnityCommunicator | None = None):
        self._communicator = communicator or UnityCommunicator()

    def start_server(self) -> bool:
        return self._communicator.start_server()

    def stop_server(self) -> None:
        self._communicator.stop_server()

    def send_action(self, action: str) -> bool:
        return UDP_sender.enviar_sinal(action)

    def send_trigger(self) -> bool:
        return self._communicator.send_trigger()

    def end_task(self) -> bool:
        return self._communicator.end_task()

    def end_session(self, message: str) -> bool:
        return self._communicator.end_session(message)

    def is_server_active(self) -> bool:
        return bool(self._communicator.is_active)

    def is_client_connected(self) -> bool:
        return bool(self._communicator.tcp_connected)

    def set_message_callback(self, callback: Callable[[str], None]) -> None:
        self._communicator.set_message_callback(callback)

    def set_connection_callback(self, callback: Callable[[bool], None]) -> None:
        self._communicator.set_connection_callback(callback)
