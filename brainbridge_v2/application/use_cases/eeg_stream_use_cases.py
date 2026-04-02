"""
EEG streaming use cases.
"""

from brainbridge_v2.application.ports.eeg_stream_gateway import EEGStreamGateway


class ConnectEEGStreamUseCase:
    def __init__(self, gateway: EEGStreamGateway):
        self._gateway = gateway

    def execute(self, host: str, port: int) -> bool:
        if not host or not host.strip():
            raise ValueError("Host EEG eh obrigatorio.")
        if port <= 0:
            raise ValueError("Porta EEG invalida.")
        return self._gateway.connect(host.strip(), int(port))


class DisconnectEEGStreamUseCase:
    def __init__(self, gateway: EEGStreamGateway):
        self._gateway = gateway

    def execute(self) -> None:
        self._gateway.disconnect()
