"""
ESP32 communication use cases.
"""

from brainbridge_v2.application.ports.esp32_gateway import ESP32Gateway


class ConnectESP32UseCase:
    def __init__(self, gateway: ESP32Gateway):
        self._gateway = gateway

    def execute(self) -> bool:
        return self._gateway.connect()


class DisconnectESP32UseCase:
    def __init__(self, gateway: ESP32Gateway):
        self._gateway = gateway

    def execute(self) -> None:
        self._gateway.disconnect()


class SendESP32SignalUseCase:
    def __init__(self, gateway: ESP32Gateway):
        self._gateway = gateway

    def execute(self, direction: str) -> bool:
        if not direction or not direction.strip():
            raise ValueError("Direcao ESP32 eh obrigatoria.")
        return self._gateway.send_direction(direction.strip())
