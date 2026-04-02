from brainbridge_v2.application.use_cases.esp32_use_cases import (
    ConnectESP32UseCase,
    DisconnectESP32UseCase,
    SendESP32SignalUseCase,
)


class FakeESP32Gateway:
    def __init__(self):
        self.connected = False
        self.sent = []

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def send_direction(self, direction: str) -> bool:
        self.sent.append(direction)
        return True

    def is_connected(self) -> bool:
        return self.connected

    def set_connection_callback(self, callback):
        self.callback = callback


def test_esp32_use_cases_cover_connection_and_signal_send():
    gateway = FakeESP32Gateway()

    assert ConnectESP32UseCase(gateway).execute() is True
    assert SendESP32SignalUseCase(gateway).execute("esquerda") is True
    DisconnectESP32UseCase(gateway).execute()

    assert gateway.sent == ["esquerda"]
    assert gateway.connected is False
