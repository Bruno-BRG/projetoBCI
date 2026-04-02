from brainbridge_v2.application.use_cases.eeg_stream_use_cases import (
    ConnectEEGStreamUseCase,
    DisconnectEEGStreamUseCase,
)


class FakeEEGStreamGateway:
    def __init__(self):
        self.connected = None
        self.disconnected = False

    def connect(self, host: str, port: int) -> bool:
        self.connected = (host, port)
        return True

    def disconnect(self) -> None:
        self.disconnected = True

    def is_running(self) -> bool:
        return False

    def is_mock_mode(self) -> bool:
        return False

    def set_data_callback(self, callback):
        self.data_callback = callback

    def set_connection_callback(self, callback):
        self.connection_callback = callback


def test_eeg_stream_use_cases_connect_and_disconnect():
    gateway = FakeEEGStreamGateway()

    assert ConnectEEGStreamUseCase(gateway).execute("localhost", 12345) is True
    DisconnectEEGStreamUseCase(gateway).execute()

    assert gateway.connected == ("localhost", 12345)
    assert gateway.disconnected is True
