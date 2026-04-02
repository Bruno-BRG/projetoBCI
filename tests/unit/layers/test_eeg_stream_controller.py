from brainbridge_v2.interface_adapters.controllers.eeg_stream_controller import (
    EEGStreamController,
)


class FakeEEGStreamGateway:
    def __init__(self):
        self.connected = None
        self.disconnected = False
        self.mock_mode = True
        self.running = False
        self.data_callback = None
        self.connection_callback = None

    def connect(self, host: str, port: int) -> bool:
        self.connected = (host, port)
        self.running = True
        return True

    def disconnect(self) -> None:
        self.disconnected = True
        self.running = False

    def is_running(self) -> bool:
        return self.running

    def is_mock_mode(self) -> bool:
        return self.mock_mode

    def set_data_callback(self, callback):
        self.data_callback = callback

    def set_connection_callback(self, callback):
        self.connection_callback = callback


def test_eeg_stream_controller_connect_callbacks_and_disconnect():
    gateway = FakeEEGStreamGateway()
    controller = EEGStreamController.from_gateway(gateway)

    controller.set_data_callback(lambda data: data)
    controller.set_connection_callback(lambda connected: connected)
    assert controller.connect("127.0.0.1", 12345) is True
    assert controller.is_running() is True
    assert controller.is_mock_mode() is True
    controller.disconnect()

    assert gateway.connected == ("127.0.0.1", 12345)
    assert gateway.disconnected is True
    assert gateway.data_callback is not None
    assert gateway.connection_callback is not None
