from brainbridge_v2.interface_adapters.controllers.esp32_controller import ESP32Controller


class FakeESP32Gateway:
    def __init__(self):
        self.connected = False
        self.sent = []
        self.connection_callback = None

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
        self.connection_callback = callback


def test_esp32_controller_connect_send_disconnect():
    gateway = FakeESP32Gateway()
    controller = ESP32Controller.from_gateway(gateway)

    controller.set_connection_callback(lambda connected: connected)
    assert controller.connect() is True
    assert controller.send_direction("direita") is True
    controller.disconnect()

    assert controller.is_connected() is False
    assert gateway.sent == ["direita"]
    assert gateway.connection_callback is not None
