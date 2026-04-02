from brainbridge_v2.interface_adapters.controllers.unity_controller import UnityController


class FakeUnityGateway:
    def __init__(self):
        self.started = False
        self.actions = []
        self.triggered = False
        self.ended_task = False
        self.ended_message = None
        self.message_callback = None
        self.connection_callback = None

    def start_server(self) -> bool:
        self.started = True
        return True

    def stop_server(self) -> None:
        self.started = False

    def send_action(self, action: str) -> bool:
        self.actions.append(action)
        return True

    def send_trigger(self) -> bool:
        self.triggered = True
        return True

    def end_task(self) -> bool:
        self.ended_task = True
        return True

    def end_session(self, message: str) -> bool:
        self.ended_message = message
        return True

    def is_server_active(self) -> bool:
        return self.started

    def is_client_connected(self) -> bool:
        return True

    def set_message_callback(self, callback):
        self.message_callback = callback

    def set_connection_callback(self, callback):
        self.connection_callback = callback


def test_unity_controller_reuses_gateway_and_exposes_callbacks():
    gateway = FakeUnityGateway()
    controller = UnityController.from_gateway(gateway)

    controller.set_message_callback(lambda message: message)
    controller.set_connection_callback(lambda connected: connected)
    assert controller.start_server() is True
    assert controller.send_action("direita") is True
    assert controller.send_trigger() is True
    assert controller.end_task() is True
    assert controller.end_session("final") is True
    assert controller.is_server_active() is True
    assert controller.is_client_connected() is True
    assert gateway.actions == ["direita"]
    assert gateway.triggered is True
    assert gateway.ended_task is True
    assert gateway.ended_message == "final"
    assert gateway.message_callback is not None
    assert gateway.connection_callback is not None
