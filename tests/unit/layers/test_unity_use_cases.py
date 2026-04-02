from brainbridge_v2.application.use_cases.unity_use_cases import (
    EndUnitySessionUseCase,
    EndUnityTaskUseCase,
    SendUnityActionUseCase,
    SendUnityTriggerUseCase,
    StartUnityServerUseCase,
    StopUnityServerUseCase,
)


class FakeUnityGateway:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.actions = []
        self.triggers = 0
        self.ended_tasks = 0
        self.ended_sessions = []

    def start_server(self) -> bool:
        self.started = True
        return True

    def stop_server(self) -> None:
        self.stopped = True

    def send_action(self, action: str) -> bool:
        self.actions.append(action)
        return True

    def send_trigger(self) -> bool:
        self.triggers += 1
        return True

    def end_task(self) -> bool:
        self.ended_tasks += 1
        return True

    def end_session(self, message: str) -> bool:
        self.ended_sessions.append(message)
        return True

    def is_server_active(self) -> bool:
        return self.started and not self.stopped

    def is_client_connected(self) -> bool:
        return True

    def set_message_callback(self, callback):
        self.message_callback = callback

    def set_connection_callback(self, callback):
        self.connection_callback = callback


def test_unity_use_cases_cover_server_action_and_session_commands():
    gateway = FakeUnityGateway()

    assert StartUnityServerUseCase(gateway).execute() is True
    assert SendUnityActionUseCase(gateway).execute("trigger_left") is True
    assert SendUnityTriggerUseCase(gateway).execute() is True
    assert EndUnityTaskUseCase(gateway).execute() is True
    assert EndUnitySessionUseCase(gateway).execute("ok") is True
    StopUnityServerUseCase(gateway).execute()

    assert gateway.actions == ["trigger_left"]
    assert gateway.triggers == 1
    assert gateway.ended_tasks == 1
    assert gateway.ended_sessions == ["ok"]
    assert gateway.stopped is True
