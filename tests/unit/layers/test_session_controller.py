from brainbridge_v2.infrastructure.state.in_memory_session_store import InMemorySessionStore
from brainbridge_v2.interface_adapters.controllers.session_controller import SessionController
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    StartSessionRequest,
)


def test_session_controller_start_get_and_end():
    controller = SessionController.from_store(InMemorySessionStore())

    started = controller.start_session(
        StartSessionRequest(
            patient_id=11,
            task_type="jogo",
            recording_id=30,
            started_at_epoch=200.0,
        )
    )
    current = controller.get_current_session()
    ended = controller.end_session()
    empty = controller.get_current_session()

    assert started.patient_id == 11
    assert started.game_mode is True
    assert current.recording_id == 30
    assert ended.task_type == "jogo"
    assert empty is None
