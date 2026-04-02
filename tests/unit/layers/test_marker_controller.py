from brainbridge_v2.infrastructure.state.in_memory_marker_state_store import (
    InMemoryMarkerStateStore,
)
from brainbridge_v2.interface_adapters.controllers.marker_controller import MarkerController


def test_marker_controller_register_and_tick_baseline():
    controller = MarkerController.from_store(InMemoryMarkerStateStore())

    state = controller.start_baseline(2)
    blocked = controller.register_marker("T1", "jogo")
    tick1 = controller.tick_baseline()
    tick2 = controller.tick_baseline()
    accepted = controller.register_marker("T2", "treino")
    reset = controller.reset_state()

    assert state.baseline_active is True
    assert blocked.accepted is False
    assert tick1.finished is False
    assert tick2.finished is True
    assert accepted.accepted is True
    assert accepted.state.t2_count == 1
    assert reset.t1_count == 0
    assert reset.t2_count == 0
