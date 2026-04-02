import pytest

from brainbridge_v2.application.use_cases.session_use_cases import (
    EndSessionUseCase,
    GetCurrentSessionUseCase,
    StartSessionUseCase,
)
from brainbridge_v2.domain.entities.session import Session
from brainbridge_v2.infrastructure.state.in_memory_session_store import InMemorySessionStore


def test_start_session_use_case_stores_active_session():
    store = InMemorySessionStore()
    use_case = StartSessionUseCase(store)

    session = use_case.execute(
        Session(
            patient_id=5,
            task_type="jogo",
            recording_id=12,
            started_at_epoch=123.4,
        )
    )

    assert session.patient_id == 5
    assert session.game_mode is True
    assert store.get_current() == session


def test_start_session_use_case_rejects_second_active_session():
    store = InMemorySessionStore()
    use_case = StartSessionUseCase(store)
    use_case.execute(
        Session(
            patient_id=1,
            task_type="treino",
            recording_id=1,
            started_at_epoch=1.0,
        )
    )

    with pytest.raises(ValueError):
        use_case.execute(
            Session(
                patient_id=2,
                task_type="jogo",
                recording_id=2,
                started_at_epoch=2.0,
            )
        )


def test_get_and_end_session_use_cases_return_current_session():
    store = InMemorySessionStore()
    session = Session(
        patient_id=9,
        task_type="baseline",
        recording_id=22,
        started_at_epoch=50.0,
    )
    store.start(session)

    current = GetCurrentSessionUseCase(store).execute()
    ended = EndSessionUseCase(store).execute()

    assert current == session
    assert ended == session
    assert store.get_current() is None
