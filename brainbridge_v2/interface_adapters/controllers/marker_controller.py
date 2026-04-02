"""
Controller that adapts presentation requests to marker and baseline use cases.
"""

from brainbridge_v2.application.ports.marker_state_store import MarkerStateStore
from brainbridge_v2.application.use_cases.marker_use_cases import (
    GetMarkerStateUseCase,
    RegisterMarkerUseCase,
    ResetMarkerStateUseCase,
    StartBaselineUseCase,
    TickBaselineUseCase,
)
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    BaselineTickViewModel,
    MarkerPresenter,
    MarkerRegistrationViewModel,
    MarkerStateViewModel,
)


class MarkerController:
    """
    Presentation-facing controller for markers and baseline workflows.
    """

    def __init__(
        self,
        get_marker_state_use_case: GetMarkerStateUseCase,
        reset_marker_state_use_case: ResetMarkerStateUseCase,
        register_marker_use_case: RegisterMarkerUseCase,
        start_baseline_use_case: StartBaselineUseCase,
        tick_baseline_use_case: TickBaselineUseCase,
    ):
        self._get_marker_state_use_case = get_marker_state_use_case
        self._reset_marker_state_use_case = reset_marker_state_use_case
        self._register_marker_use_case = register_marker_use_case
        self._start_baseline_use_case = start_baseline_use_case
        self._tick_baseline_use_case = tick_baseline_use_case

    @classmethod
    def from_store(cls, store: MarkerStateStore) -> "MarkerController":
        return cls(
            get_marker_state_use_case=GetMarkerStateUseCase(store),
            reset_marker_state_use_case=ResetMarkerStateUseCase(store),
            register_marker_use_case=RegisterMarkerUseCase(store),
            start_baseline_use_case=StartBaselineUseCase(store),
            tick_baseline_use_case=TickBaselineUseCase(store),
        )

    def get_state(self) -> MarkerStateViewModel:
        return MarkerPresenter.present_state(self._get_marker_state_use_case.execute())

    def reset_state(self) -> MarkerStateViewModel:
        return MarkerPresenter.present_state(self._reset_marker_state_use_case.execute())

    def register_marker(
        self,
        marker_type: str,
        task_type: str,
    ) -> MarkerRegistrationViewModel:
        result = self._register_marker_use_case.execute(marker_type, task_type)
        return MarkerPresenter.present_registration(result)

    def start_baseline(self, duration_seconds: int = 300) -> MarkerStateViewModel:
        return MarkerPresenter.present_state(
            self._start_baseline_use_case.execute(duration_seconds)
        )

    def tick_baseline(self) -> BaselineTickViewModel:
        result = self._tick_baseline_use_case.execute()
        return MarkerPresenter.present_baseline_tick(result)
