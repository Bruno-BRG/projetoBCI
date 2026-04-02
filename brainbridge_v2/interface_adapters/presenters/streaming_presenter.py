"""
View models and presenter helpers for the streaming workflow.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from brainbridge_v2.domain.entities.marker_state import MarkerState
from brainbridge_v2.domain.entities.model_metadata import ModelMetadata
from brainbridge_v2.domain.entities.prediction_result import PredictionResult
from brainbridge_v2.domain.entities.recording import Recording
from brainbridge_v2.domain.entities.session import Session
from brainbridge_v2.domain.entities.training_result import TrainingResult


@dataclass(frozen=True)
class StartRecordingRequest:
    patient_id: int
    filename: str
    task_type: str
    notes: str = ""


@dataclass(frozen=True)
class StartSessionRequest:
    patient_id: int
    task_type: str
    recording_id: int
    started_at_epoch: float


@dataclass(frozen=True)
class RecordingViewModel:
    id: Optional[int]
    patient_id: int
    filename: str
    task_type: str
    start_time: str
    end_time: str
    duration: Optional[int]
    notes: str


@dataclass(frozen=True)
class SessionViewModel:
    patient_id: int
    task_type: str
    recording_id: int
    started_at_epoch: float
    game_mode: bool


@dataclass(frozen=True)
class MarkerStateViewModel:
    t1_count: int
    t2_count: int
    baseline_remaining_seconds: int
    baseline_active: bool


@dataclass(frozen=True)
class MarkerRegistrationViewModel:
    accepted: bool
    reason: Optional[str]
    state: MarkerStateViewModel
    external_signal: Optional[str]
    esp32_direction: Optional[str]
    marker_type: Optional[str]


@dataclass(frozen=True)
class BaselineTickViewModel:
    state: MarkerStateViewModel
    finished: bool


@dataclass(frozen=True)
class ModelViewModel:
    path: str
    name: str
    backend: str
    input_shape: Optional[Tuple[Optional[int], ...]]
    expected_time_steps: Optional[int]
    expected_channels: Optional[int]
    modified_at_epoch: Optional[float]


@dataclass(frozen=True)
class PredictionViewModel:
    predicted_index: int
    confidence: float
    probabilities: Tuple[float, ...]
    left_probability: float
    right_probability: float


@dataclass(frozen=True)
class TrainingResultViewModel:
    model_path: str
    training_time_seconds: float
    final_accuracy: Optional[float]
    final_loss: Optional[float]
    val_accuracy: Optional[float]
    val_loss: Optional[float]
    auto_loaded: bool
    loaded_model_path: Optional[str]
    loaded_model_name: Optional[str]


@dataclass(frozen=True)
class StreamingSessionStateViewModel:
    patient_id: Optional[int]
    task_type: Optional[str]
    recording_id: Optional[int]
    started_at_epoch: Optional[float]
    game_mode: bool
    recording_active: bool
    baseline_active: bool
    baseline_remaining_seconds: int
    t1_count: int
    t2_count: int
    eeg_connected: bool
    eeg_mock_mode: bool
    unity_server_active: bool
    esp32_connected: bool
    model_loaded: bool
    model_name: Optional[str]


@dataclass(frozen=True)
class LabelStateViewModel:
    text: str
    style_sheet: str


@dataclass(frozen=True)
class ConnectionPanelViewModel:
    eeg: LabelStateViewModel
    vr: LabelStateViewModel
    orthosis: LabelStateViewModel
    connect_button_text: str
    connect_button_enabled: bool
    record_button_enabled: bool


@dataclass(frozen=True)
class AccuracyTrialViewModel:
    expected_action: str
    real_action: str
    is_correct: bool


@dataclass(frozen=True)
class AccuracyViewModel:
    summary_text: str
    details_text: str
    summary_style_sheet: str
    correct_count: int
    total_count: int
    accuracy_percent: float


@dataclass(frozen=True)
class AiStatusViewModel:
    text: str
    style_sheet: str


@dataclass(frozen=True)
class PredictionDisplayViewModel:
    prediction_text: str
    prediction_style_sheet: str
    left_probability_text: str
    right_probability_text: str


@dataclass(frozen=True)
class GameStatsViewModel:
    total_predictions_text: str
    left_predictions_text: str
    right_predictions_text: str
    transitions_text: str
    confidence_text: str


@dataclass(frozen=True)
class TaskViewStateViewModel:
    record_button_text: str
    status_table_visible: bool
    game_visible: bool
    stats_visible: bool
    accuracy_visible: bool


class RecordingPresenter:
    @staticmethod
    def present(recording: Recording) -> RecordingViewModel:
        return RecordingViewModel(
            id=recording.id,
            patient_id=recording.patient_id,
            filename=recording.filename,
            task_type=recording.task_type,
            start_time=recording.start_time or "",
            end_time=recording.end_time or "",
            duration=recording.duration,
            notes=recording.notes,
        )


class SessionPresenter:
    @staticmethod
    def present(session: Session) -> SessionViewModel:
        return SessionViewModel(
            patient_id=session.patient_id,
            task_type=session.task_type,
            recording_id=session.recording_id,
            started_at_epoch=session.started_at_epoch,
            game_mode=session.game_mode,
        )


class MarkerPresenter:
    @staticmethod
    def present_state(state: MarkerState) -> MarkerStateViewModel:
        return MarkerStateViewModel(
            t1_count=state.t1_count,
            t2_count=state.t2_count,
            baseline_remaining_seconds=state.baseline_remaining_seconds,
            baseline_active=state.baseline_active,
        )

    @staticmethod
    def present_registration(result: dict) -> MarkerRegistrationViewModel:
        return MarkerRegistrationViewModel(
            accepted=bool(result["accepted"]),
            reason=result.get("reason"),
            state=MarkerPresenter.present_state(result["state"]),
            external_signal=result.get("external_signal"),
            esp32_direction=result.get("esp32_direction"),
            marker_type=result.get("marker_type"),
        )

    @staticmethod
    def present_baseline_tick(result: dict) -> BaselineTickViewModel:
        return BaselineTickViewModel(
            state=MarkerPresenter.present_state(result["state"]),
            finished=bool(result["finished"]),
        )


class InferencePresenter:
    @staticmethod
    def present_model(model: ModelMetadata) -> ModelViewModel:
        return ModelViewModel(
            path=model.path,
            name=model.name,
            backend=model.backend,
            input_shape=model.input_shape,
            expected_time_steps=model.expected_time_steps,
            expected_channels=model.expected_channels,
            modified_at_epoch=model.modified_at_epoch,
        )

    @staticmethod
    def present_prediction(result: PredictionResult) -> PredictionViewModel:
        return PredictionViewModel(
            predicted_index=result.predicted_index,
            confidence=result.confidence,
            probabilities=tuple(result.probabilities),
            left_probability=result.left_probability,
            right_probability=result.right_probability,
        )


class TrainingPresenter:
    @staticmethod
    def present(
        result: TrainingResult,
        *,
        auto_loaded: bool,
        loaded_model: Optional[ModelMetadata] = None,
    ) -> TrainingResultViewModel:
        return TrainingResultViewModel(
            model_path=result.model_path,
            training_time_seconds=result.training_time_seconds,
            final_accuracy=result.final_accuracy,
            final_loss=result.final_loss,
            val_accuracy=result.val_accuracy,
            val_loss=result.val_loss,
            auto_loaded=auto_loaded,
            loaded_model_path=loaded_model.path if loaded_model is not None else None,
            loaded_model_name=loaded_model.name if loaded_model is not None else None,
        )


class StreamingSessionStatePresenter:
    @staticmethod
    def present(
        *,
        session: Optional[SessionViewModel],
        marker_state: MarkerStateViewModel,
        loaded_model: Optional[ModelViewModel],
        recording_active: bool,
        eeg_connected: bool,
        eeg_mock_mode: bool,
        unity_server_active: bool,
        esp32_connected: bool,
    ) -> StreamingSessionStateViewModel:
        return StreamingSessionStateViewModel(
            patient_id=session.patient_id if session is not None else None,
            task_type=session.task_type if session is not None else None,
            recording_id=session.recording_id if session is not None else None,
            started_at_epoch=session.started_at_epoch if session is not None else None,
            game_mode=session.game_mode if session is not None else False,
            recording_active=recording_active,
            baseline_active=marker_state.baseline_active,
            baseline_remaining_seconds=marker_state.baseline_remaining_seconds,
            t1_count=marker_state.t1_count,
            t2_count=marker_state.t2_count,
            eeg_connected=eeg_connected,
            eeg_mock_mode=eeg_mock_mode,
            unity_server_active=unity_server_active,
            esp32_connected=esp32_connected,
            model_loaded=loaded_model is not None,
            model_name=loaded_model.name if loaded_model is not None else None,
        )


def _status_style(color: str) -> str:
    return f"color: {color}; font-size: 14px; font-weight: 700;"


def _find_transition_count(predictions: Sequence[Tuple[object, int, float]]) -> int:
    transitions = 0
    for index in range(1, len(predictions)):
        if predictions[index][1] != predictions[index - 1][1]:
            transitions += 1
    return transitions


class ConnectionStatusPresenter:
    _STATUS_MAP = {
        "eeg": {
            "standby": ("EEG - Standby", "#ffffff"),
            "connecting": ("EEG - Conectando...", "#f6ad55"),
            "connected": ("EEG - Conectado", "#48bb78"),
            "mock": ("EEG - Simulação", "#f6ad55"),
            "failed": ("EEG - Falha", "#fc8181"),
        },
        "vr": {
            "standby": ("VR - Standby", "#ffffff"),
            "connecting": ("VR - Conectando...", "#f6ad55"),
            "connected": ("VR - Conectado", "#48bb78"),
            "failed": ("VR - Falha", "#fc8181"),
        },
        "orthosis": {
            "standby": ("ORTESE - Standby", "#ffffff"),
            "connecting": ("ORTESE - Conectando...", "#f6ad55"),
            "connected": ("ORTESE - Conectado", "#48bb78"),
            "failed": ("ORTESE - Falha", "#fc8181"),
        },
    }

    @classmethod
    def present(
        cls,
        *,
        eeg_phase: str,
        vr_phase: str,
        orthosis_phase: str,
        connect_button_enabled: bool,
        record_button_enabled: bool,
    ) -> ConnectionPanelViewModel:
        eeg_state = cls._present_device("eeg", eeg_phase)
        return ConnectionPanelViewModel(
            eeg=eeg_state,
            vr=cls._present_device("vr", vr_phase),
            orthosis=cls._present_device("orthosis", orthosis_phase),
            connect_button_text=(
                "Desconectar"
                if eeg_phase in {"connecting", "connected", "mock"}
                else "Conectar"
            ),
            connect_button_enabled=connect_button_enabled,
            record_button_enabled=record_button_enabled
            and eeg_phase in {"connected", "mock"},
        )

    @classmethod
    def _present_device(cls, device: str, phase: str) -> LabelStateViewModel:
        text, color = cls._STATUS_MAP[device].get(
            phase, cls._STATUS_MAP[device]["standby"]
        )
        return LabelStateViewModel(text=text, style_sheet=_status_style(color))


class AccuracyPresenter:
    @staticmethod
    def parse_message(message: str) -> Optional[AccuracyTrialViewModel]:
        if "," not in message:
            return None

        parts = [part.strip() for part in message.strip().split(",")]
        if len(parts) != 2:
            return None

        flower_color, trigger_action = parts
        expected_action = {
            "RED_FLOWER": "LEFT",
            "BLUE_FLOWER": "RIGHT",
        }.get(flower_color)
        real_action = {
            "TRIGGER_ACTION_LEFT": "LEFT",
            "TRIGGER_ACTION_RIGHT": "RIGHT",
        }.get(trigger_action)
        if expected_action is None or real_action is None:
            return None

        return AccuracyTrialViewModel(
            expected_action=expected_action,
            real_action=real_action,
            is_correct=expected_action == real_action,
        )

    @staticmethod
    def present(trials: Sequence[AccuracyTrialViewModel]) -> AccuracyViewModel:
        total_count = len(trials)
        correct_count = sum(1 for trial in trials if trial.is_correct)
        accuracy_percent = (
            (correct_count / total_count) * 100 if total_count > 0 else 0.0
        )
        if accuracy_percent >= 80:
            color = "#4CAF50"
        elif accuracy_percent >= 60:
            color = "#FF9800"
        else:
            color = "#f44336"

        details_text = "Esperado vs Real"
        if trials:
            last_trial = trials[-1]
            status_text = "✓ Correto" if last_trial.is_correct else "✗ Erro"
            details_text = (
                f"Último: {last_trial.expected_action} vs "
                f"{last_trial.real_action} - {status_text}"
            )

        return AccuracyViewModel(
            summary_text=(
                f"Acurácia: {accuracy_percent:.1f}% ({correct_count}/{total_count})"
            ),
            details_text=details_text,
            summary_style_sheet=(
                f"font-size: 16px; font-weight: bold; color: {color};"
            ),
            correct_count=correct_count,
            total_count=total_count,
            accuracy_percent=accuracy_percent,
        )


class GameRuntimePresenter:
    @staticmethod
    def present_ai_status(state: str) -> AiStatusViewModel:
        mapping = {
            "waiting_task": ("🤖 IA: Aguardando tarefa", "gray"),
            "stopped": ("🤖 IA: Parada", "gray"),
            "active_fallback": ("🟡 IA: Ativa (fallback)", "orange"),
            "active_window": ("🟢 IA: Ativa (5s)", "green"),
            "inactive": ("🔴 IA: Inativa", "red"),
        }
        text, color = mapping.get(state, mapping["stopped"])
        return AiStatusViewModel(
            text=text,
            style_sheet=f"color: {color}; font-weight: bold; font-size: 12px;",
        )

    @staticmethod
    def present_prediction(
        prediction: Optional[PredictionViewModel],
    ) -> PredictionDisplayViewModel:
        if prediction is None:
            return PredictionDisplayViewModel(
                prediction_text="Aguardando predição...",
                prediction_style_sheet=(
                    "font-size: 24px; font-weight: bold; color: gray; padding: 10px;"
                ),
                left_probability_text="Mão Esquerda: 0%",
                right_probability_text="Mão Direita: 0%",
            )

        predicted_index = int(prediction.predicted_index)
        if predicted_index == 0:
            return PredictionDisplayViewModel(
                prediction_text="🤚 Mão Esquerda",
                prediction_style_sheet=(
                    "font-size: 24px; font-weight: bold; color: #2196F3; padding: 10px;"
                ),
                left_probability_text=(
                    f"Mão Esquerda: {float(prediction.left_probability):.1%}"
                ),
                right_probability_text=(
                    f"Mão Direita: {float(prediction.right_probability):.1%}"
                ),
            )

        return PredictionDisplayViewModel(
            prediction_text="✋ Mão Direita",
            prediction_style_sheet=(
                "font-size: 24px; font-weight: bold; color: #FF9800; padding: 10px;"
            ),
            left_probability_text=(
                f"Mão Esquerda: {float(prediction.left_probability):.1%}"
            ),
            right_probability_text=(
                f"Mão Direita: {float(prediction.right_probability):.1%}"
            ),
        )

    @staticmethod
    def present_stats(
        predictions: Sequence[Tuple[object, int, float]],
    ) -> GameStatsViewModel:
        total_predictions = len(predictions)
        left_count = sum(1 for _, prediction, _ in predictions if prediction == 0)
        right_count = sum(1 for _, prediction, _ in predictions if prediction == 1)
        transitions = _find_transition_count(predictions)
        confidence_average = (
            sum(float(confidence) for _, _, confidence in predictions)
            / total_predictions
            if total_predictions > 0
            else 0.0
        )
        return GameStatsViewModel(
            total_predictions_text=f"Total de predições: {total_predictions}",
            left_predictions_text=f"Mão esquerda: {left_count}",
            right_predictions_text=f"Mão direita: {right_count}",
            transitions_text=f"Transições: {transitions}",
            confidence_text=f"Confiança média: {confidence_average * 100:.1f}%",
        )


class TaskViewStatePresenter:
    @staticmethod
    def present(task: str, is_recording: bool) -> TaskViewStateViewModel:
        normalized_task = task.strip().lower()
        game_mode = normalized_task == "jogo"
        if is_recording:
            record_button_text = "Parar Jogo" if game_mode else "Parar Gravação"
        else:
            record_button_text = "Iniciar Jogo" if game_mode else "Iniciar Gravação"
        return TaskViewStateViewModel(
            record_button_text=record_button_text,
            status_table_visible=game_mode,
            game_visible=game_mode,
            stats_visible=game_mode,
            accuracy_visible=game_mode,
        )
