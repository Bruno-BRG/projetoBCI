from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    AccuracyPresenter,
    ConnectionStatusPresenter,
    GameRuntimePresenter,
    MarkerStateViewModel,
    ModelViewModel,
    PredictionViewModel,
    SessionViewModel,
    StreamingSessionStatePresenter,
    TaskViewStatePresenter,
)


def test_streaming_session_state_presenter_builds_explicit_screen_state():
    state = StreamingSessionStatePresenter.present(
        session=SessionViewModel(
            patient_id=4,
            task_type="jogo",
            recording_id=22,
            started_at_epoch=100.0,
            game_mode=True,
        ),
        marker_state=MarkerStateViewModel(
            t1_count=3,
            t2_count=1,
            baseline_remaining_seconds=40,
            baseline_active=True,
        ),
        loaded_model=ModelViewModel(
            path="C:/models/latest.keras",
            name="latest.keras",
            backend="tensorflow",
            input_shape=(None, 250, 16),
            expected_time_steps=250,
            expected_channels=16,
            modified_at_epoch=20.0,
        ),
        recording_active=True,
        eeg_connected=True,
        eeg_mock_mode=False,
        unity_server_active=True,
        esp32_connected=False,
    )

    assert state.patient_id == 4
    assert state.task_type == "jogo"
    assert state.recording_active is True
    assert state.baseline_active is True
    assert state.model_loaded is True
    assert state.model_name == "latest.keras"


def test_accuracy_presenter_parses_unity_message_and_formats_summary():
    trial = AccuracyPresenter.parse_message("RED_FLOWER,TRIGGER_ACTION_LEFT")

    assert trial is not None
    assert trial.expected_action == "LEFT"
    assert trial.real_action == "LEFT"
    assert trial.is_correct is True

    accuracy_view = AccuracyPresenter.present([trial])

    assert accuracy_view.summary_text == "Acurácia: 100.0% (1/1)"
    assert accuracy_view.details_text == "Último: LEFT vs LEFT - ✓ Correto"
    assert "#4CAF50" in accuracy_view.summary_style_sheet


def test_connection_status_presenter_maps_phases_to_visible_labels():
    panel = ConnectionStatusPresenter.present(
        eeg_phase="mock",
        vr_phase="connected",
        orthosis_phase="failed",
        connect_button_enabled=False,
        record_button_enabled=True,
    )

    assert panel.eeg.text == "EEG - Simulação"
    assert panel.vr.text == "VR - Conectado"
    assert panel.orthosis.text == "ORTESE - Falha"
    assert panel.connect_button_text == "Desconectar"
    assert panel.connect_button_enabled is False
    assert panel.record_button_enabled is True


def test_game_runtime_presenter_formats_prediction_and_task_view_state():
    prediction_view = GameRuntimePresenter.present_prediction(
        PredictionViewModel(
            predicted_index=1,
            confidence=0.91,
            probabilities=(0.09, 0.91),
            left_probability=0.09,
            right_probability=0.91,
        )
    )
    task_view = TaskViewStatePresenter.present("Jogo", is_recording=False)

    assert prediction_view.prediction_text == "✋ Mão Direita"
    assert prediction_view.right_probability_text == "Mão Direita: 91.0%"
    assert "#FF9800" in prediction_view.prediction_style_sheet
    assert task_view.record_button_text == "Iniciar Jogo"
    assert task_view.game_visible is True
