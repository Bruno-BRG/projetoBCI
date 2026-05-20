import os
from datetime import datetime
from typing import List
from collections import deque
import numpy as np
import time
import importlib
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QGroupBox, QComboBox, QGridLayout,
                           QMessageBox, QCheckBox,
                           QLineEdit, QSpinBox, QDialog, QInputDialog, QFrame)
from PyQt5.QtCore import pyqtSignal, QTimer, Qt
from brainbridge_v2.application.game_inference_coordinator import (
    GameInferenceCoordinator,
)
from brainbridge_v2.application.eeg_quality import EEGWindowQualityValidator
from brainbridge_v2.application.pipeline_telemetry import PipelineTelemetry
from brainbridge_v2.application.runtime_config import DEFAULT_RUNTIME_CONFIG
from brainbridge_v2.application.unity_command_mapper import UnityCommandMapper
from brainbridge_v2.infrastructure.config.settings import get_recording_path
from brainbridge_v2.interface_adapters.controllers.eeg_stream_controller import (
    EEGStreamController,
)
from brainbridge_v2.interface_adapters.controllers.esp32_controller import (
    ESP32Controller,
)
from brainbridge_v2.interface_adapters.controllers.inference_controller import (
    InferenceController,
)
from brainbridge_v2.interface_adapters.controllers.marker_controller import (
    MarkerController,
)
from brainbridge_v2.interface_adapters.controllers.patient_controller import (
    PatientController,
)
from brainbridge_v2.interface_adapters.controllers.recording_controller import (
    RecordingController,
)
from brainbridge_v2.interface_adapters.controllers.session_controller import (
    SessionController,
)
from brainbridge_v2.interface_adapters.controllers.training_controller import (
    TrainingController,
)
from brainbridge_v2.interface_adapters.controllers.unity_controller import (
    UnityController,
)
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    AccuracyPresenter,
    AccuracyTrialViewModel,
    ConnectionStatusPresenter,
    GameRuntimePresenter,
    MarkerStateViewModel,
    ModelViewModel,
    PredictionViewModel,
    SessionViewModel,
    StartRecordingRequest,
    StartSessionRequest,
    StreamingSessionStatePresenter,
    StreamingSessionStateViewModel,
    TaskViewStatePresenter,
)
from brainbridge_v2.presentation.gui.widgets.eeg_plot import EEGPlotWidget
from brainbridge_v2.presentation.gui.styles import Theme

from brainbridge_v2.presentation.gui.dialogs.training_dialog import TrainingDialog

# Importar logger do novo módulo (compatível com OpenBCI)
try:
    from brainbridge_v2.infrastructure.acquisition.data_logger import OpenBCICSVLogger
    USE_OPENBCI_LOGGER = True
except Exception:
    USE_OPENBCI_LOGGER = False


class DeveloperSettingsDialog(QDialog):
    def __init__(self, *, telemetry_enabled: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Modo Desenvolvedor")
        self.setModal(True)
        self.resize(360, 160)
        self.setStyleSheet(Theme.get_stylesheet())
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Configurações de Desenvolvimento")
        title.setStyleSheet(Theme.section_title("15px"))
        layout.addWidget(title)

        self.telemetry_checkbox = QCheckBox("Ativar telemetria da IA")
        self.telemetry_checkbox.setChecked(bool(telemetry_enabled))
        layout.addWidget(self.telemetry_checkbox)

        buttons = QHBoxLayout()
        cancel_btn = QPushButton("Cancelar")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Aplicar")
        apply_btn.setStyleSheet(Theme.btn_blue("6px 16px", "13px", "700"))
        apply_btn.clicked.connect(self.accept)
        buttons.addStretch()
        buttons.addWidget(cancel_btn)
        buttons.addWidget(apply_btn)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def telemetry_enabled(self) -> bool:
        return self.telemetry_checkbox.isChecked()


class StreamingWidget(QWidget):
    """Widget para streaming e gravação de dados"""
    
    # Signal para processar mensagens de acurácia de forma thread-safe
    accuracy_message_signal = pyqtSignal(str)
    
    def __init__(
        self,
        eeg_stream_controller: EEGStreamController,
        inference_controller: InferenceController,
        training_controller: TrainingController,
        patient_controller: PatientController,
        recording_controller: RecordingController,
        session_controller: SessionController,
        marker_controller: MarkerController,
        unity_controller: UnityController,
        esp32_controller: ESP32Controller,
        parent=None,
    ):
        super().__init__(parent)
        self.eeg_stream_controller = eeg_stream_controller
        self.inference_controller = inference_controller
        self.training_controller = training_controller
        self.patient_controller = patient_controller
        self.recording_controller = recording_controller
        self.session_controller = session_controller
        self.marker_controller = marker_controller
        self.unity_controller = unity_controller
        self.esp32_controller = esp32_controller
        self.connect_button_enabled = True
        self.record_button_enabled = False
        self.eeg_connection_phase = "standby"
        self.unity_connection_phase = "standby"
        self.orthosis_connection_phase = "standby"
        self.disconnection_in_progress = False

    # Streaming / logging state
        self.csv_logger = None
        self.is_recording = False
        self.current_recording_id = None
        self.pending_marker = None  # Para marcadores pendentes no logger OpenBCI
        self.baseline_timer = QTimer()  # Timer para baseline

    # Timer de sessão
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_session_timer)
        self.session_elapsed_seconds = 0

    # Configuracao base do modelo
        # Force canonical window_size to 250 (HardThinking canonical)
        self.window_size = DEFAULT_RUNTIME_CONFIG.window_size  # 2s @ 125Hz
        self.channels = DEFAULT_RUNTIME_CONFIG.channels
        try:
            # HardThinking config module was added to sys.path earlier when locating adapter
            _ht_cfg_mod = importlib.import_module('config')
            _ht_get_config = getattr(_ht_cfg_mod, 'get_config', None)
            if _ht_get_config:
                _cfg = _ht_get_config()
                self.window_size = int(_cfg.data.window_size)
                self.channels = int(_cfg.data.channels)
        except Exception:
            # keep fallbacks
            pass

        self.predictions = deque(maxlen=50)  # Últimas predições

    # Estados do servidor UDP
        self.udp_server_active = False

    # Inicializar callbacks de comunicação
        self.eeg_stream_controller.set_data_callback(self.on_data_received)
        self.eeg_stream_controller.set_connection_callback(self.on_connection_status)
        self.unity_controller.set_message_callback(self._on_unity_message)
        self.unity_controller.set_connection_callback(self._on_unity_connection)
        self.esp32_controller.set_connection_callback(self._on_esp32_connection)
        self.esp32_connected = False

    # Timer para ações automáticas no jogo
        self.game_action_timer = QTimer()
        self.game_action_timer.timeout.connect(self.game_random_action)

    # Controle para aguardar resposta antes do próximo sinal
        self.waiting_for_response = False

    # Controle de janela de tempo para IA (configurável; default reduzido)
        self.ai_prediction_enabled = False
        self.task_start_time = None
        # Reduce default AI window to 2 seconds to send triggers sooner (milliseconds)
        self.ai_window_duration = DEFAULT_RUNTIME_CONFIG.ai_window_duration_ms
        self.game_inference = GameInferenceCoordinator(
            window_size=self.window_size,
            channels=self.channels,
            window_duration_ms=self.ai_window_duration,
        )
        self.developer_mode_enabled = False
        self.pipeline_telemetry = PipelineTelemetry(enabled=False)
        self.eeg_quality_validator = EEGWindowQualityValidator(
            max_abs_amplitude=DEFAULT_RUNTIME_CONFIG.eeg_max_abs_amplitude,
            min_channel_std=DEFAULT_RUNTIME_CONFIG.eeg_min_channel_std,
        )
        self.eeg_buffer = self.game_inference.eeg_buffer
        self.samples_since_last_prediction = self.game_inference.samples_since_window_start
        self.prediction_locked = self.game_inference.prediction_locked
        # Fallback interval for automatic game actions (was 30s); reduce to 10s
        self.game_action_interval = DEFAULT_RUNTIME_CONFIG.game_action_interval_ms

    # Variáveis para cálculo de acurácia
        self.accuracy_trials: List[AccuracyTrialViewModel] = []

    # UDP receiver para acurácia (recebe mensagens do sistema externo)
        self.accuracy_udp_receiver = None
        self.accuracy_thread = None

    # Conectar signal para processar mensagens de acurácia
        self.accuracy_message_signal.connect(self.process_accuracy_message)
        self.streaming_state = StreamingSessionStateViewModel(
            patient_id=None,
            task_type=None,
            recording_id=None,
            started_at_epoch=None,
            game_mode=False,
            recording_active=False,
            baseline_active=False,
            baseline_remaining_seconds=0,
            t1_count=0,
            t2_count=0,
            eeg_connected=False,
            eeg_mock_mode=False,
            unity_server_active=False,
            esp32_connected=False,
            model_loaded=False,
            model_name=None,
        )
        self.setup_ui()
        self._refresh_streaming_state()

    def _get_current_session(self):
        return self.session_controller.get_current_session()

    def _refresh_streaming_state(
        self,
        *,
        session: SessionViewModel | None = None,
        marker_state: MarkerStateViewModel | None = None,
        loaded_model: ModelViewModel | None = None,
    ) -> StreamingSessionStateViewModel:
        current_session = session if session is not None else self.session_controller.get_current_session()
        current_marker_state = (
            marker_state if marker_state is not None else self.marker_controller.get_state()
        )
        current_loaded_model = (
            loaded_model
            if loaded_model is not None
            else self.inference_controller.get_loaded_model()
        )
        self.streaming_state = StreamingSessionStatePresenter.present(
            session=current_session,
            marker_state=current_marker_state,
            loaded_model=current_loaded_model,
            recording_active=self.is_recording,
            eeg_connected=self.eeg_stream_controller.is_running(),
            eeg_mock_mode=self.eeg_stream_controller.is_mock_mode(),
            unity_server_active=self.udp_server_active,
            esp32_connected=self.esp32_connected,
        )
        return self.streaming_state

    def _get_session_started_at(self):
        return self.streaming_state.started_at_epoch

    def _is_game_mode(self) -> bool:
        return self.streaming_state.game_mode

    def _sync_ai_prediction_state(self):
        self.eeg_buffer = self.game_inference.eeg_buffer
        self.samples_since_last_prediction = (
            self.game_inference.samples_since_window_start
        )
        self.ai_prediction_enabled = self.game_inference.is_window_open
        self.prediction_locked = self.game_inference.prediction_locked
        self.task_start_time = self.game_inference.window_started_at_ms

    def _reset_ai_prediction_window(self):
        self.game_inference.reset()
        self._sync_ai_prediction_state()

    def _record_pipeline_event(self, name: str, **details):
        try:
            self.pipeline_telemetry.record(name, **details)
        except Exception as exc:
            print(f"[PIPELINE] Falha ao registrar evento {name}: {exc}")

    def open_developer_settings(self):
        dialog = DeveloperSettingsDialog(
            telemetry_enabled=self.developer_mode_enabled,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            self.set_developer_mode(dialog.telemetry_enabled())

    def set_developer_mode(self, enabled: bool):
        self.developer_mode_enabled = bool(enabled)
        self.pipeline_telemetry.set_enabled(self.developer_mode_enabled)
        if hasattr(self, "developer_settings_btn"):
            label = "Dev: On" if self.developer_mode_enabled else "Dev: Off"
            self.developer_settings_btn.setText(label)
            self.developer_settings_btn.setStyleSheet(
                Theme.btn_dev(self.developer_mode_enabled)
            )
        
    @staticmethod
    def _v_separator():
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet(Theme.vertical_separator())
        return line

    def setup_ui(self):
        """Configura a interface conforme bci_system.html (paleta Theme)."""
        T = Theme
        task_btn = T.btn_default("8px 20px", "14px", "700")
        task_btn_jogo = T.btn_default("8px 24px", "14px", "700")
        calib_btn = T.btn_default("6px 14px", "13px", "600")
        calib_btn_sm = T.btn_default("6px 14px", "12px", "600")
        connect_sm = T.btn_green("4px 10px", "11px", "600") + " border-radius: 4px;"
        combo_style = (
            f"padding: 4px 8px; font-size: 13px; background: {T.BTN_BG}; color: {T.TEXT_DARK}; "
            f"border: 1px solid {T.BTN_BORDER}; border-radius: 4px; font-weight: 600;"
        )
        patient_title = (
            f"color: {T.WHITE}; font-size: 22px; font-weight: 800; "
            "letter-spacing: 0.5px; background: transparent;"
        )
        subtitle_18 = f"color: {T.WHITE}; font-size: 18px; font-weight: 800; background: transparent;"
        subtitle_16 = f"color: {T.WHITE}; font-size: 16px; font-weight: 800; background: transparent;"
        bci_status = (
            f"color: {T.GREEN}; font-size: 22px; font-weight: 800; "
            "letter-spacing: 0.5px; padding: 14px 0 8px 0; background: transparent;"
        )
        session_timer = (
            f"color: {T.WHITE}; font-size: 20px; font-weight: 800; "
            "letter-spacing: 0.5px; padding: 12px 0 16px 0; background: transparent;"
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(0)

        top_container = QWidget()
        top_container.setStyleSheet(T.panel())
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(20, 16, 20, 12)
        top_layout.setSpacing(12)

        # ---- COL LEFT: .col-left { flex-direction: column; gap: 10px } ----
        col_left = QVBoxLayout()
        col_left.setSpacing(10)

        # .patient-label { font-size: 22px; font-weight: 800; letter-spacing: 0.5px }
        self.patient_display_label = QLabel("Paciente: ####")
        self.patient_display_label.setStyleSheet(patient_title)
        col_left.addWidget(self.patient_display_label)

        # div: display flex, align-items stretch, gap 14px, margin-top 6px
        left_mid = QHBoxLayout()
        left_mid.setSpacing(14)
        left_mid.setContentsMargins(0, 6, 0, 0)

        # Baseline + Jogo: flex-direction column, gap 10px, justify-content center
        task_col = QVBoxLayout()
        task_col.setSpacing(10)
        # button.btn: padding 8px 20px, font-size 14px, font-weight 700, width 100%
        self.btn_baseline = QPushButton("Baseline")
        self.btn_baseline.setStyleSheet(task_btn)
        self.btn_baseline.clicked.connect(lambda: self._set_task("Baseline"))
        self.btn_jogo = QPushButton("Jogo")
        self.btn_jogo.setStyleSheet(task_btn_jogo)
        self.btn_jogo.clicked.connect(lambda: self._set_task("Jogo"))
        task_col.addWidget(self.btn_baseline)
        task_col.addWidget(self.btn_jogo)
        left_mid.addLayout(task_col)

        # .calibration-box { border: 2px solid #4a5568; border-radius: 6px; padding: 8px 14px;
        #   flex-direction column; align-items center; gap 6px; background rgba(45,55,72,0.4) }
        calib_frame = QWidget()
        calib_frame.setStyleSheet(T.calibration_box())
        calib_inner = QVBoxLayout()
        calib_inner.setContentsMargins(14, 8, 14, 8)
        calib_inner.setSpacing(6)
        # button.btn: font-size 13px, font-weight 600
        self.btn_iniciar_treino = QPushButton("Iniciar Treino")
        self.btn_iniciar_treino.setStyleSheet(calib_btn)
        self.btn_iniciar_treino.clicked.connect(lambda: self._set_task("Treino"))
        calib_title = QLabel("Calibração")
        calib_title.setStyleSheet(
            f"font-size: 16px; font-weight: 700; color: {T.WHITE}; border: none; background: transparent;"
        )
        calib_title.setAlignment(Qt.AlignCenter)
        # .calibration-btns { display flex; gap 8px }
        calib_btns_row = QHBoxLayout()
        calib_btns_row.setSpacing(8)
        # button.btn: font-size 12px
        self.btn_calib_esq = QPushButton("Esquerda")
        self.btn_calib_esq.setStyleSheet(calib_btn_sm)
        self.btn_calib_esq.clicked.connect(lambda: self.add_marker("T1"))
        self.btn_calib_dir = QPushButton("Direita")
        self.btn_calib_dir.setStyleSheet(calib_btn_sm)
        self.btn_calib_dir.clicked.connect(lambda: self.add_marker("T2"))
        calib_btns_row.addWidget(self.btn_calib_esq)
        calib_btns_row.addWidget(self.btn_calib_dir)
        calib_inner.addWidget(self.btn_iniciar_treino)
        calib_inner.addWidget(calib_title)
        calib_inner.addLayout(calib_btns_row)
        calib_frame.setLayout(calib_inner)
        left_mid.addWidget(calib_frame)
        left_mid.addStretch()

        col_left.addLayout(left_mid)
        col_left.addStretch()
        top_layout.addLayout(col_left, 1)
        top_layout.addWidget(self._v_separator())

        # ---- COL CENTER ----
        col_center = QVBoxLayout()
        col_center.setSpacing(6)
        col_center.setContentsMargins(10, 0, 0, 0)

        # .status-title { font-size: 22px; font-weight: 800 }
        status_title = QLabel("Status")
        status_title.setStyleSheet(T.section_title())
        col_center.addWidget(status_title)

        self.connect_btn = QPushButton("Conectar")
        self.connect_btn.setStyleSheet(T.btn_green())
        self.connect_btn.clicked.connect(self.toggle_connection)
        col_center.addWidget(self.connect_btn, 0, Qt.AlignLeft)

        self.developer_settings_btn = QPushButton("Dev: Off")
        self.developer_settings_btn.setStyleSheet(T.btn_dev(False))
        self.developer_settings_btn.clicked.connect(self.open_developer_settings)
        col_center.addWidget(self.developer_settings_btn, 0, Qt.AlignLeft)

        # .status-list { flex-direction column; gap 3px; margin-top 4px }
        # .status-item { font-size: 14px; font-weight: 700 }
        status_grid = QGridLayout()
        status_grid.setSpacing(8)
        status_grid.setContentsMargins(0, 4, 0, 0)
        
        self.status_eeg = QLabel("EEG - Standby")
        self.status_eeg.setStyleSheet(T.status_text("off"))
        self.connect_eeg_btn = QPushButton("Conectar")
        self.connect_eeg_btn.setStyleSheet(connect_sm)
        self.connect_eeg_btn.clicked.connect(self.toggle_eeg_connection)
        
        self.status_vr = QLabel("VR - Standby")
        self.status_vr.setStyleSheet(T.status_text("off"))
        self.connect_vr_btn = QPushButton("Conectar")
        self.connect_vr_btn.setStyleSheet(connect_sm)
        self.connect_vr_btn.clicked.connect(self.toggle_udp_server)
        
        self.status_ortese = QLabel("ORTESE - Standby")
        self.status_ortese.setStyleSheet(T.status_text("off"))
        self.connect_ortese_btn = QPushButton("Conectar")
        self.connect_ortese_btn.setStyleSheet(connect_sm)
        self.connect_ortese_btn.clicked.connect(self.toggle_esp32_connection)
        
        status_grid.addWidget(self.status_eeg, 0, 0)
        status_grid.addWidget(self.connect_eeg_btn, 0, 1)
        status_grid.addWidget(self.status_vr, 1, 0)
        status_grid.addWidget(self.connect_vr_btn, 1, 1)
        status_grid.addWidget(self.status_ortese, 2, 0)
        status_grid.addWidget(self.connect_ortese_btn, 2, 1)
        
        col_center.addLayout(status_grid)
        col_center.addStretch()
        top_layout.addLayout(col_center, 1)
        top_layout.addWidget(self._v_separator())

        # ---- COL RIGHT ----
        # Internamente: .right-panel-top { display flex; align-items flex-start; justify-content space-between }
        col_right = QVBoxLayout()
        col_right.setSpacing(8)

        right_top = QHBoxLayout()
        right_top.setSpacing(16)

        # .right-panel-content { flex-direction column; gap 6px }
        grav_col = QVBoxLayout()
        grav_col.setSpacing(6)

        # .gravacao-title { font-size: 18px; font-weight: 800 }
        gravacao_title = QLabel("Gravação")
        gravacao_title.setStyleSheet(subtitle_18)
        grav_col.addWidget(gravacao_title)

        # .gravacao-row { display flex; align-items center; gap 8px }
        pac_row = QHBoxLayout()
        pac_row.setSpacing(8)
        # .gravacao-label { font-size: 14px; font-weight: 700 }
        pac_label = QLabel("Paciente")
        pac_label.setStyleSheet(T.status_text("default") + " font-size: 14px;")
        self.patient_combo = QComboBox()
        self.patient_combo.setStyleSheet(combo_style)
        self.patient_combo.setMaximumWidth(100)
        self.patient_combo.currentTextChanged.connect(self._on_patient_changed)
        pac_row.addWidget(pac_label)
        pac_row.addWidget(self.patient_combo)
        grav_col.addLayout(pac_row)

        # button.btn.btn-atualizar: padding 5px 14px, font-size 12px
        self.refresh_patients_btn = QPushButton("Atualizar")
        self.refresh_patients_btn.setStyleSheet(T.btn_default("5px 14px", "12px", "600"))
        self.refresh_patients_btn.clicked.connect(self.refresh_patients)
        grav_col.addWidget(self.refresh_patients_btn)

        # .gravacao-actions { display flex; align-items center; gap 10px; margin-top 2px }
        grav_actions = QHBoxLayout()
        grav_actions.setSpacing(10)
        grav_actions.setContentsMargins(0, 2, 0, 0)
        # button.btn.btn-green: font-size 12px, padding 5px 14px
        self.record_btn = QPushButton("Iniciar Gravação")
        self.record_btn.setStyleSheet(T.btn_green("5px 14px", "12px", "600"))
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        self.gravacao_status = QLabel("Não gravando")
        self.gravacao_status.setStyleSheet(Theme.recording_status_label())
        self.gravacao_status.setWordWrap(False)
        grav_actions.addWidget(self.record_btn)
        grav_actions.addWidget(self.gravacao_status)
        grav_col.addLayout(grav_actions)
        right_top.addLayout(grav_col)

        # Hands + IA Table — .right-side-panel { flex-direction column; align-items flex-end; gap 6px }
        ia_col = QVBoxLayout()
        ia_col.setSpacing(4)
        ia_col.setContentsMargins(0, 0, 0, 0)

        # .hand-icons { display flex; gap 16px (style override: gap 24px); justify-content center }
        # .hand-icon { font-size: 32px; filter drop-shadow }
        hands_row = QHBoxLayout()
        hands_row.setSpacing(24)
        hands_row.addStretch()
        hand_left = QLabel("✋")
        # .hand-icon.left { color: #f6ad55 }
        hand_left.setStyleSheet(f"font-size: 32px; color: {T.ORANGE}; background: transparent;")
        hand_right = QLabel("🤚")
        hand_right.setStyleSheet(f"font-size: 32px; color: {T.LIGHT_BLUE}; background: transparent;")
        hands_row.addWidget(hand_left)
        hands_row.addWidget(hand_right)
        hands_row.addStretch()
        ia_col.addLayout(hands_row)

        # .ia-table-area: grid com .ia-row grid-template-columns 80px 40px 40px
        # .ia-cell { padding 4px 6px; font-size 13px; font-weight 700 }
        # .ia-cell-label { justify-content flex-end; padding-right 10px }
        # .dot { width 14px; height 14px; border-radius 50%; background #2d3748 }
        # .acertos-label { color: #f6ad55; font-weight: 800 }
        # .acerto-val { font-size: 18px; font-weight: 800 }
        ia_table = QGridLayout()
        ia_table.setSpacing(0)
        ia_table.setContentsMargins(0, 4, 0, 0)
        ia_table.setColumnMinimumWidth(0, 80)
        ia_table.setColumnMinimumWidth(1, 40)
        ia_table.setColumnMinimumWidth(2, 40)

        DOT_INACTIVE = "●"
        dot_style      = f"color: {T.BTN_DARK}; font-size: 14px; background: transparent;"
        label_style    = (
            f"color: {T.WHITE}; font-size: 13px; font-weight: 700; "
            "padding: 4px 10px 4px 6px; background: transparent;"
        )
        acertos_style  = (
            f"color: {T.ORANGE}; font-size: 13px; font-weight: 800; "
            "padding: 4px 10px 4px 6px; background: transparent;"
        )
        cell_style     = (
            "font-size: 13px; font-weight: 700; border: 1px solid rgba(74, 85, 104, 0.3); "
            "padding: 4px 6px; background: transparent;"
        )
        val_style      = (
            f"color: {T.WHITE}; font-size: 18px; font-weight: 800; "
            "border: 1px solid rgba(74, 85, 104, 0.3); padding: 4px 6px; background: transparent;"
        )

        table_rows = [
            ("IA",       label_style,   DOT_INACTIVE, dot_style + cell_style, DOT_INACTIVE, dot_style + cell_style),
            ("Paciente", label_style,   DOT_INACTIVE, dot_style + cell_style, DOT_INACTIVE, dot_style + cell_style),
            ("Tarefa",   label_style,   DOT_INACTIVE, dot_style + cell_style, DOT_INACTIVE, dot_style + cell_style),
            ("Acertos",  acertos_style, "0",          val_style,              "0",          val_style),
        ]
        self.ia_table_cells = {}
        for r, (name, ls, v1, v1s, v2, v2s) in enumerate(table_rows):
            lbl = QLabel(name)
            lbl.setStyleSheet(ls)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            c1 = QLabel(v1)
            c1.setStyleSheet(v1s)
            c1.setAlignment(Qt.AlignCenter)
            c2 = QLabel(v2)
            c2.setStyleSheet(v2s)
            c2.setAlignment(Qt.AlignCenter)
            ia_table.addWidget(lbl, r, 0)
            ia_table.addWidget(c1, r, 1)
            ia_table.addWidget(c2, r, 2)
            self.ia_table_cells[name] = (lbl, c1, c2)

        ia_col.addLayout(ia_table)
        ia_col.addStretch()
        right_top.addLayout(ia_col)

        col_right.addLayout(right_top)
        col_right.addStretch()
        top_layout.addLayout(col_right, 1)

        top_container.setLayout(top_layout)
        layout.addWidget(top_container)

        # ============ MARCADORES BAR ============
        marcadores_bar = QWidget()
        marcadores_bar.setStyleSheet(T.marcadores_bar())
        marc_layout = QHBoxLayout()
        marc_layout.setContentsMargins(20, 12, 20, 12)

        self.marcador_text = QLabel("Marcadores -  T1: 0  |  T2: 0")
        self.marcador_text.setStyleSheet(subtitle_18)
        marc_layout.addWidget(self.marcador_text)
        marc_layout.addStretch()

        teste_label = QLabel("Teste Manual")
        teste_label.setStyleSheet(subtitle_16)
        marc_layout.addWidget(teste_label)
        marc_layout.addSpacing(12)

        self.t1_btn = QPushButton("T1")
        self.t1_btn.setStyleSheet(T.btn_dark())
        self.t1_btn.clicked.connect(lambda: self.add_marker("T1"))
        self.t2_btn = QPushButton("T2")
        self.t2_btn.setStyleSheet(T.btn_blue())
        self.t2_btn.clicked.connect(lambda: self.add_marker("T2"))
        marc_layout.addWidget(self.t1_btn)
        marc_layout.addSpacing(8)
        marc_layout.addWidget(self.t2_btn)

        marcadores_bar.setLayout(marc_layout)
        layout.addWidget(marcadores_bar)

        # ============ BCI STATUS ============
        self.bci_status_label = QLabel("Sistema BCI inicializado")
        self.bci_status_label.setStyleSheet(bci_status)
        self.bci_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.bci_status_label)

        # ============ EEG CHART ============
        self.plot_widget = EEGPlotWidget()
        self.plot_widget.setMinimumHeight(300)
        layout.addWidget(self.plot_widget)

        # ============ SESSION TIMER ============
        self.session_timer_label = QLabel("Sessão: 00:00:00")
        self.session_timer_label.setStyleSheet(session_timer)
        self.session_timer_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.session_timer_label)

        self.setLayout(layout)

        # ============ Widgets internos (não visíveis, para compatibilidade) ============
        self.host_edit = QLineEdit("localhost")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(12345)
        self.status_label = self.status_eeg  # alias
        self.recording_label = self.gravacao_status  # alias
        self.t1_counter_label = QLabel("T1: 0")
        self.t2_counter_label = QLabel("T2: 0")
        self.baseline_label = QLabel("")
        self.baseline_timer = QTimer()
        self.baseline_timer.timeout.connect(self.update_baseline_timer)

        # task_combo interno (não visível) para compatibilidade com on_task_changed
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Baseline", "Treino", "Teste", "Jogo"])
        self.task_combo.currentTextChanged.connect(self.on_task_changed)

        # Game mode labels (ocultos, para compatibilidade)
        self.accuracy_details_label = QLabel("")
        self.accuracy_group = QWidget()
        self.status_table_group = QWidget()
        self.total_predictions_label = QLabel("0")
        self.left_predictions_label = QLabel("0")
        self.right_predictions_label = QLabel("0")
        self.transitions_label = QLabel("0")
        self.confidence_label = QLabel("0%")
        self.stats_group = QWidget()
        self.game_group = QWidget()
        self.prediction_label = QLabel("")
        self.prob_left_label = QLabel("")
        self.prob_right_label = QLabel("")
        self.model_status_label = QLabel("")
        self.accuracy_label = QLabel("Acurácia: 0% (0/0)")
        self.ai_status_label = QLabel("")

        # Inicializar UDP auto-send checkboxes (para compatibilidade)
        self.udp_auto_send_checkbox = QCheckBox()
        self.udp_auto_send_checkbox.setChecked(True)
        self.esp32_auto_send_checkbox = QCheckBox()
        self.esp32_auto_send_checkbox.setChecked(True)
        self.udp_toggle_btn = QPushButton("")
        self.udp_status_label = QLabel("")
        self.udp_test_left_btn = QPushButton("")
        self.udp_test_right_btn = QPushButton("")
        self.esp32_toggle_btn = QPushButton("")
        self.esp32_status_label = QLabel("")
        self.esp32_test_left_btn = QPushButton("")
        self.esp32_test_right_btn = QPushButton("")

        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_game_stats)
        self.stats_timer.start(1000)

        self._apply_connection_panel()
        self._apply_task_view_state()
        self._apply_accuracy_display()
        self._apply_prediction_display()
        self._set_ai_status("stopped")
        self.refresh_patients()

    def _set_task(self, task):
        """Muda a tarefa ativa via botão"""
        idx = self.task_combo.findText(task)
        if idx >= 0:
            self.task_combo.setCurrentIndex(idx)

    def _on_patient_changed(self, text):
        """Atualiza o label de paciente quando muda no combo"""
        if text and text != "Selecionar paciente...":
            self.patient_display_label.setText(f"Paciente: {text}")
        else:
            self.patient_display_label.setText("Paciente: ####")

    def _update_marcador_text(self):
        """Atualiza o texto dos marcadores na barra"""
        self.marcador_text.setText(
            f"Marcadores -  T1: {self.streaming_state.t1_count}  |  T2: {self.streaming_state.t2_count}"
        )

    def _update_marker_labels(self, state: MarkerStateViewModel):
        self._refresh_streaming_state(marker_state=state)
        self.t1_counter_label.setText(f"T1: {state.t1_count}")
        self.t2_counter_label.setText(f"T2: {state.t2_count}")
        self._update_marcador_text()

    def _apply_status_label(self, label: QLabel, text: str, style_sheet: str):
        label.setText(text)
        label.setStyleSheet(style_sheet)

    def _apply_connection_panel(self):
        panel = ConnectionStatusPresenter.present(
            eeg_phase=self.eeg_connection_phase,
            vr_phase=self.unity_connection_phase,
            orthosis_phase=self.orthosis_connection_phase,
            connect_button_enabled=self.connect_button_enabled,
            record_button_enabled=self.record_button_enabled,
        )
        self._apply_status_label(self.status_eeg, panel.eeg.text, panel.eeg.style_sheet)
        self._apply_status_label(self.status_vr, panel.vr.text, panel.vr.style_sheet)
        self._apply_status_label(
            self.status_ortese,
            panel.orthosis.text,
            panel.orthosis.style_sheet,
        )
        self.connect_btn.setText(panel.connect_button_text)
        self.connect_btn.setStyleSheet(panel.connect_button_style)
        self.connect_btn.setEnabled(panel.connect_button_enabled)
        
        # Atualizar botões individuais
        if hasattr(self, 'connect_eeg_btn'):
            self.connect_eeg_btn.setText(panel.eeg_button_text)
            self.connect_eeg_btn.setStyleSheet(panel.eeg_button_style)
            self.connect_eeg_btn.setEnabled(panel.connect_button_enabled)
        if hasattr(self, 'connect_vr_btn'):
            self.connect_vr_btn.setText(panel.vr_button_text)
            self.connect_vr_btn.setStyleSheet(panel.vr_button_style)
            self.connect_vr_btn.setEnabled(panel.connect_button_enabled)
        if hasattr(self, 'connect_ortese_btn'):
            self.connect_ortese_btn.setText(panel.orthosis_button_text)
            self.connect_ortese_btn.setStyleSheet(panel.orthosis_button_style)
            self.connect_ortese_btn.setEnabled(panel.connect_button_enabled)
            
        self.record_btn.setEnabled(panel.record_button_enabled)

    def _recording_status_text(self, hint: str = "") -> str:
        task = self.task_combo.currentText()
        if not self.is_recording:
            return "Não gravando"
        base = "Jogando" if task == "Jogo" else "Gravando"
        if hint:
            return f"{base} · {hint}"
        return base

    def _apply_recording_ui(self, hint: str = ""):
        """Atualiza botão e texto curto de gravação (sem caminhos de arquivo)."""
        if not hasattr(self, "record_btn"):
            return
        self._apply_task_view_state()
        if self.is_recording:
            self.record_btn.setStyleSheet(Theme.btn_recording_active("5px 14px", "12px", "600"))
        elif self.record_btn.isEnabled():
            self.record_btn.setStyleSheet(Theme.btn_green("5px 14px", "12px", "600"))
        if hasattr(self, "gravacao_status"):
            self.gravacao_status.setText(self._recording_status_text(hint))
            self.gravacao_status.setStyleSheet(Theme.recording_status_label())

    def _apply_task_view_state(self):
        task_view = TaskViewStatePresenter.present(
            self.task_combo.currentText(),
            self.is_recording,
        )
        self.record_btn.setText(task_view.record_button_text)
        self.status_table_group.setVisible(task_view.status_table_visible)
        self.game_group.setVisible(task_view.game_visible)
        self.stats_group.setVisible(task_view.stats_visible)
        self.accuracy_group.setVisible(task_view.accuracy_visible)

    def _apply_accuracy_display(self):
        accuracy_view = AccuracyPresenter.present(self.accuracy_trials)
        self.accuracy_label.setText(accuracy_view.summary_text)
        self.accuracy_label.setStyleSheet(accuracy_view.summary_style_sheet)
        self.accuracy_details_label.setText(accuracy_view.details_text)

    def _set_ai_status(self, state: str):
        ai_status = GameRuntimePresenter.present_ai_status(state)
        self.ai_status_label.setText(ai_status.text)
        self.ai_status_label.setStyleSheet(ai_status.style_sheet)

    def _apply_prediction_display(
        self,
        prediction: PredictionViewModel | None = None,
    ):
        prediction_view = GameRuntimePresenter.present_prediction(prediction)
        self.prediction_label.setText(prediction_view.prediction_text)
        self.prediction_label.setStyleSheet(prediction_view.prediction_style_sheet)
        self.prob_left_label.setText(prediction_view.left_probability_text)
        self.prob_right_label.setText(prediction_view.right_probability_text)

    def _apply_game_stats(self):
        stats_view = GameRuntimePresenter.present_stats(self.predictions)
        self.total_predictions_label.setText(stats_view.total_predictions_text)
        self.left_predictions_label.setText(stats_view.left_predictions_text)
        self.right_predictions_label.setText(stats_view.right_predictions_text)
        self.transitions_label.setText(stats_view.transitions_text)
        self.confidence_label.setText(stats_view.confidence_text)
        
    def refresh_patients(self):
        """Atualiza a lista de pacientes"""
        self.patient_combo.clear()
        self.patient_combo.addItem("Selecionar paciente...")
        
        try:
            patients = self.patient_controller.list_patients()
            for patient in patients:
                self.patient_combo.addItem(
                    f"{patient['name']} (ID: {patient['id']})",
                    patient['id']
                )
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar pacientes: {e}")
    
    def toggle_connection(self):
        """Conecta/desconecta tudo de uma vez (EEG + UDP + ESP32)"""
        if not self.eeg_stream_controller.is_running():
            # === CONECTAR TUDO ===
            host = self.host_edit.text()
            port = self.port_spin.value()

            self.eeg_connection_phase = "connecting"
            self.unity_connection_phase = "connecting"
            self.orthosis_connection_phase = "connecting"
            self.connect_button_enabled = False
            self.record_button_enabled = False
            self._apply_connection_panel()
            
            # 1. EEG
            self.eeg_stream_controller.connect(host, port)

            # 2. UDP Unity
            try:
                if self.unity_controller.start_server():
                    self.udp_server_active = True
                    self.unity_connection_phase = "connected"
                else:
                    self.unity_connection_phase = "failed"
            except Exception:
                self.unity_connection_phase = "failed"

            # 3. ESP32
            try:
                if self.esp32_controller.connect():
                    self.esp32_connected = True
                    self.orthosis_connection_phase = "connected"
                else:
                    self.esp32_connected = False
                    self.orthosis_connection_phase = "standby"
            except Exception:
                self.esp32_connected = False
                self.orthosis_connection_phase = "failed"
            self._apply_connection_panel()

        else:
            # === DESCONECTAR TUDO ===
            self.disconnection_in_progress = True
            try:
                self.unity_controller.stop_server()
                self.udp_server_active = False
            except Exception:
                pass
            try:
                self.esp32_controller.disconnect()
                self.esp32_connected = False
            except Exception:
                pass
            self.eeg_connection_phase = "standby"
            self.unity_connection_phase = "standby"
            self.orthosis_connection_phase = "standby"
            self.connect_button_enabled = True
            self.record_button_enabled = False
            self._apply_connection_panel()
            self.eeg_stream_controller.disconnect()
        self._refresh_streaming_state()

    def toggle_eeg_connection(self):
        """Conecta ou desconecta apenas do EEG"""
        if not self.eeg_stream_controller.is_running():
            host = self.host_edit.text()
            port = self.port_spin.value()
            self.eeg_connection_phase = "connecting"
            self._apply_connection_panel()
            self.eeg_stream_controller.connect(host, port)
        else:
            self.disconnection_in_progress = True
            self.eeg_connection_phase = "standby"
            self._apply_connection_panel()
            self.eeg_stream_controller.disconnect()
        self._refresh_streaming_state()
    
    def manual_esp32_test(self, direction):
        """Teste manual do envio serial para ESP32"""
        if self.esp32_connected:
            success = self.esp32_controller.send_direction(direction)
            
            if success:
                side_text = "esquerda" if direction == 'esquerda' else "direita"
                QMessageBox.information(self, "Teste ESP32", f"Trigger enviado: Mão {side_text}")
            else:
                QMessageBox.critical(self, "Erro", "Falha ao enviar comando para ESP32!")
        else:
            QMessageBox.warning(self, "Aviso", "ESP32 não está conectado!")
    
    def send_esp32_signal(self, direction):
        """Envia sinal serial para ESP32 se conectado e o envio automático estiver habilitado"""
        if self.esp32_connected and self.esp32_auto_send_checkbox.isChecked():
            success = self.esp32_controller.send_direction(direction)
            
            if not success:
                print(f"Falha ao enviar sinal serial para ESP32: {direction}")
            return success
        return False

    def toggle_esp32_connection(self):
        """Conecta ou desconecta do ESP32"""
        try:
            if not self.esp32_connected:
                self.orthosis_connection_phase = "connecting"
                self._apply_connection_panel()
                
                # Tentar conectar
                connected = self.esp32_controller.connect()
                
                if connected:
                    self.esp32_connected = True
                    self.orthosis_connection_phase = "connected"
                    QMessageBox.information(self, "Sucesso", "ESP32 conectado com sucesso na COM4!")
                else:
                    self.orthosis_connection_phase = "failed"
                    QMessageBox.critical(self, "Erro", "Falha ao conectar ESP32.\nVerifique se o ESP32 está conectado na COM4.")
            else:
                # Desconectar
                self.esp32_controller.disconnect()
                self.esp32_connected = False
                self.orthosis_connection_phase = "standby"
                QMessageBox.information(self, "Sucesso", "ESP32 desconectado com sucesso!")
                
        except Exception as e:
            self.orthosis_connection_phase = "failed"
            QMessageBox.critical(self, "Erro", f"Erro ao conectar/desconectar ESP32: {e}")
        self._apply_connection_panel()
        self._refresh_streaming_state()

    def _on_esp32_connection(self, connected: bool):
        """Callback para mudanças de conexão ESP32"""
        self.esp32_connected = connected
        self.orthosis_connection_phase = "connected" if connected else "standby"
        if not connected and hasattr(self, 'esp32_status_label'):
            self.esp32_status_label.setText("ESP32: Desconectado")
            self.esp32_status_label.setStyleSheet(Theme.status_text("error") + " font-size: 12px;")
            self.esp32_toggle_btn.setText("Conectar ESP32")
            self.esp32_toggle_btn.setStyleSheet(Theme.btn_dev(True))
            self.esp32_test_left_btn.setEnabled(False)
            self.esp32_test_right_btn.setEnabled(False)
        self._apply_connection_panel()
        self._refresh_streaming_state()
    
    def manual_udp_test(self, direction):
        """Teste manual do envio UDP"""
        if self.udp_server_active:
            success = self.unity_controller.send_action(direction)
            if success:
                side_text = "esquerda" if direction == 'esquerda' else "direita"
                QMessageBox.information(self, "Teste UDP", f"Sinal enviado: Mão {side_text}")
            else:
                QMessageBox.critical(self, "Erro", "Falha ao enviar sinal UDP!")
        else:
            QMessageBox.warning(self, "Aviso", "Servidor UDP não está ativo!")
    
    def send_udp_signal(self, direction):
        """Envia sinal UDP se o servidor estiver ativo e o envio automático estiver habilitado"""
        if self.udp_server_active and self.udp_auto_send_checkbox.isChecked():
            success = self.unity_controller.send_action(direction)
            if not success:
                print(f"Falha ao enviar sinal UDP para {direction}")
            return success
        return False

    def toggle_udp_server(self):
        """Inicia ou para o servidor UDP manualmente (conectado ao botão)."""
        try:
            if not self.udp_server_active:
                self.unity_connection_phase = "connecting"
                self._apply_connection_panel()
                
                # Tentar iniciar servidor
                started = False
                try:
                    started = self.unity_controller.start_server()
                except Exception as e:
                    print(f"Erro ao iniciar servidor UDP: {e}")

                if started:
                    self.udp_server_active = True
                    self.unity_connection_phase = "connected"
                    QMessageBox.information(self, "Sucesso", "Servidor UDP iniciado com sucesso!\nBroadcast do IP enviado automaticamente.")
                else:
                    self.unity_connection_phase = "failed"
                    QMessageBox.critical(self, "Erro", "Falha ao iniciar servidor UDP")
            else:
                # Parar servidor
                try:
                    self.unity_controller.stop_server()
                except Exception:
                    pass
                self.udp_server_active = False
                self.unity_connection_phase = "standby"
                QMessageBox.information(self, "Sucesso", "Servidor UDP parado com sucesso!")
        except Exception as e:
            self.unity_connection_phase = "failed"
            QMessageBox.critical(self, "Erro", f"Erro ao alternar servidor UDP: {e}")
        self._apply_connection_panel()
        self._refresh_streaming_state()
    
    def toggle_recording(self):
        """Inicia/para a gravação"""
        if not self.is_recording:
            # Iniciar gravação
            if self.patient_combo.currentIndex() == 0:
                QMessageBox.warning(self, "Erro", "Selecione um paciente!")
                return
            
            selected_patient_id = int(self.patient_combo.currentData())
            patient_name = self.patient_combo.currentText().split(" (ID:")[0]
            
            # Obter tarefa do dropdown
            task = self.task_combo.currentText().lower().replace(" ", "_")  # ex: "Baseline" -> "baseline"
            
            # Verificar se é modo jogo
            if task == "jogo":
                if not self.inference_controller.has_loaded_model():
                    if not self.load_model():
                        return
                # Limpar variáveis do jogo
                self.predictions.clear()
                self._reset_ai_prediction_window()
                self._apply_prediction_display()
                self._apply_game_stats()
                
                # Resetar dados de acurácia
                self.reset_accuracy_data()
                
                # Resetar controle de resposta
                self.waiting_for_response = False
                
                # Resetar status visual da IA
                self._set_ai_status("waiting_task")
                
                # Resetar contadores de ações no início da gravação
                self.reset_action_counters()
                
                # Iniciar UDP receiver para acurácia - agora sempre disponível
                try:
                    self.start_accuracy_udp_receiver()
                except Exception as e:
                    print(f"Erro ao iniciar UDP receiver de acurácia: {e}")
                
                # Iniciar primeiro sinal aleatório imediatamente (não usar timer automático)
                # O próximo sinal será enviado apenas após receber CORRECT/WRONG
                QTimer.singleShot(1000, self.send_next_random_signal)  # Aguardar 1 segundo para inicializar
                
                # Manter timer como fallback caso não receba resposta (usar game_action_interval)
                self.game_action_timer.start(self.game_action_interval)
            
            try:
                # Usar logger OpenBCI se disponível
                if USE_OPENBCI_LOGGER:
                    self.csv_logger = OpenBCICSVLogger(
                        patient_id=f"P{selected_patient_id:03d}",
                        task=task,
                        patient_name=patient_name,  # Adicionar nome do paciente
                        base_path=os.path.dirname(get_recording_path(""))
                    )
                    filename = self.csv_logger.filename
                    # Mostrar caminho relativo para feedback visual
                    display_path = f"{self.csv_logger.patient_folder}/{filename}"

                
                self.is_recording = True
                self._apply_recording_ui()
                
                # Habilitar botões de marcadores
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                # self.baseline_btn.setEnabled(True)  # Botão removido
                
                # Resetar contadores
                self.reset_action_counters()
                
                # Registrar gravação via application/interface adapters
                recording_path = display_path if USE_OPENBCI_LOGGER else filename
                self.current_recording_id = self.recording_controller.start_recording(
                    StartRecordingRequest(
                        patient_id=selected_patient_id,
                        filename=recording_path,
                        task_type=task,
                    )
                )
                started_session = self.session_controller.start_session(
                    StartSessionRequest(
                        patient_id=selected_patient_id,
                        task_type=task,
                        recording_id=self.current_recording_id,
                        started_at_epoch=time.time(),
                    )
                )
                self._refresh_streaming_state(session=started_session)
                
                # =====================================================================
                # ENVIAR TRIGGER PARA ATIVAR A TAREFA NO VR
                # =====================================================================
                try:
                    if self.unity_controller.is_server_active() and self.unity_controller.is_client_connected():
                        time.sleep(0.5)  # Pequeno delay para garantir que tudo está pronto
                        self.unity_controller.send_trigger()
                        print(f"[GRAVAÇÃO] send_trigger() enviado para VR", flush=True)
                except Exception as e:
                    print(f"[GRAVAÇÃO] Erro ao enviar send_trigger(): {e}", flush=True)
                # =====================================================================
                
                # Iniciar timer de sessão
                self.session_timer.start(1000)  # Atualizar a cada segundo
                
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao iniciar gravação: {e}")
        else:
            # Parar gravação
            # Parar logging, mas manter referência para obter o caminho do arquivo
            logger = None
            if self.csv_logger:
                logger = self.csv_logger
                try:
                    logger.stop_logging()
                except Exception:
                    pass
            # Limpar a referência de longo prazo (UI não mais grava)
            self.csv_logger = None
            
            self.is_recording = False
            
            # =====================================================================
            # ENVIAR END_TASK PARA O VR
            # =====================================================================
            current_task = self.task_combo.currentText()
            try:
                if self.unity_controller.is_server_active() and self.unity_controller.is_client_connected():
                    # Enviar end_task
                    self.unity_controller.end_task()
                    print(f"[GRAVAÇÃO] end_task() enviado para VR", flush=True)
                    
                    # Se for jogo, também enviar end_session com mensagem motivacional
                    if current_task == "Jogo":
                        time.sleep(0.3)  # Pequeno delay
                        self.unity_controller.end_session("Parabéns! Sessão finalizada com sucesso!")
                        print(f"[GRAVAÇÃO] end_session() com mensagem enviada para VR", flush=True)
            except Exception as e:
                print(f"[GRAVAÇÃO] Erro ao enviar end_task/end_session: {e}", flush=True)
            
            # =====================================================================
            
            # Parar UDP receiver de acurácia
            self.stop_accuracy_udp_receiver()
            
            # Parar timer de ações automáticas no jogo
            if self.game_action_timer.isActive():
                self.game_action_timer.stop()
            
            # Resetar controle de resposta
            self.waiting_for_response = False
            
            # Resetar controle de IA
            self._reset_ai_prediction_window()
            
            # Resetar status visual da IA
            self._set_ai_status("stopped")
            
            # Resetar contadores de ações
            self.reset_action_counters()
                
            self._apply_recording_ui()
            
            # Desabilitar botões de marcadores
            self.t1_btn.setEnabled(False)
            self.t2_btn.setEnabled(False)
            # self.baseline_btn.setEnabled(False)  # Botão removido
            
            # Parar timer de baseline se estiver rodando
            if self.baseline_timer.isActive():
                self.baseline_timer.stop()
                self.baseline_label.setText("")
            self.marker_controller.reset_state()
            self._update_marker_labels(self.marker_controller.get_state())
            
            # Parar timer de sessão
            self.session_timer.stop()
            self.session_elapsed_seconds = 0
            self.update_session_timer()

            current_session = self._get_current_session()
            if self.current_recording_id is not None:
                duration_seconds = 0
                if current_session is not None:
                    duration_seconds = max(
                        0,
                        int(time.time() - float(current_session.started_at_epoch)),
                    )
                self.recording_controller.stop_recording(self.current_recording_id, duration_seconds)
                self.current_recording_id = None
            ended_session = self.session_controller.end_session()
            self._refresh_streaming_state()
            
            # Verificar se é tarefa de treino para mostrar popup de treinamento
            print(f"[DEBUG] stop_recording: current_task={current_task}, logger_present={logger is not None}")
            if current_task == "Treino":
                # Obter informações para o treino
                patient_name = self.patient_combo.currentText().split(" (ID:")[0]
                csv_file_path = None
                
                # Obter caminho do arquivo CSV gravado a partir da referência local 'logger'
                if USE_OPENBCI_LOGGER and hasattr(logger, 'get_full_path'):
                    try:
                        csv_file_path = logger.get_full_path()
                    except Exception:
                        csv_file_path = None
                elif logger is not None and hasattr(logger, 'filename'):
                    # Construir caminho completo para logger simples
                    csv_file_path = str(get_recording_path(logger.filename))
                
                print(f"[DEBUG] stop_recording: csv_file_path={csv_file_path}")
                if csv_file_path and os.path.exists(csv_file_path):
                    # Iniciar fluxo automático de treino sem pedir confirmação
                    # show_training_dialog agora suporta auto_start=True
                    try:
                        print("[DEBUG] stop_recording: launching auto training dialog")
                        patient_id = self.patient_combo.currentData()
                        if ended_session is not None:
                            patient_id = ended_session.patient_id
                        self.show_training_dialog(csv_file_path, patient_id, patient_name, auto_start=True)
                    except Exception as e:
                        print(f"[DEBUG] stop_recording: failed to start training dialog: {e}")
                        QMessageBox.information(self, "Sucesso", "Gravação de treino finalizada!")
                else:
                    QMessageBox.information(self, "Sucesso", "Gravação de treino finalizada!")
            else:
                QMessageBox.information(self, "Sucesso", "Gravação finalizada!")
    

    def game_random_action(self):
        """Executa uma ação aleatória no jogo (fallback caso não receba resposta)"""
        if self.is_recording and self.csv_logger:
            # Verificar se não está aguardando resposta
            if self.waiting_for_response:
                print("⚠️  Timeout: Não recebeu resposta CORRECT/WRONG, enviando sinal de fallback")
                # Resetar estado e enviar novo sinal
                self.waiting_for_response = False
                
            import random
            actions = ['T1', 'T2'] #T1 para movimento esquerda, T2 para movimento direita
            action = random.choice(actions)
            
            # Marcar que está aguardando resposta
            self.waiting_for_response = True

            self.add_marker(action)
            self._record_pipeline_event(
                "TASK_SENT",
                marker=action,
                source="fallback",
                window_size=self.window_size,
            )
            self._start_ai_prediction_window("active_fallback", source="fallback")

    def send_next_random_signal(self):
        """Envia o próximo sinal aleatório após receber resposta"""
        if self.is_recording and self.csv_logger:
            print("🎲 Enviando próximo sinal aleatório")
            import random
            actions = ['T1', 'T2'] #T1 para movimento esquerda, T2 para movimento direita
            action = random.choice(actions)
            
            # Marcar que está aguardando resposta
            self.waiting_for_response = True

            self.add_marker(action)
            self._record_pipeline_event(
                "TASK_SENT",
                marker=action,
                source="random",
                window_size=self.window_size,
            )
            self._start_ai_prediction_window("active_window")

    def _start_ai_prediction_window(self, status: str, source: str = ""):
        """
        Abre a janela de inferencia zerando qualquer amostra anterior ao sinal.

        A predicao so pode acontecer depois que window_size amostras novas
        chegarem a partir do marcador enviado para Unity em add_marker().
        """
        self.game_inference.start_window(started_at_ms=time.time() * 1000)
        self._sync_ai_prediction_state()
        self._record_pipeline_event(
            "AI_WINDOW_OPENED",
            duration_ms=self.ai_window_duration,
            window_size=self.window_size,
            source=source or "random",
        )

        suffix = f" ({source})" if source else ""
        print(f"🤖 Janela de IA aberta por {self.ai_window_duration/1000}s{suffix}")

        self._set_ai_status(status)
        QTimer.singleShot(self.ai_window_duration, self.close_ai_window)

    def close_ai_window(self):
        """Fecha a janela de IA após o tempo configurado."""
        self.game_inference.close_window()
        self._sync_ai_prediction_state()
        self._record_pipeline_event(
            "AI_WINDOW_CLOSED",
            samples_collected=self.samples_since_last_prediction,
        )
        print("🚫 Janela de IA fechada automaticamente")
        
        # Atualizar status visual
        self._set_ai_status("inactive")

    def add_marker(self, marker_type):
        """Adiciona um marcador durante a gravação"""
        if self.is_recording and self.csv_logger:
            current_task = self.task_combo.currentText()
            result = self.marker_controller.register_marker(marker_type, current_task)
            if not result.accepted:
                if result.reason == "baseline_active":
                    QMessageBox.warning(
                        self,
                        "Baseline Ativo",
                        "Não é possível adicionar marcadores durante o baseline",
                    )
                return

            state = result.state
            self._update_marker_labels(state)

            if result.external_signal and self.udp_server_active:
                self.unity_controller.send_action(result.external_signal)
            if result.esp32_direction:
                self.send_esp32_signal(result.esp32_direction)

            if USE_OPENBCI_LOGGER:
                # Marcar para adicionar na próxima amostra
                self.pending_marker = marker_type
            else:
                # Logger simples
                marker = self.csv_logger.add_marker(marker_type)
            
            self._apply_recording_ui(hint=marker_type)
            QTimer.singleShot(2000, lambda: self._apply_recording_ui())
    
    def start_baseline(self):
        """Inicia o período de baseline"""
        if self.is_recording and self.csv_logger:
            baseline_state = self.marker_controller.start_baseline(300)
            if USE_OPENBCI_LOGGER:
                # Logger OpenBCI
                if hasattr(self.csv_logger, 'start_baseline'):
                    self.csv_logger.start_baseline()
                else:
                    # Fallback
                    self.csv_logger.add_marker("BASELINE")
            else:
                # Logger simples
                self.csv_logger.add_marker("BASELINE")

            # Iniciar timer visual
            if not self.baseline_timer.isActive():
                self.baseline_timer.start(1000)
            self._update_marker_labels(baseline_state)
             
            # Desabilitar outros botões por 5 minutos
            self.t1_btn.setEnabled(False)
            self.t2_btn.setEnabled(False) 
            # self.baseline_btn.setEnabled(False)  # Botão removido
            
            self._apply_recording_ui(hint="Baseline")
    
    def update_baseline_timer(self):
        """Atualiza o timer de baseline"""
        result = self.marker_controller.tick_baseline()
        state = result.state
        remaining = state.baseline_remaining_seconds
        self._update_marker_labels(state)

        if remaining > 0:
            minutes = remaining // 60
            seconds = remaining % 60
            self.baseline_label.setText(f"Baseline: {minutes:02d}:{seconds:02d}")
            self._apply_recording_ui(hint=f"{minutes:02d}:{seconds:02d}")
        else:
            # Baseline terminado
            self.baseline_timer.stop()
            self.baseline_label.setText("")
            
            # Reabilitar botões se ainda estiver gravando
            if self.is_recording:
                self.t1_btn.setEnabled(True)
                self.t2_btn.setEnabled(True)
                # self.baseline_btn.setEnabled(True)  # Botão removido
                self._apply_recording_ui()
                QMessageBox.information(self, "Baseline", "Período de baseline finalizado!")

    def reset_recording_label(self):
        """Mantém compatibilidade com timers antigos."""
        self._apply_recording_ui()
    
    def load_model(self):
        """Carrega modelo CNN para inferência"""
        try:
            model = self.inference_controller.load_latest_model()
            self._update_model_status(model)
            print(f"Carregando modelo TensorFlow encontrado: {model.path}")
            return True
        except ValueError as exc:
            self.model_status_label.setText(f"Erro: {exc}")
            QMessageBox.warning(
                self,
                "Erro",
                f"Modelo não encontrado! {exc}",
            )
            return False
        except Exception as e:
            self.model_status_label.setText(f"Erro ao carregar modelo: {e}")
            QMessageBox.critical(self, "Erro", f"Erro ao carregar modelo: {e}")
            return False

    def load_model_from_path(self, model_path: str) -> bool:
        """Tenta carregar um modelo explicitamente a partir de um caminho.

        Retorna True se carregado com sucesso, False caso contrário.
        """
        try:
            model = self.inference_controller.load_model(model_path)
            self._update_model_status(model)
            self._refresh_streaming_state(loaded_model=model)
            print(f"Modelo TensorFlow carregado: {model_path}")
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.model_status_label.setText(f"Erro ao carregar modelo: {e}")
            return False

    def find_tf_models(self) -> List[str]:
        """Procura por arquivos .keras/.h5 em locais comuns e retorna caminhos absolutos ordenados por data (mais recente primeiro)."""
        try:
            return [model.path for model in self.inference_controller.list_models()]
        except Exception as e:
            print(f"Erro ao listar modelos TensorFlow: {e}")
            return []

    def _update_model_status(self, model: ModelViewModel):
        self._refresh_streaming_state(loaded_model=model)
        self.model_status_label.setText(f"Modelo carregado: {model.name}")
        expected_time_steps = model.expected_time_steps
        if expected_time_steps is not None and expected_time_steps != self.window_size:
            print(
                f"Aviso: modelo espera {expected_time_steps} timesteps, "
                f"runtime window_size={self.window_size}. Adaptacao sera aplicada na inferencia."
            )
            
    def update_game_stats(self):
        """Atualiza estatísticas do jogo"""
        if not self._is_game_mode():
            return
        self._apply_game_stats()
        
    def process_accuracy_message(self, message):
        """Processa mensagem UDP recebida para cálculo de acurácia"""
        print(f"🔍 DEBUG: Mensagem recebida para acurácia: '{message}'")
        
        if not self._is_game_mode():
            print("🔍 DEBUG: Ignorando mensagem - não está em modo jogo")
            return
            
        try:
            trial = AccuracyPresenter.parse_message(message)
            if trial is None:
                print(f"Mensagem de acurácia ignorada: {message}")
                return

            self.accuracy_trials.append(trial)
            self.update_accuracy_display()

            status = "✓" if trial.is_correct else "✗"
            print(
                f"Acurácia: {trial.expected_action} vs {trial.real_action} {status}"
            )
        except Exception as e:
            print(f"Erro ao processar mensagem de acurácia: {e}")
            
    def update_accuracy_display(self):
        """Atualiza a interface de acurácia"""
        self._apply_accuracy_display()
        
    def reset_accuracy_data(self):
        """Reseta os dados de acurácia"""
        self.accuracy_trials.clear()
        self._apply_accuracy_display()
    
    def reset_action_counters(self):
        """Reseta os contadores de ações T1 e T2"""
        state = self.marker_controller.reset_state()
        self._update_marker_labels(state)
        print("🔄 Contadores de ações resetados")
        
    def start_accuracy_udp_receiver(self):
        """
        Inicia o receptor de acurácia.
        Agora usa o sistema de callbacks do UnityCommunicator.
        """
        print("✅ Sistema de acurácia ativo - usando callbacks do UnityCommunicator")
        # O receptor de mensagens já está ativo através dos callbacks do unity_communicator
        # As mensagens serão processadas automaticamente via _on_unity_message()
        
    def stop_accuracy_udp_receiver(self):
        """Para o UDP receiver de acurácia"""
        print("Sistema de acurácia parado - callbacks mantidos ativos")
        
    def predict_movement(self, eeg_data):
        """Faz predição do movimento com o modelo CNN"""
        if not self._is_game_mode() or not self.inference_controller.has_loaded_model():
            return
        if not self.game_inference.is_window_open or self.game_inference.prediction_locked:
            return
            
        try:
            inference_start = time.perf_counter()
            prediction = self.inference_controller.predict(eeg_data)
            inference_latency_ms = (time.perf_counter() - inference_start) * 1000
            pred = int(prediction.predicted_index)
            runtime_action = UnityCommandMapper.from_prediction(pred)

            # Atualizar interface
            timestamp = datetime.now()
            self._apply_prediction_display(prediction)
            unity_success = self.send_udp_signal(runtime_action.direction)
            self.send_esp32_signal(runtime_action.direction)

            self.game_inference.mark_prediction_used()
            self._sync_ai_prediction_state()
            self._record_pipeline_event(
                "PREDICTION_DONE",
                predicted_index=pred,
                confidence=float(prediction.confidence),
                latency_ms=round(inference_latency_ms, 2),
                samples=self.window_size,
            )
            self._record_pipeline_event(
                "UNITY_COMMAND_SENT",
                direction=runtime_action.direction,
                success=bool(unity_success),
            )
            
            # Salvar predição
            self.predictions.append((timestamp, pred, float(prediction.confidence)))
            
        except Exception as e:
            print(f"Erro na predição: {e}")
    
    def on_data_received(self, data):
        """Callback para dados recebidos"""
        # Confirmar conexão do EEG no primeiro dado recebido
        if self.eeg_connection_phase == "connecting":
            if self.eeg_stream_controller.is_mock_mode():
                self.eeg_connection_phase = "mock"
            else:
                self.eeg_connection_phase = "connected"
            
            self.record_button_enabled = True
            self._apply_connection_panel()
            self._refresh_streaming_state()
            print(f"[EEG] Conexão confirmada: Primeiro dado recebido ({len(data)} canais)")

        # Enviar para plot
        self.plot_widget.add_data(data)
        current_time_seconds = time.time()
        current_time_ms = current_time_seconds * 1000
        try:
            self.pipeline_telemetry.observe_eeg_sample(
                now_seconds=current_time_seconds,
            )
        except Exception as exc:
            print(f"[PIPELINE] Falha ao medir taxa EEG: {exc}")
        
        # Adicionar ao buffer de dados e verificar predição
        if self._is_game_mode():
            sample_result = self.game_inference.add_sample(
                data,
                now_ms=current_time_ms,
            )
            self._sync_ai_prediction_state()

            if sample_result.status == GameInferenceCoordinator.STATUS_EXPIRED:
                self._record_pipeline_event(
                    "AI_WINDOW_EXPIRED",
                    samples_collected=sample_result.samples_collected,
                    duration_ms=self.ai_window_duration,
                )
                print(f"🚫 Janela de IA fechada após {self.ai_window_duration/1000}s")
                self.close_ai_window()
            elif sample_result.status == GameInferenceCoordinator.STATUS_READY:
                elapsed_ms = None
                if self.task_start_time is not None:
                    elapsed_ms = round(current_time_ms - self.task_start_time, 2)
                self._record_pipeline_event(
                    "SAMPLE_250_READY",
                    samples=self.window_size,
                    elapsed_ms=elapsed_ms,
                    eeg_rate_hz=round(
                        self.pipeline_telemetry.sample_rate.latest_rate_hz,
                        2,
                    ),
                )
                quality_result = self.eeg_quality_validator.validate(
                    sample_result.window
                )
                if quality_result.accepted:
                    self.predict_movement(np.array(sample_result.window))
                else:
                    self._record_pipeline_event(
                        "WINDOW_REJECTED",
                        reason=quality_result.reason,
                        samples=self.window_size,
                    )
                    print(
                        f"Janela EEG rejeitada antes da IA: {quality_result.reason}"
                    )
                    self.game_inference.mark_prediction_used()
                    self._sync_ai_prediction_state()
                
        # Enviar para logger se estiver gravando
        if self.is_recording and self.csv_logger:
            if USE_OPENBCI_LOGGER and hasattr(self.csv_logger, 'log_sample'):
                # Logger OpenBCI - verificar marcador pendente
                marker = self.pending_marker
                self.pending_marker = None  # Limpar marcador pendente
                
                # Garantir que temos 'channels' canais
                if len(data) == self.channels:
                    self.csv_logger.log_sample(data, marker)
                else:
                    # Ajustar dados se necessário
                    if len(data) >= self.channels:
                        eeg_data = data[:self.channels]
                    else:
                        eeg_data = data + [0.0] * (self.channels - len(data))
                    self.csv_logger.log_sample(eeg_data, marker)
            else:
                # Logger simples (fallback)
                self.csv_logger.log_data(data)
    
    def on_connection_status(self, connected):
        """Callback para status da conexão - agora aguarda o primeiro dado"""
        if connected:
            # Se for mock, podemos considerar conectado imediatamente se quiser,
            # mas vamos manter a lógica de esperar dado para ambos para consistência
            # ou apenas setar como 'connecting' para o real.
            if self.eeg_stream_controller.is_mock_mode():
                # No modo mock, como os dados são gerados localmente, 
                # podemos considerar 'mock' já ou esperar o primeiro dado.
                # Vamos esperar o dado para garantir que o loop está rodando.
                if self.eeg_connection_phase != "mock":
                    self.eeg_connection_phase = "connecting"
            else:
                self.eeg_connection_phase = "connecting"
            
            # Não habilitamos record_btn aqui, esperamos on_data_received
        else:
            self.eeg_connection_phase = (
                "standby" if self.disconnection_in_progress else "failed"
            )
            self.record_button_enabled = False
            self.disconnection_in_progress = False

        self.connect_button_enabled = True
        self._apply_connection_panel()
        self._refresh_streaming_state()

    def stop_streaming(self):
        """Stops the EEG stream if it is running."""
        self.eeg_stream_controller.disconnect()
    
    def update_session_timer(self):
        """Atualiza o display do timer de sessão"""
        started_at = self._get_session_started_at()
        if started_at is not None:
            # Calcular tempo decorrido
            elapsed = int(time.time() - float(started_at))
        else:
            elapsed = 0
        
        # Formatar tempo como HH:MM:SS
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        if self.is_recording:
            self.session_timer_label.setText(f"Tempo: {time_str}")
            self.session_timer_label.setStyleSheet(Theme.status_text("connected") + " font-size: 20px;")
        else:
            self.session_timer_label.setText(f"Tempo: {time_str}")
            self.session_timer_label.setStyleSheet(Theme.status_text("off") + " font-size: 20px;")
    
    def on_task_changed(self):
        """Callback chamado quando a tarefa é alterada"""
        task = self.task_combo.currentText()

        # Resetar contadores de ações sempre que mudar de tarefa
        self.reset_action_counters()
        self._apply_task_view_state()

        if not self.is_recording:
            if task == "Jogo":
                if not self.inference_controller.has_loaded_model():
                    try:
                        candidates = self.find_tf_models()
                    except Exception:
                        candidates = []

                    if candidates:
                        items = [os.path.basename(p) for p in candidates]
                        item, ok = QInputDialog.getItem(self, "Selecionar Modelo", "Modelos TensorFlow encontrados:", items, 0, False)
                        if ok and item:
                            sel_index = items.index(item)
                            sel_path = candidates[sel_index]
                            loaded = self.load_model_from_path(sel_path)
                            if loaded:
                                QMessageBox.information(self, "Modelo carregado", f"Modelo carregado: {sel_path}")
                            else:
                                QMessageBox.warning(self, "Falha ao carregar", f"Falha ao carregar o modelo selecionado: {sel_path}")
                        else:
                            QMessageBox.information(self, "Nenhum modelo selecionado", "Nenhum modelo foi selecionado. Você pode treinar um modelo ou colocar um arquivo .keras em bci/models.")
                    else:
                        QMessageBox.warning(
                            self,
                            "Modelo não encontrado",
                            "Nenhum modelo TensorFlow (.keras/.h5) foi encontrado nos diretórios configurados.\n"
                            "Coloque um arquivo .keras em um diretório de modelos conhecido ou treine um modelo pela interface."
                        )
    
    def update_record_button_text(self):
        """Atualiza o texto do botão de gravação baseado no estado e tarefa"""
        self._apply_recording_ui()
    
    def _on_unity_message(self, message: str):
        """Callback para mensagens recebidas do Unity"""
        print(f"[Unity] Mensagem recebida: {message}")
        
        # Verificar se recebeu resposta CORRECT ou WRONG
        if "CORRECT" in message or "WRONG" in message:
            print(f"✅ Resposta recebida: {message}")
            self._record_pipeline_event(
                "UNITY_RESPONSE",
                message=message,
                waiting_for_response=bool(self.waiting_for_response),
            )
            if self.waiting_for_response:
                self.waiting_for_response = False
                print("🔓 Liberado para enviar próximo sinal aleatório")
                # Aguardar 7 segundos antes do próximo sinal
                QTimer.singleShot(7000, self.send_next_random_signal)
        
        # Processar mensagens específicas do Unity
        if "FLOWER" in message:
            # Usar o signal existente para processar mensagens de acurácia
            self.accuracy_message_signal.emit(message)
        elif "CONNECTED" in message:
            print("[Unity] Confirmação de conexão recebida")
        elif "STATUS" in message:
            print(f"[Unity] Status: {message}")
    
    def _on_unity_connection(self, connected: bool):
        """Callback para mudanças no status de conexão com Unity"""
        if connected:
            self.unity_connection_phase = "connected"
            print("[Unity] TCP conectado")
        else:
            if not self.udp_server_active:
                self.unity_connection_phase = "standby"
            print("[Unity] TCP desconectado")
        self._apply_connection_panel()
        self._refresh_streaming_state()
    
    def show_training_dialog(self, csv_file_path, patient_id, patient_name, auto_start: bool = False):
        """Mostra o diálogo de confirmação e execução do treino"""
        try:
            dialog = TrainingDialog(
                self.training_controller,
                csv_file_path,
                int(patient_id),
                patient_name,
                auto_load_model=auto_start,
                parent=self,
            )
            dialog.training_progress_signal.connect(self._on_training_progress)
            dialog.training_finished_signal.connect(self._on_training_finished)
            dialog.model_ready_signal.connect(self._on_trained_model_ready)

            if auto_start:
                self._apply_recording_ui(hint="Treino")
                dialog.start_training()
                dialog.show()
                try:
                    dialog.raise_()
                    dialog.activateWindow()
                except Exception:
                    pass
                return

            result = dialog.exec_()
            if result == QDialog.Accepted:
                print(f"Iniciando treino para paciente {patient_name} com arquivo {csv_file_path}")
            else:
                QMessageBox.information(self, "Sucesso", "Gravação de treino finalizada!")
                
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao abrir diálogo de treino: {e}")
            QMessageBox.information(self, "Sucesso", "Gravação de treino finalizada!")

    def _on_training_progress(self, message: str):
        if hasattr(self, "gravacao_status"):
            short = (message[:24] + "…") if len(message) > 24 else message
            self.gravacao_status.setText(f"Treino · {short}")

    def _on_training_finished(self, success: bool, _message: str):
        self._refresh_streaming_state()
        self._apply_recording_ui()

    def _on_trained_model_ready(self, model_path: str):
        loaded_model = self.inference_controller.get_loaded_model()
        if loaded_model is not None:
            self._update_model_status(loaded_model)
        else:
            self._refresh_streaming_state()
            self.model_status_label.setText(
                f"Modelo treinado pronto: {os.path.basename(model_path)}"
            )
