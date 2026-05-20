"""
Diálogo de treinamento para integração com o StreamingWidget
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QProgressBar, QMessageBox
)
from PyQt5.QtCore import pyqtSignal

from brainbridge_v2.interface_adapters.controllers.training_controller import (
    TrainingController,
)
from brainbridge_v2.presentation.gui.training.model_trainer import ModelTrainerThread
from brainbridge_v2.presentation.gui.styles import Theme


class TrainingDialog(QDialog):
    training_progress_signal = pyqtSignal(str)
    training_finished_signal = pyqtSignal(bool, str)
    model_ready_signal = pyqtSignal(str)

    def __init__(
        self,
        training_controller: TrainingController,
        csv_file_path: str,
        patient_id: int,
        patient_name: str,
        auto_load_model: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.training_controller = training_controller
        self.csv_file_path = csv_file_path
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.auto_load_model = auto_load_model
        self.trainer_thread = None

        self.setWindowTitle("Treinar Modelo EEG")
        self.resize(500, 420)
        self.setStyleSheet(Theme.get_stylesheet())
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("Treinar Modelo com Gravação")
        title.setStyleSheet(Theme.section_title("16px"))
        layout.addWidget(title)

        info_label = QLabel(f"Paciente: {self.patient_name} (ID: {self.patient_id})")
        info_label.setStyleSheet(f"font-size: 12px; color: {Theme.WHITE};")
        layout.addWidget(info_label)

        file_label = QLabel(f"Arquivo: {self.csv_file_path}")
        file_label.setWordWrap(True)
        file_label.setStyleSheet(f"font-size: 10px; color: {Theme.GRAY};")
        layout.addWidget(file_label)

        self.progress_label = QLabel("Preparando...")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setVisible(False)
        self.log_text.setMaximumHeight(160)
        self.log_text.setStyleSheet(Theme.log_console())
        layout.addWidget(self.log_text)

        btns = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.clicked.connect(self.reject)
        self.train_btn = QPushButton("Treinar Modelo")
        self.train_btn.setStyleSheet(Theme.btn_green("8px 20px", "13px", "700"))
        self.train_btn.clicked.connect(self.start_training)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.train_btn)
        layout.addLayout(btns)

        self.setLayout(layout)

    def start_training(self):
        self.train_btn.setEnabled(False)
        self.cancel_btn.setText("Fechar")
        self.progress_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.log_text.setVisible(True)

        self.trainer_thread = ModelTrainerThread(
            self.training_controller,
            self.csv_file_path,
            self.patient_id,
            auto_load_model=self.auto_load_model,
            parent=self,
        )
        self.trainer_thread.progress_signal.connect(self._on_progress)
        self.trainer_thread.finished_signal.connect(self._on_finished)
        self.trainer_thread.model_path_signal.connect(self._on_model_ready)
        self.trainer_thread.start()

    def _on_progress(self, message: str):
        self.progress_label.setText(message)
        self.log_text.append(message)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.training_progress_signal.emit(message)

    def _on_finished(self, success: bool, message: str):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        if success:
            self.progress_label.setText("Concluído com sucesso")
            self.progress_label.setStyleSheet(Theme.status_text("connected") + " font-size: 13px;")
            QMessageBox.information(self, "Sucesso", message)
            try:
                self.accept()
            except Exception:
                pass
        else:
            self.progress_label.setText("Erro durante o treinamento")
            self.progress_label.setStyleSheet(Theme.status_text("error") + " font-size: 13px;")
            QMessageBox.critical(self, "Erro", message)

        self.log_text.append(message)
        self.cancel_btn.setText("Fechar")
        self.cancel_btn.setEnabled(True)
        self.training_finished_signal.emit(success, message)

    def _on_model_ready(self, model_path: str):
        self.log_text.append(f"Modelo pronto: {model_path}")
        self.model_ready_signal.emit(model_path)

