"""
Diálogo de treinamento para integração com o StreamingWidget
"""

from PyQt5.QtWidgets import (
	QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
	QTextEdit, QProgressBar, QMessageBox
)
from PyQt5.QtCore import pyqtSignal

from gui.training.model_trainer import ModelTrainerThread


class TrainingDialog(QDialog):
	def __init__(self, csv_file_path: str, patient_id: int, patient_name: str, parent=None):
		super().__init__(parent)
		self.csv_file_path = csv_file_path
		self.patient_id = patient_id
		self.patient_name = patient_name
		self.trainer_thread = None

		self.setWindowTitle("Treinar Modelo EEG")
		self.resize(500, 420)
		self._setup_ui()

	def _setup_ui(self):
		layout = QVBoxLayout()

		title = QLabel("Treinar Modelo com Gravação")
		title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
		layout.addWidget(title)

		info_label = QLabel(f"Paciente: {self.patient_name} (ID: {self.patient_id})")
		info_label.setStyleSheet("font-size: 12px; margin-bottom: 5px;")
		layout.addWidget(info_label)

		file_label = QLabel(f"Arquivo: {self.csv_file_path}")
		file_label.setWordWrap(True)
		file_label.setStyleSheet("font-size: 10px; color: gray; margin-bottom: 12px;")
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
		self.log_text.setStyleSheet("background-color: #f8f8f8; font-family: Consolas, monospace; font-size: 10px;")
		layout.addWidget(self.log_text)

		btns = QHBoxLayout()
		self.cancel_btn = QPushButton("Cancelar")
		self.cancel_btn.clicked.connect(self.reject)
		self.train_btn = QPushButton("Treinar Modelo")
		self.train_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
		self.train_btn.clicked.connect(self._start_training)
		btns.addWidget(self.cancel_btn)
		btns.addWidget(self.train_btn)
		layout.addLayout(btns)

		self.setLayout(layout)

	def _start_training(self):
		self.train_btn.setEnabled(False)
		self.cancel_btn.setText("Fechar")
		self.progress_label.setVisible(True)
		self.progress_bar.setVisible(True)
		self.log_text.setVisible(True)

		self.trainer_thread = ModelTrainerThread(self.csv_file_path, self.patient_id)
		self.trainer_thread.progress_signal.connect(self._on_progress)
		self.trainer_thread.finished_signal.connect(self._on_finished)
		if hasattr(self.trainer_thread, 'model_path_signal'):
			self.trainer_thread.model_path_signal.connect(self._on_model_ready)
		self.trainer_thread.start()

	def _on_progress(self, message: str):
		self.progress_label.setText(message)
		self.log_text.append(message)
		sb = self.log_text.verticalScrollBar()
		sb.setValue(sb.maximum())

	def _on_finished(self, success: bool, message: str):
		self.progress_bar.setRange(0, 1)
		self.progress_bar.setValue(1)
		if success:
			self.progress_label.setText("Concluído com sucesso")
			self.progress_label.setStyleSheet("color: green; font-weight: bold;")
			QMessageBox.information(self, "Sucesso", message)
			try:
				self.accept()
			except Exception:
				pass
		else:
			self.progress_label.setText("Erro durante o treinamento")
			self.progress_label.setStyleSheet("color: red; font-weight: bold;")
			QMessageBox.critical(self, "Erro", message)
		self.log_text.append(message)
		self.cancel_btn.setText("Fechar")
		self.cancel_btn.setEnabled(True)

	def _on_model_ready(self, model_path: str):
		self.log_text.append(f"Modelo pronto: {model_path}")
		try:
			self.accept()
		except Exception:
			pass

