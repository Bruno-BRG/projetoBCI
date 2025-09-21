"""
Diálogo para confirmação e progresso do treino do modelo
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QTextEdit, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from ..training.model_trainer import ModelTrainerThread


class TrainingDialog(QDialog):
    """Diálogo para confirmar e acompanhar o treino do modelo"""
    
    def __init__(self, csv_file_path, patient_id, patient_name, parent=None):
        super().__init__(parent)
        self.csv_file_path = csv_file_path
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.trainer_thread = None
        
        self.setWindowTitle("Treinar Modelo EEG")
        self.setModal(True)
        self.resize(500, 400)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Configura a interface do diálogo"""
        layout = QVBoxLayout()
        
        # Título
        title = QLabel("Treinar Modelo com Gravação")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Informações
        info_layout = QVBoxLayout()
        
        info_label = QLabel(f"Paciente: {self.patient_name} (ID: {self.patient_id})")
        info_label.setStyleSheet("font-size: 12px; margin-bottom: 5px;")
        info_layout.addWidget(info_label)
        
        file_label = QLabel(f"Arquivo: {self.csv_file_path}")
        file_label.setWordWrap(True)
        file_label.setStyleSheet("font-size: 10px; color: gray; margin-bottom: 15px;")
        info_layout.addWidget(file_label)
        
        layout.addLayout(info_layout)
        
        # Descrição
        description = QLabel(
            "Esta operação irá treinar um novo modelo de IA usando os dados da gravação recém finalizada.\\n\\n"
            "O processo pode levar alguns minutos e criará um modelo personalizado para este paciente.\\n\\n"
            "Deseja prosseguir com o treinamento?"
        )
        description.setWordWrap(True)
        description.setStyleSheet("margin-bottom: 20px;")
        layout.addWidget(description)
        
        # Área de progresso (inicialmente oculta)
        self.progress_widget = QVBoxLayout()
        
        self.progress_label = QLabel("Preparando treinamento...")
        self.progress_label.setVisible(False)
        self.progress_widget.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Progresso indeterminado
        self.progress_widget.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setVisible(False)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("background-color: #f0f0f0; font-family: monospace; font-size: 9px;")
        self.progress_widget.addWidget(self.log_text)
        
        layout.addLayout(self.progress_widget)
        
        # Botões
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.train_btn = QPushButton("Treinar Modelo")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.train_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def start_training(self):
        """Inicia o treinamento do modelo"""
        # Muda para modo de progresso
        self.train_btn.setEnabled(False)
        self.cancel_btn.setText("Fechar")
        
        # Mostra elementos de progresso
        self.progress_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.log_text.setVisible(True)
        
        # Ajusta tamanho da janela
        self.resize(500, 500)
        
        # Cria e inicia thread de treinamento
        self.trainer_thread = ModelTrainerThread(self.csv_file_path, self.patient_id)
        self.trainer_thread.progress_signal.connect(self.update_progress)
        self.trainer_thread.finished_signal.connect(self.training_finished)
        # Conectar sinal que informa o caminho do modelo gerado
        if hasattr(self.trainer_thread, 'model_path_signal'):
            self.trainer_thread.model_path_signal.connect(self.on_model_ready)
        self.trainer_thread.start()

    def on_model_ready(self, model_path: str):
        """Chamado quando o trainer emite o caminho do modelo gerado."""
        # Anexar mensagem ao log e permitir fechamento automático
        self.log_text.append(f"[MODEL READY] {model_path}")
        # Você pode aceitar automaticamente o diálogo para propagar que o treino terminou
        try:
            self.accept()
        except Exception:
            pass
    
    def update_progress(self, message):
        """Atualiza o progresso do treinamento"""
        self.progress_label.setText(message)
        self.log_text.append(f"[{self.get_timestamp()}] {message}")
        
        # Rola para o final
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def training_finished(self, success, message):
        """Chamado quando o treinamento termina"""
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        
        if success:
            self.progress_label.setText("✅ Treinamento concluído com sucesso!")
            self.progress_label.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(self, "Sucesso", message)
            # Fecha o diálogo automaticamente quando o treinamento terminar com sucesso
            try:
                self.accept()
            except Exception:
                pass
        else:
            self.progress_label.setText("❌ Erro durante o treinamento")
            self.progress_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.critical(self, "Erro", message)
        
        self.log_text.append(f"[{self.get_timestamp()}] {message}")
        
        # Habilita botão para fechar
        self.cancel_btn.setText("Fechar")
        self.cancel_btn.setEnabled(True)
    
    def get_timestamp(self):
        """Retorna timestamp atual formatado"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def closeEvent(self, event):
        """Intercepta o fechamento da janela"""
        if self.trainer_thread and self.trainer_thread.isRunning():
            reply = QMessageBox.question(
                self, 
                "Confirmar", 
                "O treinamento está em andamento. Deseja realmente fechar?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.trainer_thread.terminate()
                self.trainer_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
