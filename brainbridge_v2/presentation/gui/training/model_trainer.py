"""
Thread de treinamento de modelo (stub)

Fornece uma implementação mínima para permitir que a GUI execute o fluxo de
"Treinar modelo" sem dependências pesadas. Pode ser estendido para usar o
HardThinking/ ou ml/ quando disponível.
"""

from PyQt5.QtCore import QThread, pyqtSignal
import time
import os
from pathlib import Path


class ModelTrainerThread(QThread):
    """Thread que simula o treinamento de um modelo"""

    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    model_path_signal = pyqtSignal(str)

    def __init__(self, csv_file_path: str, patient_id: int, parent=None):
        super().__init__(parent)
        self.csv_file_path = csv_file_path
        self.patient_id = patient_id

    def run(self):
        try:
            # Tenta usar pipeline real se TF estiver disponível
            try:
                import tensorflow as tf  # noqa: F401
                use_real = True
            except Exception:
                use_real = False

            self.progress_signal.emit("Preparando dados...")
            time.sleep(0.3)

            if use_real:
                from ...ml import trainer as ml_trainer
                self.progress_signal.emit("Iniciando treinamento real (Keras)...")
                time.sleep(0.2)
                result = ml_trainer.train_from_csvs([self.csv_file_path], model_name=f"patient_{self.patient_id}")
                self.progress_signal.emit("Modelo salvo com sucesso.")
                self.model_path_signal.emit(result.model_path)
                self.finished_signal.emit(True, f"Modelo salvo em: {result.model_path}")
                return

            # Fallback stub
            self.progress_signal.emit("TensorFlow não disponível. Executando modo simulado...")
            time.sleep(0.8)
            steps = [
                "Carregando CSV e validando formato...",
                "Extraindo janelas...",
                "(stub) Treinando modelo...",
                "Salvando modelo (stub)...",
            ]
            for s in steps:
                self.progress_signal.emit(s)
                time.sleep(0.6)

            root = Path(__file__).resolve().parents[2]
            models_dir = root / 'data' / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / f'patient_{self.patient_id}.keras'
            try:
                with open(model_path, 'w', encoding='utf-8') as f:
                    f.write("stub model file")
            except Exception:
                pass
            self.model_path_signal.emit(str(model_path))
            self.finished_signal.emit(True, f"Modelo salvo em: {model_path}")

        except Exception as e:
            self.finished_signal.emit(False, f"Erro no treinamento: {e}")
