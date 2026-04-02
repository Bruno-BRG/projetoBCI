from PyQt5.QtCore import QThread, pyqtSignal

from brainbridge_v2.interface_adapters.controllers.training_controller import (
    TrainingController,
)


class ModelTrainerThread(QThread):
    """Thread que executa o treinamento via controller."""

    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    model_path_signal = pyqtSignal(str)

    def __init__(
        self,
        training_controller: TrainingController,
        csv_file_path: str,
        patient_id: int,
        auto_load_model: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.training_controller = training_controller
        self.csv_file_path = csv_file_path
        self.patient_id = patient_id
        self.auto_load_model = auto_load_model

    def run(self):
        try:
            if self.auto_load_model:
                result = self.training_controller.train_and_load_model(
                    self.csv_file_path,
                    self.patient_id,
                    progress_callback=self.progress_signal.emit,
                )
            else:
                result = self.training_controller.train_model(
                    self.csv_file_path,
                    self.patient_id,
                    progress_callback=self.progress_signal.emit,
                )

            model_path = str(result.model_path)
            self.model_path_signal.emit(model_path)

            if result.auto_loaded:
                self.finished_signal.emit(
                    True,
                    f"Modelo treinado e carregado: {model_path}",
                )
            else:
                self.finished_signal.emit(True, f"Modelo salvo em: {model_path}")
        except Exception as e:
            self.finished_signal.emit(False, f"Erro no treinamento: {e}")
