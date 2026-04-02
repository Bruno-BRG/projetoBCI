from brainbridge_v2.domain.entities.model_metadata import ModelMetadata
from brainbridge_v2.domain.entities.training_result import TrainingResult
from brainbridge_v2.interface_adapters.controllers.training_controller import (
    TrainingController,
)


class FakeTrainingGateway:
    def __init__(self):
        self.calls = []

    def train(self, csv_file_path, patient_id, progress_callback=None):
        self.calls.append((csv_file_path, patient_id))
        if progress_callback is not None:
            progress_callback("Iniciando treinamento...")
        return TrainingResult(
            model_path=f"C:/models/patient_{patient_id}.keras",
            training_time_seconds=2.0,
            final_accuracy=0.9,
            val_accuracy=0.85,
        )


class FakeInferenceGateway:
    def __init__(self):
        self.loaded_model = None

    def load_model(self, model_path: str):
        self.loaded_model = ModelMetadata(path=model_path, name=model_path.split("/")[-1])
        return self.loaded_model

    def get_loaded_model(self):
        return self.loaded_model

    def predict(self, eeg_window):
        raise NotImplementedError


def test_training_controller_trains_and_auto_loads_model():
    training_gateway = FakeTrainingGateway()
    inference_gateway = FakeInferenceGateway()
    controller = TrainingController.from_gateways(training_gateway, inference_gateway)

    result = controller.train_model("C:/recordings/train.csv", 3)
    loaded_result = controller.train_and_load_model("C:/recordings/train.csv", 3)

    assert result.model_path.endswith("patient_3.keras")
    assert result.auto_loaded is False
    assert loaded_result.auto_loaded is True
    assert loaded_result.loaded_model_name == "patient_3.keras"
