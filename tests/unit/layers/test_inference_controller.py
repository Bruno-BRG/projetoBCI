from brainbridge_v2.domain.entities.model_metadata import ModelMetadata
from brainbridge_v2.domain.entities.prediction_result import PredictionResult
from brainbridge_v2.interface_adapters.controllers.inference_controller import (
    InferenceController,
)


class FakeModelCatalogGateway:
    def __init__(self):
        self.models = [
            ModelMetadata(
                path="C:/models/latest.keras",
                name="latest.keras",
                expected_time_steps=250,
                expected_channels=16,
                modified_at_epoch=20.0,
            ),
            ModelMetadata(
                path="C:/models/older.h5",
                name="older.h5",
                expected_time_steps=200,
                expected_channels=8,
                modified_at_epoch=10.0,
            ),
        ]

    def list_models(self):
        return self.models


class FakeInferenceGateway:
    def __init__(self):
        self.loaded_model = None
        self.prediction_window = None

    def load_model(self, model_path: str):
        self.loaded_model = ModelMetadata(
            path=model_path,
            name=model_path.split("/")[-1],
            expected_time_steps=250,
            expected_channels=16,
        )
        return self.loaded_model

    def get_loaded_model(self):
        return self.loaded_model

    def predict(self, eeg_window):
        self.prediction_window = eeg_window
        return PredictionResult(
            predicted_index=1,
            confidence=0.9,
            probabilities=(0.1, 0.9),
        )


def test_inference_controller_lists_loads_and_predicts():
    controller = InferenceController.from_gateways(
        FakeModelCatalogGateway(),
        FakeInferenceGateway(),
    )

    models = controller.list_models()
    latest = controller.load_latest_model()
    current = controller.get_loaded_model()
    prediction = controller.predict([[1.0, 2.0], [3.0, 4.0]])

    assert [model.name for model in models] == ["latest.keras", "older.h5"]
    assert latest.path == "C:/models/latest.keras"
    assert controller.has_loaded_model() is True
    assert current is not None
    assert current.expected_channels == 16
    assert prediction.predicted_index == 1
    assert prediction.right_probability == 0.9
