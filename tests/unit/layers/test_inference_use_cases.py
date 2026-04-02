from brainbridge_v2.application.use_cases.inference_use_cases import (
    GetLoadedModelUseCase,
    ListAvailableModelsUseCase,
    LoadModelUseCase,
    RunInferenceUseCase,
    SelectModelUseCase,
)
from brainbridge_v2.domain.entities.model_metadata import ModelMetadata
from brainbridge_v2.domain.entities.prediction_result import PredictionResult


class FakeModelCatalogGateway:
    def __init__(self):
        self.models = [
            ModelMetadata(path="C:/models/latest.keras", name="latest.keras"),
            ModelMetadata(path="C:/models/older.h5", name="older.h5"),
        ]

    def list_models(self):
        return self.models


class FakeInferenceGateway:
    def __init__(self):
        self.loaded_model = None
        self.loaded_paths = []
        self.prediction_windows = []

    def load_model(self, model_path: str):
        self.loaded_paths.append(model_path)
        self.loaded_model = ModelMetadata(path=model_path, name=model_path.split("/")[-1])
        return self.loaded_model

    def get_loaded_model(self):
        return self.loaded_model

    def predict(self, eeg_window):
        self.prediction_windows.append(eeg_window)
        return PredictionResult(
            predicted_index=1,
            confidence=0.75,
            probabilities=(0.25, 0.75),
        )


def test_inference_use_cases_cover_catalog_load_selection_and_prediction():
    model_catalog_gateway = FakeModelCatalogGateway()
    inference_gateway = FakeInferenceGateway()

    models = ListAvailableModelsUseCase(model_catalog_gateway).execute()
    selected = SelectModelUseCase(model_catalog_gateway, inference_gateway).execute(
        "C:/models/latest.keras"
    )
    loaded = LoadModelUseCase(inference_gateway).execute("C:/models/direct.keras")
    current = GetLoadedModelUseCase(inference_gateway).execute()
    prediction = RunInferenceUseCase(inference_gateway).execute([[1.0, 2.0], [3.0, 4.0]])

    assert [model.name for model in models] == ["latest.keras", "older.h5"]
    assert selected.path == "C:/models/latest.keras"
    assert loaded.path == "C:/models/direct.keras"
    assert current is not None
    assert current.path == "C:/models/direct.keras"
    assert prediction.predicted_index == 1
    assert inference_gateway.prediction_windows == [[[1.0, 2.0], [3.0, 4.0]]]


def test_select_model_use_case_rejects_unknown_path():
    model_catalog_gateway = FakeModelCatalogGateway()
    inference_gateway = FakeInferenceGateway()

    try:
        SelectModelUseCase(model_catalog_gateway, inference_gateway).execute(
            "C:/models/missing.keras"
        )
    except ValueError as exc:
        assert "catalogo" in str(exc)
    else:
        raise AssertionError("Era esperado erro ao selecionar modelo desconhecido.")
