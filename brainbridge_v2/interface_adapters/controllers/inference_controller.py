"""
Controller that adapts presentation requests to inference use cases.
"""

from typing import List, Optional, Sequence

from brainbridge_v2.application.ports.inference_gateway import InferenceGateway
from brainbridge_v2.application.ports.model_catalog_gateway import ModelCatalogGateway
from brainbridge_v2.application.use_cases.inference_use_cases import (
    GetLoadedModelUseCase,
    ListAvailableModelsUseCase,
    LoadModelUseCase,
    RunInferenceUseCase,
    SelectModelUseCase,
)
from brainbridge_v2.interface_adapters.presenters.streaming_presenter import (
    InferencePresenter,
    ModelViewModel,
    PredictionViewModel,
)


class InferenceController:
    """
    Presentation-facing controller for model discovery, loading and inference.
    """

    def __init__(
        self,
        list_available_models_use_case: ListAvailableModelsUseCase,
        load_model_use_case: LoadModelUseCase,
        select_model_use_case: SelectModelUseCase,
        get_loaded_model_use_case: GetLoadedModelUseCase,
        run_inference_use_case: RunInferenceUseCase,
    ):
        self._list_available_models_use_case = list_available_models_use_case
        self._load_model_use_case = load_model_use_case
        self._select_model_use_case = select_model_use_case
        self._get_loaded_model_use_case = get_loaded_model_use_case
        self._run_inference_use_case = run_inference_use_case

    @classmethod
    def from_gateways(
        cls,
        model_catalog_gateway: ModelCatalogGateway,
        inference_gateway: InferenceGateway,
    ) -> "InferenceController":
        return cls(
            list_available_models_use_case=ListAvailableModelsUseCase(
                model_catalog_gateway
            ),
            load_model_use_case=LoadModelUseCase(inference_gateway),
            select_model_use_case=SelectModelUseCase(
                model_catalog_gateway,
                inference_gateway,
            ),
            get_loaded_model_use_case=GetLoadedModelUseCase(inference_gateway),
            run_inference_use_case=RunInferenceUseCase(inference_gateway),
        )

    def list_models(self) -> List[ModelViewModel]:
        models = self._list_available_models_use_case.execute()
        return [InferencePresenter.present_model(model) for model in models]

    def load_model(self, model_path: str) -> ModelViewModel:
        model = self._load_model_use_case.execute(model_path)
        return InferencePresenter.present_model(model)

    def select_model(self, model_path: str) -> ModelViewModel:
        model = self._select_model_use_case.execute(model_path)
        return InferencePresenter.present_model(model)

    def load_latest_model(self) -> ModelViewModel:
        models = self.list_models()
        if not models:
            raise ValueError("Nenhum modelo TensorFlow (.keras/.h5) foi encontrado.")
        return self.select_model(models[0].path)

    def get_loaded_model(self) -> Optional[ModelViewModel]:
        model = self._get_loaded_model_use_case.execute()
        if model is None:
            return None
        return InferencePresenter.present_model(model)

    def has_loaded_model(self) -> bool:
        return self.get_loaded_model() is not None

    def predict(self, eeg_window: Sequence[Sequence[float]]) -> PredictionViewModel:
        normalized_window: Sequence[Sequence[float]]
        if hasattr(eeg_window, "tolist"):
            normalized_window = eeg_window.tolist()
        else:
            normalized_window = eeg_window

        result = self._run_inference_use_case.execute(normalized_window)
        return InferencePresenter.present_prediction(result)
