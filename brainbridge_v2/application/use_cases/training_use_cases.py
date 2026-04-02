"""
Training-related use cases.
"""

from typing import Callable, Optional

from brainbridge_v2.application.ports.inference_gateway import InferenceGateway
from brainbridge_v2.application.ports.training_gateway import TrainingGateway
from brainbridge_v2.domain.entities.training_result import TrainingResult


class TrainModelUseCase:
    """
    Trains a model from a captured EEG recording.
    """

    def __init__(self, training_gateway: TrainingGateway):
        self._training_gateway = training_gateway

    def execute(
        self,
        csv_file_path: str,
        patient_id: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TrainingResult:
        normalized_path = str(csv_file_path).strip()
        if not normalized_path:
            raise ValueError("Caminho do CSV de treinamento eh obrigatorio.")
        if patient_id <= 0:
            raise ValueError("Paciente invalido para treinamento.")

        result = self._training_gateway.train(
            normalized_path,
            patient_id,
            progress_callback=progress_callback,
        )
        result.validate()
        return result


class AutoLoadTrainedModelUseCase:
    """
    Trains a model and immediately loads it into the inference runtime.
    """

    def __init__(
        self,
        training_gateway: TrainingGateway,
        inference_gateway: InferenceGateway,
    ):
        self._training_gateway = training_gateway
        self._inference_gateway = inference_gateway

    def execute(
        self,
        csv_file_path: str,
        patient_id: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TrainingResult:
        normalized_path = str(csv_file_path).strip()
        if not normalized_path:
            raise ValueError("Caminho do CSV de treinamento eh obrigatorio.")
        if patient_id <= 0:
            raise ValueError("Paciente invalido para treinamento.")

        result = self._training_gateway.train(
            normalized_path,
            patient_id,
            progress_callback=progress_callback,
        )
        result.validate()
        self._inference_gateway.load_model(result.model_path)
        return result
