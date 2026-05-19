"""
Inference gateway backed by TensorFlow models.
"""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from brainbridge_v2.domain.entities.model_metadata import ModelMetadata
from brainbridge_v2.domain.entities.prediction_result import PredictionResult
from brainbridge_v2.application.runtime_config import DEFAULT_RUNTIME_CONFIG
from brainbridge_v2.infrastructure.ml.tensorflow_adapter import TensorFlowMLAdapter


class TensorFlowInferenceGatewayAdapter:
    """
    Loads TensorFlow models and executes inference on normalized EEG windows.
    """

    def __init__(
        self,
        adapter_factory: Optional[Callable[[], TensorFlowMLAdapter]] = None,
        *,
        warmup_enabled: bool = DEFAULT_RUNTIME_CONFIG.tensorflow_warmup_enabled,
    ):
        self._adapter_factory = adapter_factory or (
            lambda: TensorFlowMLAdapter(config={})
        )
        self._warmup_enabled = bool(warmup_enabled)
        self._adapter: Optional[TensorFlowMLAdapter] = None
        self._loaded_model: Optional[ModelMetadata] = None

    def load_model(self, model_path: str) -> ModelMetadata:
        adapter = self._adapter_factory()
        model = adapter.load_model(model_path)

        self._adapter = adapter
        self._loaded_model = self._build_model_metadata(model_path, model)
        self._loaded_model.validate()
        if self._warmup_enabled:
            self._warmup_model()
        return self._loaded_model

    def get_loaded_model(self) -> Optional[ModelMetadata]:
        return self._loaded_model

    def predict(self, eeg_window: Sequence[Sequence[float]]) -> PredictionResult:
        if self._adapter is None or self._adapter.model is None or self._loaded_model is None:
            raise RuntimeError("Nenhum modelo foi carregado para inferencia.")

        window = np.asarray(eeg_window, dtype="float32")
        if window.ndim != 2:
            raise ValueError("Janela EEG deve possuir shape (timesteps, channels).")
        if window.shape[0] == 0 or window.shape[1] == 0:
            raise ValueError("Janela EEG nao pode ser vazia.")

        normalized_window = self._normalize_window(window)
        adapted_window = self._adapt_window(
            normalized_window,
            self._loaded_model.expected_time_steps,
            self._loaded_model.expected_channels,
        )
        batch = adapted_window.reshape(1, adapted_window.shape[0], adapted_window.shape[1])
        raw_output = np.asarray(self._adapter.predict(batch), dtype="float32")

        if raw_output.ndim == 2:
            probabilities = raw_output[0]
        elif raw_output.ndim == 1:
            probabilities = raw_output
        else:
            raise ValueError("Saida de inferencia inesperada.")

        predicted_index = int(np.argmax(probabilities))
        result = PredictionResult(
            predicted_index=predicted_index,
            confidence=float(probabilities[predicted_index]),
            probabilities=tuple(float(value) for value in probabilities.tolist()),
        )
        result.validate()
        return result

    def _build_model_metadata(self, model_path: str, model: object) -> ModelMetadata:
        input_shape = self._extract_input_shape(model)
        expected_time_steps, expected_channels = self._extract_expected_dimensions(
            input_shape
        )

        return ModelMetadata(
            path=str(model_path),
            name=str(model_path).split("\\")[-1].split("/")[-1],
            input_shape=input_shape,
            expected_time_steps=expected_time_steps,
            expected_channels=expected_channels,
        )

    def _warmup_model(self) -> None:
        if self._adapter is None or self._loaded_model is None:
            return
        time_steps = self._loaded_model.expected_time_steps
        channels = self._loaded_model.expected_channels
        if time_steps is None or channels is None:
            return
        dummy_batch = np.zeros((1, int(time_steps), int(channels)), dtype="float32")
        self._adapter.predict(dummy_batch)

    @staticmethod
    def _extract_input_shape(model: object) -> Optional[Tuple[Optional[int], ...]]:
        try:
            if hasattr(model, "input_shape") and model.input_shape is not None:
                return tuple(model.input_shape)
            if hasattr(model, "inputs") and getattr(model, "inputs"):
                return tuple(model.inputs[0].shape.as_list())
        except Exception:
            return None
        return None

    @staticmethod
    def _extract_expected_dimensions(
        input_shape: Optional[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], Optional[int]]:
        if input_shape is None:
            return None, None

        dims = list(input_shape)
        if len(dims) == 3 and dims[0] in (None, -1):
            return (
                TensorFlowInferenceGatewayAdapter._safe_int(dims[1]),
                TensorFlowInferenceGatewayAdapter._safe_int(dims[2]),
            )
        if len(dims) == 2:
            return (
                TensorFlowInferenceGatewayAdapter._safe_int(dims[0]),
                TensorFlowInferenceGatewayAdapter._safe_int(dims[1]),
            )
        return None, None

    @staticmethod
    def _safe_int(value: object) -> Optional[int]:
        if value in (None, -1):
            return None
        return int(value)

    @staticmethod
    def _normalize_window(window: np.ndarray) -> np.ndarray:
        normalized = window.copy()
        for channel_index in range(normalized.shape[1]):
            channel_data = normalized[:, channel_index]
            q75, q25 = np.percentile(channel_data, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1.0
            channel_mean = float(np.mean(channel_data))
            normalized[:, channel_index] = (channel_data - channel_mean) / iqr
        return normalized

    @staticmethod
    def _adapt_window(
        window: np.ndarray,
        expected_time_steps: Optional[int],
        expected_channels: Optional[int],
    ) -> np.ndarray:
        adapted = window

        if expected_time_steps is not None and expected_time_steps != adapted.shape[0]:
            if expected_time_steps < adapted.shape[0]:
                start = (adapted.shape[0] - expected_time_steps) // 2
                adapted = adapted[start : start + expected_time_steps, :]
            else:
                padding_rows = expected_time_steps - adapted.shape[0]
                padding = np.zeros((padding_rows, adapted.shape[1]), dtype=adapted.dtype)
                adapted = np.vstack([adapted, padding])

        if expected_channels is not None and expected_channels != adapted.shape[1]:
            if expected_channels < adapted.shape[1]:
                adapted = adapted[:, :expected_channels]
            else:
                padding_columns = expected_channels - adapted.shape[1]
                padding = np.zeros((adapted.shape[0], padding_columns), dtype=adapted.dtype)
                adapted = np.hstack([adapted, padding])

        return adapted
