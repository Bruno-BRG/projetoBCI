import numpy as np

from brainbridge_v2.infrastructure.ml.tensorflow_inference_gateway_adapter import (
    TensorFlowInferenceGatewayAdapter,
)


class FakeModel:
    input_shape = (None, 3, 2)


class FakeTensorFlowAdapter:
    def __init__(self):
        self.model = None
        self.predicted_batches = []

    def load_model(self, model_path: str):
        self.model = FakeModel()
        return self.model

    def predict(self, data):
        self.predicted_batches.append(data)
        return np.array([[0.25, 0.75]], dtype=np.float32)


def test_tensorflow_inference_gateway_adapter_loads_model_and_predicts():
    created_adapters = []

    def build_adapter():
        adapter = FakeTensorFlowAdapter()
        created_adapters.append(adapter)
        return adapter

    gateway = TensorFlowInferenceGatewayAdapter(adapter_factory=build_adapter)

    model = gateway.load_model("C:/models/test.keras")
    result = gateway.predict([[1.0], [2.0], [3.0], [4.0]])

    assert model.expected_time_steps == 3
    assert model.expected_channels == 2
    assert gateway.get_loaded_model() is not None
    assert created_adapters[0].predicted_batches[0].shape == (1, 3, 2)
    assert result.predicted_index == 1
    assert result.right_probability == 0.75
