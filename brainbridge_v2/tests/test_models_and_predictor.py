import numpy as np
import pytest


def test_build_model_if_tf_available():
    try:
        from brainbridge_v2.ml.models import build_cnn_1d
        model = build_cnn_1d(input_shape=(250, 16), num_classes=2)
        # Ensure compiled and has correct input shape
        assert model.input_shape[-2:] == (250, 16)
    except ImportError:
        pytest.skip("TensorFlow not installed")


def test_predictor_monkeypatched():
    from brainbridge_v2.ml.predictor import Predictor

    class DummyModel:
        def predict(self, x, verbose=0):
            # Always predict class 1 with high prob
            import numpy as _np
            batch = x.shape[0]
            probs = _np.zeros((batch, 2), dtype=float)
            probs[:, 1] = 0.9
            probs[:, 0] = 0.1
            return probs

    # Monkeypatch loader
    import brainbridge_v2.ml.models as M
    orig_loader = M.load_keras_model
    try:
        M.load_keras_model = lambda p: DummyModel()
        pred = Predictor('dummy.keras')
        out = pred.predict_window(np.zeros((250, 16), dtype='float32'))
        assert out['label'] == 'right'
        assert len(out['probs']) == 2
    finally:
        M.load_keras_model = orig_loader
