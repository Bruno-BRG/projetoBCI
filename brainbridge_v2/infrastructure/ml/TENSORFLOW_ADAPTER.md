# TensorFlow ML Adapter Documentation

## Overview
The `TensorFlowMLAdapter` is a bridge class that provides a clean interface for loading and using TensorFlow/Keras models in the BrainBridge system. It handles model loading, predictions, and provides compatibility with both in-process and subprocess-based inference.

## Location
`brainbridge_v2/ml/tensorflow_adapter.py`

## Class: TensorFlowMLAdapter

### Constructor
```python
adapter = TensorFlowMLAdapter(config: Optional[Dict[str, Any]] = None)
```

**Parameters:**
- `config` (dict, optional): Configuration dictionary (currently unused but reserved for future extensions)

**Example:**
```python
from ml.tensorflow_adapter import TensorFlowMLAdapter

adapter = TensorFlowMLAdapter(config={})
```

---

## Methods

### load_model(model_path: str)
Loads a Keras model from a .keras or .h5 file.

**Parameters:**
- `model_path` (str): Absolute or relative path to the model file

**Returns:**
- The loaded Keras model object

**Raises:**
- `FileNotFoundError`: If the model file doesn't exist
- `ImportError`: If TensorFlow is not installed

**Example:**
```python
model = adapter.load_model('/path/to/model.keras')
print(f"Model loaded: {model.name}")
```

---

### predict(data: np.ndarray) → np.ndarray
Runs inference on batch data.

**Parameters:**
- `data` (np.ndarray): Input batch with shape `(batch_size, timesteps, channels)`

**Returns:**
- Predictions array with shape `(batch_size, num_classes)`

**Raises:**
- `RuntimeError`: If no model has been loaded

**Example:**
```python
import numpy as np

# Create sample data: batch of 10 samples, 250 timesteps, 16 channels
batch = np.random.randn(10, 250, 16).astype('float32')
predictions = adapter.predict(batch)
print(predictions.shape)  # (10, 2) for binary classification
```

---

### predict_on_window(window: np.ndarray) → Dict[str, Any]
Predicts on a single EEG window (compatible with streaming prediction).

**Parameters:**
- `window` (np.ndarray): Single window with shape `(timesteps, channels)` = `(250, 16)` by default

**Returns:**
- Dictionary with keys:
  - `'probs'`: List of probabilities for each class `[p_left, p_right]`
  - `'label'`: Predicted class ('left' or 'right')
  - `'confidence'`: Float confidence score for the predicted class

**Raises:**
- `RuntimeError`: If no model has been loaded
- `ValueError`: If window shape is incorrect

**Example:**
```python
import numpy as np

# Single window: 250 timesteps, 16 channels
window = np.random.randn(250, 16).astype('float32')
result = adapter.predict_on_window(window)

print(f"Prediction: {result['label']}")
print(f"Probabilities: {result['probs']}")
print(f"Confidence: {result['confidence']:.2%}")
# Output:
# Prediction: left
# Probabilities: [0.75, 0.25]
# Confidence: 75.00%
```

---

### get_model_info() → Dict[str, Any]
Returns information about the currently loaded model.

**Returns:**
- Dictionary with model metadata:
  - `'loaded'`: Boolean indicating if a model is loaded
  - `'input_shape'`: Tuple of input shape (if available)
  - `'output_shape'`: Tuple of output shape (if available)
  - `'name'`: Model name (if available)

**Example:**
```python
adapter.load_model('/path/to/model.keras')
info = adapter.get_model_info()
print(info)
# Output:
# {
#     'loaded': True,
#     'input_shape': (None, 250, 16),
#     'output_shape': (None, 2),
#     'name': 'cnn1d_mi'
# }
```

---

## Integration with GUI

### Streaming Widget
The adapter is automatically used by the streaming widget for real-time predictions:

```python
# In brainbridge_v2/gui/widgets/streaming.py
from ml.tensorflow_adapter import TensorFlowMLAdapter

adapter = TensorFlowMLAdapter(config={})
model = adapter.load_model('data/models/trained_model.keras')

# For streaming predictions
window = get_eeg_window()  # shape: (250, 16)
result = adapter.predict_on_window(window)
```

---

## Integration with Training Pipeline

### Trainer Module
The trainer creates models compatible with this adapter:

```python
from ml.models import build_cnn_1d
from ml.trainer import train_model

# Build and train model
model = build_cnn_1d(input_shape=(250, 16), num_classes=2)
result = train_model(model, training_data, validation_data)

# Save model
model.save('data/models/my_model.keras')

# Later: load with adapter
adapter = TensorFlowMLAdapter()
loaded_model = adapter.load_model('data/models/my_model.keras')
```

---

## Error Handling

### TensorFlow Not Installed
If TensorFlow is not installed, the adapter will raise an `ImportError`:

```python
adapter = TensorFlowMLAdapter()
try:
    model = adapter.load_model('model.keras')
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    # Fallback to no-ML mode
```

### Model File Not Found
If the model file doesn't exist:

```python
try:
    model = adapter.load_model('/invalid/path/model.keras')
except FileNotFoundError as e:
    print(f"Model not found: {e}")
```

### No Model Loaded
If trying to predict without loading a model first:

```python
adapter = TensorFlowMLAdapter()
try:
    result = adapter.predict_on_window(window)
except RuntimeError as e:
    print(f"Model not loaded: {e}")
```

---

## Performance Considerations

### Batch Predictions vs Window Predictions
- **`predict()`** (batch): Efficient for multiple samples; use when processing recordings
- **`predict_on_window()`** (single): Slightly slower per-sample but simpler interface for streaming

### Model Size
- CNN models: typically 200KB-500KB (.keras format)
- Load time: ~100-500ms depending on system and storage
- Prediction latency: ~1-5ms per window (125 Hz = 8ms per window)

### GPU Acceleration
TensorFlow will automatically use GPU if available:
```bash
# Check if GPU is available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Examples

### Complete Example: Training and Predicting
```python
import numpy as np
from ml.models import build_cnn_1d
from ml.tensorflow_adapter import TensorFlowMLAdapter

# 1. Create and train model
model = build_cnn_1d(input_shape=(250, 16), num_classes=2)

# Training data (dummy for example)
X_train = np.random.randn(1000, 250, 16).astype('float32')
y_train = np.random.randint(0, 2, 1000)

model.fit(X_train, y_train, epochs=5, validation_split=0.2)
model.save('data/models/trained.keras')

# 2. Load with adapter
adapter = TensorFlowMLAdapter()
loaded_model = adapter.load_model('data/models/trained.keras')

# 3. Make predictions
window = np.random.randn(250, 16).astype('float32')
result = adapter.predict_on_window(window)
print(f"Predicted: {result['label']} (confidence: {result['confidence']:.1%})")
```

---

## Compatibility

### Model Format
- ✅ `.keras` (TensorFlow 2.11+): Recommended
- ✅ `.h5` (Legacy HDF5): Supported for backward compatibility
- ❌ `.pb` (SavedModel): Not directly supported (requires extra setup)

### Python Version
- Python 3.8+
- TensorFlow 2.10+

### Operating Systems
- Windows (tested)
- Linux (compatible)
- macOS (compatible)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ImportError: No module named 'tensorflow' | Install: `pip install tensorflow` |
| Model shape mismatch | Ensure input shape matches model: `(250, 16)` |
| Slow predictions | Enable GPU: check TensorFlow GPU setup |
| File not found | Use absolute paths or verify file exists |
| Out of memory | Reduce batch size or use single window predictions |

---

## References
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- BrainBridge Training Pipeline: `brainbridge_v2/ml/trainer.py`
- BrainBridge Streaming: `brainbridge_v2/gui/widgets/streaming.py`
