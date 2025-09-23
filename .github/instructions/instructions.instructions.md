# BrainBridge AI Agent Instructions

## Project Overview

**BrainBridge** is a dual-architecture EEG Brain-Computer Interface system for motor imagery classification. It consists of two main components:

- **`bci/`**: Original PyQt5-based GUI for real-time EEG data streaming, patient registration, and data recording
- **`HardThinking/`**: Refactored training system using hexagonal architecture for model development and evaluation

## Architecture Patterns

### Hexagonal Architecture (HardThinking)
The `HardThinking/` module follows Clean Architecture principles:
- **Domain layer**: Entities (`EEGData`, `Subject`, `Model`) and value objects in `src/domain/`
- **Application layer**: Use cases and ports (interfaces) in `src/application/`
- **Infrastructure layer**: Adapters for TensorFlow, filesystem, logging in `src/infrastructure/`
- **Interface layer**: CLI interface in `src/interfaces/cli/`

Key pattern: Dependencies point inward. Always inject dependencies through constructors, never import infrastructure from domain.

### BCI Module Patterns
- **PyQt5 Threading**: Heavy operations run in `QThread` subclasses to avoid UI freezing
- **Real-time data**: Uses `deque` for sliding window visualization and UDP sockets for external communication
- **Patient-linked recording**: All CSV files are tied to registered patients with automatic filename generation

## Entry Points & Workflows

### Running the System
```bash
# GUI Interface (streaming/recording)
cd BrainBridge
python -m bci                    # Preferred method
python bci/main.py              # Direct execution

# Training System (model development)
cd BrainBridge/HardThinking
python main.py                  # CLI interface
python main.py --help          # See all options
```

### Key Dependencies
- **PyQt5**: GUI framework for real-time interface
- **TensorFlow**: ML models with graceful degradation if unavailable
- **Matplotlib**: Real-time EEG plotting in GUI
- **NumPy/Pandas**: Signal processing and data manipulation

## Critical Implementation Patterns

### TensorFlow Integration
The system includes fallback mechanisms for TensorFlow unavailability:
```python
# In HardThinking CLI
try:
    from ...infrastructure.adapters.tensorflow_ml_adapter import TensorFlowMLAdapter
except Exception:
    TensorFlowMLAdapter = TensorFlowMLAdapterStub  # Provides clear error messages
```

### Python Path Management
Both modules use dynamic path injection for compatibility:
```python
# Common pattern in main.py files
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
```

### Data Flow Architecture
1. **UDP Reception**: `bci/network/udp_receiver_BCI.py` receives real-time EEG data
2. **CSV Logging**: `bci/network/openbci_csv_logger.py` saves timestamped data with markers
3. **Training Pipeline**: `HardThinking/` processes CSV files into trained TensorFlow models
4. **Model Storage**: Saved models go to both `models/` directories for different use cases

### Logging & Configuration
- **Centralized config**: `HardThinking/src/config.py` uses dataclasses for type-safe configuration
- **Structured logging**: Logs saved to `logs/` with timestamped files for debugging
- **Graceful degradation**: Missing dependencies trigger informative error messages, not crashes

## Domain-Specific Knowledge

### EEG Data Handling
- **Sample rate**: 125 Hz standard across the system
- **Channels**: 16 EEG channels (0-15)
- **Window size**: 250 samples (2 seconds) with 50% overlap for classification
- **Markers**: T1 (left motor imagery), T2 (right motor imagery), T0 (end marker)

### Model Training Patterns
- **Cross-validation**: Subject-specific and cross-subject validation supported
- **Architecture**: 1D CNN with configurable layers (default: [64, 64, 128] filters)
- **Data augmentation**: Built into the training pipeline for better generalization

## Integration Points

### External Communication
- **UDP Protocol**: Default port 12345 for real-time data streaming
- **Serial Communication**: ESP32 integration in `bci/network/esp32_serial_communication.py`
- **Unity Integration**: `bci/network/unity_communication.py` for game engine connectivity

### Database Integration
- **SQLite**: Patient registration stored in `bci/database/database_manager.py`
- **File-based storage**: CSV files linked to patient records for data lineage

## Development Guidelines

When modifying this codebase:
1. **Respect architectural boundaries**: Don't import infrastructure into domain layer
2. **Maintain compatibility**: Both entry points should continue working
3. **Handle TensorFlow gracefully**: Always provide fallbacks for missing dependencies
4. **Test real-time components**: GUI threading and UDP communication need careful testing
5. **Follow naming conventions**: Patient files use UUID-based naming for uniqueness

## Common Debugging Scenarios

- **TensorFlow DLL issues**: Check the preload pattern in `bci/main.py`
- **PyQt threading**: GUI freezes usually indicate missing `QThread` usage
- **Path import errors**: Verify the sys.path injection patterns are correct
- **UDP data loss**: Check buffer sizes and threading in receiver classes