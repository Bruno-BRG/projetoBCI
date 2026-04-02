"""
Application use cases.
"""

from .eeg_stream_use_cases import (
    ConnectEEGStreamUseCase,
    DisconnectEEGStreamUseCase,
)
from .esp32_use_cases import (
    ConnectESP32UseCase,
    DisconnectESP32UseCase,
    SendESP32SignalUseCase,
)
from .inference_use_cases import (
    GetLoadedModelUseCase,
    ListAvailableModelsUseCase,
    LoadModelUseCase,
    RunInferenceUseCase,
    SelectModelUseCase,
)
from .marker_use_cases import (
    GetMarkerStateUseCase,
    RegisterMarkerUseCase,
    ResetMarkerStateUseCase,
    StartBaselineUseCase,
    TickBaselineUseCase,
)
from .patient_use_cases import RegisterPatientUseCase, ListPatientsUseCase
from .recording_use_cases import (
    ListPatientRecordingsUseCase,
    StartRecordingUseCase,
    StopRecordingUseCase,
)
from .session_use_cases import (
    EndSessionUseCase,
    GetCurrentSessionUseCase,
    StartSessionUseCase,
)
from .training_use_cases import AutoLoadTrainedModelUseCase, TrainModelUseCase
from .unity_use_cases import (
    EndUnitySessionUseCase,
    EndUnityTaskUseCase,
    SendUnityActionUseCase,
    SendUnityTriggerUseCase,
    StartUnityServerUseCase,
    StopUnityServerUseCase,
)

__all__ = [
    "ConnectEEGStreamUseCase",
    "DisconnectEEGStreamUseCase",
    "ConnectESP32UseCase",
    "DisconnectESP32UseCase",
    "SendESP32SignalUseCase",
    "ListAvailableModelsUseCase",
    "LoadModelUseCase",
    "SelectModelUseCase",
    "GetLoadedModelUseCase",
    "RunInferenceUseCase",
    "RegisterPatientUseCase",
    "ListPatientsUseCase",
    "GetMarkerStateUseCase",
    "ResetMarkerStateUseCase",
    "RegisterMarkerUseCase",
    "StartBaselineUseCase",
    "TickBaselineUseCase",
    "StartRecordingUseCase",
    "StopRecordingUseCase",
    "ListPatientRecordingsUseCase",
    "StartSessionUseCase",
    "GetCurrentSessionUseCase",
    "EndSessionUseCase",
    "TrainModelUseCase",
    "AutoLoadTrainedModelUseCase",
    "StartUnityServerUseCase",
    "StopUnityServerUseCase",
    "SendUnityActionUseCase",
    "SendUnityTriggerUseCase",
    "EndUnityTaskUseCase",
    "EndUnitySessionUseCase",
]

