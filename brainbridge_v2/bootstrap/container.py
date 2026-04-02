"""
Centralized dependency composition for the desktop application.
"""

from dataclasses import dataclass

from brainbridge_v2.infrastructure.acquisition.eeg_stream_gateway_adapter import (
    EEGStreamGatewayAdapter,
)
from brainbridge_v2.infrastructure.communication.esp32_gateway_adapter import (
    ESP32GatewayAdapter,
)
from brainbridge_v2.infrastructure.communication.unity_gateway_adapter import (
    UnityGatewayAdapter,
)
from brainbridge_v2.infrastructure.database.manager import DatabaseManager
from brainbridge_v2.infrastructure.ml.model_catalog_gateway_adapter import (
    FileSystemModelCatalogGatewayAdapter,
)
from brainbridge_v2.infrastructure.ml.tensorflow_inference_gateway_adapter import (
    TensorFlowInferenceGatewayAdapter,
)
from brainbridge_v2.infrastructure.ml.training_gateway_adapter import (
    ModelTrainingGatewayAdapter,
)
from brainbridge_v2.infrastructure.repositories.sqlite_patient_repository import (
    SQLitePatientRepository,
)
from brainbridge_v2.infrastructure.repositories.sqlite_recording_repository import (
    SQLiteRecordingRepository,
)
from brainbridge_v2.infrastructure.state.in_memory_marker_state_store import (
    InMemoryMarkerStateStore,
)
from brainbridge_v2.infrastructure.state.in_memory_session_store import (
    InMemorySessionStore,
)
from brainbridge_v2.interface_adapters.controllers.eeg_stream_controller import (
    EEGStreamController,
)
from brainbridge_v2.interface_adapters.controllers.esp32_controller import (
    ESP32Controller,
)
from brainbridge_v2.interface_adapters.controllers.inference_controller import (
    InferenceController,
)
from brainbridge_v2.interface_adapters.controllers.marker_controller import (
    MarkerController,
)
from brainbridge_v2.interface_adapters.controllers.patient_controller import (
    PatientController,
)
from brainbridge_v2.interface_adapters.controllers.recording_controller import (
    RecordingController,
)
from brainbridge_v2.interface_adapters.controllers.session_controller import (
    SessionController,
)
from brainbridge_v2.interface_adapters.controllers.training_controller import (
    TrainingController,
)
from brainbridge_v2.interface_adapters.controllers.unity_controller import (
    UnityController,
)


@dataclass(frozen=True)
class AppContainer:
    db_manager: DatabaseManager
    eeg_stream_controller: EEGStreamController
    inference_controller: InferenceController
    training_controller: TrainingController
    patient_controller: PatientController
    recording_controller: RecordingController
    session_controller: SessionController
    marker_controller: MarkerController
    unity_controller: UnityController
    esp32_controller: ESP32Controller


def build_app_container() -> AppContainer:
    """
    Builds the object graph for the desktop application.
    """
    db_manager = DatabaseManager()
    if db_manager.test_connection():
        print("Sistema BCI inicializado com banco de dados funcionando")
    else:
        print("Aviso: Problemas com o banco de dados")

    patient_controller = PatientController.from_repository(
        SQLitePatientRepository(db_manager)
    )
    eeg_stream_controller = EEGStreamController.from_gateway(EEGStreamGatewayAdapter())
    model_catalog_gateway = FileSystemModelCatalogGatewayAdapter()
    inference_gateway = TensorFlowInferenceGatewayAdapter()
    inference_controller = InferenceController.from_gateways(
        model_catalog_gateway,
        inference_gateway,
    )
    training_controller = TrainingController.from_gateways(
        ModelTrainingGatewayAdapter(),
        inference_gateway,
    )
    recording_controller = RecordingController.from_repository(
        SQLiteRecordingRepository(db_manager)
    )
    session_controller = SessionController.from_store(InMemorySessionStore())
    marker_controller = MarkerController.from_store(InMemoryMarkerStateStore())
    unity_controller = UnityController.from_gateway(UnityGatewayAdapter())
    esp32_controller = ESP32Controller.from_gateway(ESP32GatewayAdapter())

    return AppContainer(
        db_manager=db_manager,
        eeg_stream_controller=eeg_stream_controller,
        inference_controller=inference_controller,
        training_controller=training_controller,
        patient_controller=patient_controller,
        recording_controller=recording_controller,
        session_controller=session_controller,
        marker_controller=marker_controller,
        unity_controller=unity_controller,
        esp32_controller=esp32_controller,
    )
