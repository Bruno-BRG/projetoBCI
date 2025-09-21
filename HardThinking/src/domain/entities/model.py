"""
Entidade Model - Representa um modelo de ML treinado
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
from enum import Enum

class ModelStatus(Enum):
    """Status do modelo"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ERROR = "error"

class ModelArchitecture(Enum):
    """Arquiteturas de modelo suportadas"""
    CNN_1D = "CNN_1D"
    LSTM = "LSTM"
    EEGNET = "EEGNET"
    TRANSFORMER = "TRANSFORMER"

@dataclass
class TrainingMetrics:
    """Métricas de treinamento"""
    accuracy: float
    loss: float
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    training_time: Optional[float] = None
    epochs_trained: Optional[int] = None

@dataclass
class Model:
    """Entidade que representa um modelo de ML"""
    
    id: str
    name: str
    architecture: ModelArchitecture
    status: ModelStatus
    file_path: Optional[str] = None
    subject_ids: List[str] = field(default_factory=list)
    training_metrics: Optional[TrainingMetrics] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    trained_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    @classmethod
    def create(cls,
               name: str,
               architecture: ModelArchitecture,
               hyperparameters: Optional[Dict[str, Any]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> 'Model':
        """Factory method para criar instância de Model"""
        
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            architecture=architecture,
            status=ModelStatus.TRAINING,
            subject_ids=[],
            hyperparameters=hyperparameters or {},
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def set_training_completed(self, 
                              metrics: TrainingMetrics,
                              file_path: str):
        """Marca modelo como treinado"""
        self.status = ModelStatus.TRAINED
        self.training_metrics = metrics
        self.file_path = file_path
        self.trained_at = datetime.now()
        self.updated_at = datetime.now()
    
    def set_validated(self, validation_metrics: Dict[str, float]):
        """Marca modelo como validado"""
        if self.status == ModelStatus.TRAINED:
            self.status = ModelStatus.VALIDATED
            self.metadata.update({"validation_metrics": validation_metrics})
            self.updated_at = datetime.now()
    
    def set_deployed(self):
        """Marca modelo como deployado"""
        if self.status in [ModelStatus.TRAINED, ModelStatus.VALIDATED]:
            self.status = ModelStatus.DEPLOYED
            self.updated_at = datetime.now()
    
    def set_error(self, error_message: str):
        """Marca modelo com erro"""
        self.status = ModelStatus.ERROR
        self.metadata["error"] = error_message
        self.updated_at = datetime.now()
    
    def add_subject(self, subject_id: str):
        """Adiciona sujeito usado no treinamento"""
        if subject_id not in self.subject_ids:
            self.subject_ids.append(subject_id)
            self.updated_at = datetime.now()
    
    def get_accuracy(self) -> Optional[float]:
        """Retorna acurácia do modelo"""
        return self.training_metrics.accuracy if self.training_metrics else None
    
    def is_ready_for_prediction(self) -> bool:
        """Verifica se modelo está pronto para predição"""
        return (self.status in [ModelStatus.TRAINED, ModelStatus.VALIDATED, ModelStatus.DEPLOYED] 
                and self.file_path is not None)
    
    def validate(self) -> bool:
        """Valida se os dados do modelo estão consistentes"""
        if not self.name or not self.name.strip():
            return False
        if self.status == ModelStatus.TRAINED and not self.file_path:
            return False
        return True
