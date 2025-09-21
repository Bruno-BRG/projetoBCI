"""
Interface do repositório de modelos
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..entities.model import Model, ModelStatus, ModelArchitecture
from ..value_objects.training_types import ModelPerformance, ValidationResult

class ModelRepository(ABC):
    """Interface para repositório de modelos"""
    
    @abstractmethod
    def save_model(self, model: Model) -> bool:
        """Salva modelo"""
        pass
    
    @abstractmethod
    def get_model_by_id(self, model_id: str) -> Optional[Model]:
        """Obtém modelo por ID"""
        pass
    
    @abstractmethod
    def get_models_by_subject(self, subject_id: str) -> List[Model]:
        """Obtém modelos treinados para um sujeito"""
        pass
    
    @abstractmethod
    def get_models_by_status(self, status: ModelStatus) -> List[Model]:
        """Obtém modelos por status"""
        pass
    
    @abstractmethod
    def get_models_by_architecture(self, architecture: ModelArchitecture) -> List[Model]:
        """Obtém modelos por arquitetura"""
        pass
    
    @abstractmethod
    def get_best_model_for_subject(self, subject_id: str) -> Optional[Model]:
        """Obtém melhor modelo para um sujeito"""
        pass
    
    @abstractmethod
    def update_model(self, model: Model) -> bool:
        """Atualiza modelo"""
        pass
    
    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Remove modelo"""
        pass
    
    @abstractmethod
    def load_model_file(self, model_id: str) -> Optional[Any]:
        """Carrega arquivo do modelo para predição"""
        pass
    
    @abstractmethod
    def save_model_file(self, model: Model, model_object: Any) -> bool:
        """Salva arquivo do modelo"""
        pass
    
    @abstractmethod
    def get_model_performance_history(self, model_id: str) -> List[ModelPerformance]:
        """Obtém histórico de performance do modelo"""
        pass
    
    @abstractmethod
    def compare_models(self, model_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Compara performance de múltiplos modelos"""
        pass
