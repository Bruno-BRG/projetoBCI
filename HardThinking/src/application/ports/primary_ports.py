"""
Ports de entrada da aplicação (Primary Ports)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from ..domain.entities.eeg_data import EEGData
from ..domain.entities.subject import Subject
from ..domain.entities.model import Model
from ..domain.value_objects.eeg_types import EEGSegment, Prediction
from ..domain.value_objects.training_types import TrainingStrategy, ValidationResult

class EEGDataProcessingPort(ABC):
    """Port para processamento de dados EEG"""
    
    @abstractmethod
    def load_eeg_data_from_file(self, file_path: str, subject_id: str) -> Optional[EEGData]:
        """Carrega dados EEG de arquivo"""
        pass
    
    @abstractmethod
    def extract_motor_imagery_segments(self, eeg_data: EEGData) -> List[EEGSegment]:
        """Extrai segmentos de imagética motora"""
        pass
    
    @abstractmethod
    def preprocess_segments(self, segments: List[EEGSegment]) -> List[EEGSegment]:
        """Pré-processa segmentos"""
        pass
    
    @abstractmethod
    def analyze_data_quality(self, eeg_data: EEGData) -> Dict[str, Any]:
        """Analisa qualidade dos dados"""
        pass

class ModelTrainingPort(ABC):
    """Port para treinamento de modelos"""
    
    @abstractmethod
    def train_single_subject_model(self, 
                                  subject_id: str,
                                  segments: List[EEGSegment]) -> Model:
        """Treina modelo para um único sujeito"""
        pass
    
    @abstractmethod
    def train_cross_validation_model(self,
                                   segments: List[EEGSegment],
                                   cv_folds: int = 5) -> ValidationResult:
        """Treina modelo com validação cruzada"""
        pass
    
    @abstractmethod
    def train_leave_one_out_model(self,
                                subjects_data: Dict[str, List[EEGSegment]]) -> ValidationResult:
        """Treina modelo com leave-one-out validation"""
        pass
    
    @abstractmethod
    def train_multi_subject_model(self,
                                subjects_data: Dict[str, List[EEGSegment]]) -> Model:
        """Treina modelo com dados de múltiplos sujeitos"""
        pass

class ModelPredictionPort(ABC):
    """Port para predição com modelos"""
    
    @abstractmethod
    def predict_motor_imagery(self, model_id: str, segment: EEGSegment) -> Prediction:
        """Faz predição de imagética motora"""
        pass
    
    @abstractmethod
    def predict_realtime(self, model_id: str, raw_data: Any) -> Prediction:
        """Faz predição em tempo real"""
        pass
    
    @abstractmethod
    def batch_predict(self, model_id: str, segments: List[EEGSegment]) -> List[Prediction]:
        """Faz predições em lote"""
        pass

class DataAnalysisPort(ABC):
    """Port para análise de dados"""
    
    @abstractmethod
    def analyze_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Analisa dados de um sujeito"""
        pass
    
    @abstractmethod
    def analyze_all_data(self) -> Dict[str, Any]:
        """Analisa todos os dados disponíveis"""
        pass
    
    @abstractmethod
    def generate_analysis_report(self, analysis_data: Dict[str, Any]) -> str:
        """Gera relatório de análise"""
        pass
    
    @abstractmethod
    def compare_subjects(self, subject_ids: List[str]) -> Dict[str, Any]:
        """Compara dados entre sujeitos"""
        pass

class ModelManagementPort(ABC):
    """Port para gerenciamento de modelos"""
    
    @abstractmethod
    def list_models(self, filters: Optional[Dict[str, Any]] = None) -> List[Model]:
        """Lista modelos disponíveis"""
        pass
    
    @abstractmethod
    def get_model_details(self, model_id: str) -> Optional[Model]:
        """Obtém detalhes de um modelo"""
        pass
    
    @abstractmethod
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compara performance de modelos"""
        pass
    
    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Remove modelo"""
        pass
    
    @abstractmethod
    def export_model(self, model_id: str, export_path: str) -> bool:
        """Exporta modelo"""
        pass
