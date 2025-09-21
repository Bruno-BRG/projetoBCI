"""
Interface do repositório de dados EEG
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..entities.eeg_data import EEGData
from ..entities.subject import Subject
from ..value_objects.eeg_types import EEGSegment, TimeWindow, MotorImageryClass

class EEGDataRepository(ABC):
    """Interface para repositório de dados EEG"""
    
    @abstractmethod
    def save_eeg_data(self, eeg_data: EEGData) -> bool:
        """Salva dados EEG"""
        pass
    
    @abstractmethod
    def get_eeg_data_by_id(self, data_id: str) -> Optional[EEGData]:
        """Obtém dados EEG por ID"""
        pass
    
    @abstractmethod
    def get_eeg_data_by_subject(self, subject_id: str) -> List[EEGData]:
        """Obtém todos os dados EEG de um sujeito"""
        pass
    
    @abstractmethod
    def get_all_subjects(self) -> List[str]:
        """Obtém lista de todos os sujeitos"""
        pass
    
    @abstractmethod
    def delete_eeg_data(self, data_id: str) -> bool:
        """Remove dados EEG"""
        pass
    
    @abstractmethod
    def load_from_csv(self, file_path: str, subject_id: str) -> Optional[EEGData]:
        """Carrega dados EEG de arquivo CSV"""
        pass
    
    @abstractmethod
    def extract_segments(self, eeg_data: EEGData) -> List[EEGSegment]:
        """Extrai segmentos de dados EEG baseados em marcadores"""
        pass
    
    @abstractmethod
    def get_data_statistics(self, subject_id: str) -> Dict[str, Any]:
        """Obtém estatísticas dos dados de um sujeito"""
        pass

class SubjectRepository(ABC):
    """Interface para repositório de sujeitos"""
    
    @abstractmethod
    def save_subject(self, subject: Subject) -> bool:
        """Salva sujeito"""
        pass
    
    @abstractmethod
    def get_subject_by_id(self, subject_id: str) -> Optional[Subject]:
        """Obtém sujeito por ID"""
        pass
    
    @abstractmethod
    def get_all_subjects(self) -> List[Subject]:
        """Obtém todos os sujeitos"""
        pass
    
    @abstractmethod
    def update_subject(self, subject: Subject) -> bool:
        """Atualiza sujeito"""
        pass
    
    @abstractmethod
    def delete_subject(self, subject_id: str) -> bool:
        """Remove sujeito"""
        pass
    
    @abstractmethod
    def find_subjects_by_criteria(self, criteria: Dict[str, Any]) -> List[Subject]:
        """Busca sujeitos por critérios"""
        pass
