"""
Entidade EEG Data - Representa dados de EEG
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime
import uuid

@dataclass
class EEGData:
    """Entidade que representa dados de EEG"""
    
    id: str
    subject_id: str
    session_id: Optional[str]
    raw_data: np.ndarray  # Shape: (samples, channels)
    annotations: List[str]
    sample_rate: int
    channels: int
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def create(cls, 
               subject_id: str,
               raw_data: np.ndarray,
               annotations: List[str],
               sample_rate: int = 125,
               channels: int = 16,
               session_id: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> 'EEGData':
        """Factory method para criar instância de EEGData"""
        
        return cls(
            id=str(uuid.uuid4()),
            subject_id=subject_id,
            session_id=session_id,
            raw_data=raw_data,
            annotations=annotations,
            sample_rate=sample_rate,
            channels=channels,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
    
    def get_duration_seconds(self) -> float:
        """Retorna a duração dos dados em segundos"""
        return len(self.raw_data) / self.sample_rate
    
    def get_channel_data(self, channel_index: int) -> np.ndarray:
        """Retorna dados de um canal específico"""
        if channel_index >= self.channels:
            raise ValueError(f"Canal {channel_index} não existe (máximo: {self.channels-1})")
        return self.raw_data[:, channel_index]
    
    def get_time_window(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Retorna uma janela temporal dos dados"""
        return self.raw_data[start_sample:end_sample]
    
    def validate(self) -> bool:
        """Valida se os dados estão consistentes"""
        if self.raw_data.shape[1] != self.channels:
            return False
        if len(self.annotations) != len(self.raw_data):
            return False
        if self.sample_rate <= 0:
            return False
        return True
