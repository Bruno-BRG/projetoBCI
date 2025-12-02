"""
Classes para dados EEG
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class EEGSample:
    """Representa uma amostra de dados EEG"""
    timestamp: datetime
    channels: np.ndarray  # Array com dados dos canais
    marker: Optional[str] = None
    
    def __post_init__(self):
        """Validação dos dados"""
        if not isinstance(self.channels, np.ndarray):
            self.channels = np.array(self.channels)


class EEGBuffer:
    """Buffer circular para dados EEG em tempo real"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.data: List[EEGSample] = []
        self._index = 0
    
    def add_sample(self, sample: EEGSample):
        """Adiciona uma nova amostra ao buffer"""
        if len(self.data) < self.max_size:
            self.data.append(sample)
        else:
            self.data[self._index] = sample
            self._index = (self._index + 1) % self.max_size
    
    def get_latest_samples(self, n: int) -> List[EEGSample]:
        """Retorna as n amostras mais recentes"""
        if n > len(self.data):
            return self.data.copy()
        return self.data[-n:]
    
    def get_window(self, size: int) -> Optional[np.ndarray]:
        """Retorna uma janela de dados como array 2D (amostras x canais)"""
        if len(self.data) < size:
            return None
        
        samples = self.get_latest_samples(size)
        return np.array([sample.channels for sample in samples])
    
    def clear(self):
        """Limpa o buffer"""
        self.data.clear()
        self._index = 0


class EEGSession:
    """Representa uma sessão de coleta de dados EEG"""
    
    def __init__(self, subject_id: str, session_type: str = "motor_imagery"):
        self.subject_id = subject_id
        self.session_type = session_type
        self.start_time = datetime.now()
        self.samples: List[EEGSample] = []
        self.metadata = {}
    
    def add_sample(self, sample: EEGSample):
        """Adiciona uma amostra à sessão"""
        self.samples.append(sample)
    
    def get_samples_by_marker(self, marker: str) -> List[EEGSample]:
        """Retorna amostras filtradas por marcador"""
        return [sample for sample in self.samples if sample.marker == marker]
    
    def get_duration(self) -> float:
        """Retorna a duração da sessão em segundos"""
        if not self.samples:
            return 0.0
        
        last_sample = self.samples[-1]
        return (last_sample.timestamp - self.start_time).total_seconds()
    
    def to_array(self) -> np.ndarray:
        """Converte a sessão para array numpy (amostras x canais)"""
        if not self.samples:
            return np.array([])
        
        return np.array([sample.channels for sample in self.samples])