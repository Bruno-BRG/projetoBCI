"""
Value Objects relacionados a dados EEG
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from enum import Enum

class MotorImageryClass(Enum):
    """Classes de imagética motora"""
    LEFT = "left"
    RIGHT = "right"
    REST = "rest"

@dataclass(frozen=True)
class EEGSegment:
    """Value object que representa um segmento de dados EEG"""
    
    data: np.ndarray  # Shape: (window_size, channels)
    label: MotorImageryClass
    confidence: float = 1.0
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence deve estar entre 0.0 e 1.0")
        if len(self.data.shape) != 2:
            raise ValueError("Data deve ter shape (window_size, channels)")
    
    @property
    def window_size(self) -> int:
        """Retorna tamanho da janela"""
        return self.data.shape[0]
    
    @property
    def channels(self) -> int:
        """Retorna número de canais"""
        return self.data.shape[1]
    
    def get_channel(self, channel_index: int) -> np.ndarray:
        """Retorna dados de um canal específico"""
        return self.data[:, channel_index]

@dataclass(frozen=True)
class TimeWindow:
    """Value object que representa uma janela temporal"""
    
    start_sample: int
    end_sample: int
    sample_rate: int
    
    def __post_init__(self):
        if self.start_sample < 0:
            raise ValueError("Start sample deve ser >= 0")
        if self.end_sample <= self.start_sample:
            raise ValueError("End sample deve ser > start sample")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate deve ser > 0")
    
    @property
    def duration_samples(self) -> int:
        """Retorna duração em amostras"""
        return self.end_sample - self.start_sample
    
    @property
    def duration_seconds(self) -> float:
        """Retorna duração em segundos"""
        return self.duration_samples / self.sample_rate
    
    @classmethod
    def from_seconds(cls, start_seconds: float, end_seconds: float, sample_rate: int) -> 'TimeWindow':
        """Cria TimeWindow a partir de tempos em segundos"""
        start_sample = int(start_seconds * sample_rate)
        end_sample = int(end_seconds * sample_rate)
        return cls(start_sample, end_sample, sample_rate)

@dataclass(frozen=True)
class Prediction:
    """Value object que representa uma predição"""
    
    predicted_class: MotorImageryClass
    confidence: float
    probabilities: Tuple[float, ...]  # Probabilidades para cada classe
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence deve estar entre 0.0 e 1.0")
        if not all(0.0 <= p <= 1.0 for p in self.probabilities):
            raise ValueError("Todas as probabilidades devem estar entre 0.0 e 1.0")
        if abs(sum(self.probabilities) - 1.0) > 1e-6:
            raise ValueError("Probabilidades devem somar 1.0")
    
    @property
    def is_confident(self) -> bool:
        """Verifica se a predição tem confiança suficiente"""
        return self.confidence >= 0.65  # Threshold padrão
    
    def is_confident_with_threshold(self, threshold: float) -> bool:
        """Verifica se a predição tem confiança acima do threshold"""
        return self.confidence >= threshold

@dataclass(frozen=True)
class DataQuality:
    """Value object que representa qualidade dos dados"""
    
    signal_noise_ratio: float
    artifact_level: float
    completeness: float  # Porcentagem de dados válidos
    
    def __post_init__(self):
        if not 0.0 <= self.completeness <= 1.0:
            raise ValueError("Completeness deve estar entre 0.0 e 1.0")
        if self.artifact_level < 0.0:
            raise ValueError("Artifact level deve ser >= 0.0")
    
    @property
    def is_good_quality(self) -> bool:
        """Verifica se os dados têm qualidade adequada"""
        return (self.signal_noise_ratio > 10.0 and 
                self.artifact_level < 0.3 and 
                self.completeness > 0.8)
    
    @property
    def quality_score(self) -> float:
        """Retorna score de qualidade (0.0 a 1.0)"""
        snr_score = min(self.signal_noise_ratio / 20.0, 1.0)
        artifact_score = max(1.0 - self.artifact_level, 0.0)
        return (snr_score + artifact_score + self.completeness) / 3.0
