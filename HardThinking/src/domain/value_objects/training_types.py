"""
Value Objects relacionados a treinamento de modelos
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class TrainingStrategy(Enum):
    """Estratégias de treinamento"""
    SINGLE_SUBJECT = "single_subject"
    CROSS_VALIDATION = "cross_validation"
    LEAVE_ONE_OUT = "leave_one_out"
    MULTI_SUBJECT = "multi_subject"

@dataclass(frozen=True)
class TrainingConfiguration:
    """Configuração de treinamento"""
    
    strategy: TrainingStrategy
    batch_size: int
    epochs: int
    learning_rate: float
    validation_split: float
    early_stopping_patience: int
    
    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("Batch size deve ser > 0")
        if self.epochs <= 0:
            raise ValueError("Epochs deve ser > 0")
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError("Learning rate deve estar entre 0.0 e 1.0")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("Validation split deve estar entre 0.0 e 1.0")
        if self.early_stopping_patience <= 0:
            raise ValueError("Early stopping patience deve ser > 0")

@dataclass(frozen=True)
class ModelPerformance:
    """Performance de um modelo"""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    
    def __post_init__(self):
        for metric in [self.accuracy, self.precision, self.recall, self.f1_score]:
            if not 0.0 <= metric <= 1.0:
                raise ValueError(f"Métrica deve estar entre 0.0 e 1.0: {metric}")
        if self.loss < 0.0:
            raise ValueError("Loss deve ser >= 0.0")
    
    @property
    def is_good_performance(self) -> bool:
        """Verifica se a performance é boa"""
        return (self.accuracy >= 0.7 and 
                self.f1_score >= 0.65 and 
                self.precision >= 0.6 and 
                self.recall >= 0.6)
    
    @property
    def overall_score(self) -> float:
        """Score geral da performance"""
        return (self.accuracy + self.precision + self.recall + self.f1_score) / 4.0

@dataclass(frozen=True)
class ValidationResult:
    """Resultado de validação"""
    
    train_performance: ModelPerformance
    validation_performance: ModelPerformance
    test_performance: Optional[ModelPerformance] = None
    cross_validation_scores: Optional[List[float]] = None
    
    @property
    def is_overfitting(self) -> bool:
        """Verifica se há overfitting"""
        accuracy_diff = self.train_performance.accuracy - self.validation_performance.accuracy
        return accuracy_diff > 0.15  # Threshold de 15%
    
    @property
    def generalization_gap(self) -> float:
        """Calcula gap de generalização"""
        return self.train_performance.accuracy - self.validation_performance.accuracy

@dataclass(frozen=True)
class HyperParameters:
    """Hiperparâmetros do modelo"""
    
    parameters: Dict[str, float]
    
    def __post_init__(self):
        if not self.parameters:
            raise ValueError("Parameters não pode estar vazio")
    
    def get_parameter(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """Obtém um parâmetro específico"""
        return self.parameters.get(name, default)
    
    def update_parameter(self, name: str, value: float) -> 'HyperParameters':
        """Retorna nova instância com parâmetro atualizado"""
        new_params = self.parameters.copy()
        new_params[name] = value
        return HyperParameters(new_params)
