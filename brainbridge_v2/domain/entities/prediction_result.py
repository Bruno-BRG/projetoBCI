"""
Prediction result entity for runtime inference.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PredictionResult:
    """
    Domain representation of a model prediction.
    """

    predicted_index: int
    confidence: float
    probabilities: Tuple[float, ...]

    def validate(self) -> None:
        if self.predicted_index < 0:
            raise ValueError("Indice da predicao nao pode ser negativo.")
        if not self.probabilities:
            raise ValueError("Predicao deve conter probabilidades.")
        if self.predicted_index >= len(self.probabilities):
            raise ValueError("Indice da predicao esta fora do intervalo de probabilidades.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confianca da predicao deve ficar entre 0 e 1.")
        for probability in self.probabilities:
            if not 0.0 <= probability <= 1.0:
                raise ValueError("Probabilidades da predicao devem ficar entre 0 e 1.")

    @property
    def left_probability(self) -> float:
        if len(self.probabilities) < 1:
            return 0.0
        return float(self.probabilities[0])

    @property
    def right_probability(self) -> float:
        if len(self.probabilities) < 2:
            return 0.0
        return float(self.probabilities[1])
