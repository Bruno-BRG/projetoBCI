"""
Training result entity for model training flows.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainingResult:
    """
    Domain representation of a completed training run.
    """

    model_path: str
    training_time_seconds: float
    final_accuracy: Optional[float] = None
    final_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None

    def validate(self) -> None:
        if not self.model_path or not self.model_path.strip():
            raise ValueError("Caminho do modelo treinado eh obrigatorio.")
        if self.training_time_seconds < 0:
            raise ValueError("Tempo de treinamento nao pode ser negativo.")

        for metric in (
            self.final_accuracy,
            self.final_loss,
            self.val_accuracy,
            self.val_loss,
        ):
            if metric is not None and metric < 0:
                raise ValueError("Metricas de treinamento nao podem ser negativas.")
