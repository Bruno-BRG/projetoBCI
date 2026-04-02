"""
Metadata for discovered or loaded inference models.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelMetadata:
    """
    Domain representation of a model available to the application.
    """

    path: str
    name: str
    backend: str = "tensorflow"
    input_shape: Optional[Tuple[Optional[int], ...]] = None
    expected_time_steps: Optional[int] = None
    expected_channels: Optional[int] = None
    modified_at_epoch: Optional[float] = None

    def validate(self) -> None:
        if not self.path or not self.path.strip():
            raise ValueError("Caminho do modelo eh obrigatorio.")
        if not self.name or not self.name.strip():
            raise ValueError("Nome do modelo eh obrigatorio.")
        if not self.backend or not self.backend.strip():
            raise ValueError("Backend do modelo eh obrigatorio.")
        if self.expected_time_steps is not None and self.expected_time_steps <= 0:
            raise ValueError("Quantidade de timesteps esperada deve ser positiva.")
        if self.expected_channels is not None and self.expected_channels <= 0:
            raise ValueError("Quantidade de canais esperada deve ser positiva.")
        if self.modified_at_epoch is not None and self.modified_at_epoch < 0:
            raise ValueError("Data de modificacao do modelo nao pode ser negativa.")
