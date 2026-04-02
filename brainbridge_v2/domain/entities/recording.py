"""
Recording entity for the domain layer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Recording:
    """
    Core recording entity independent from infrastructure details.
    """

    patient_id: int
    filename: str
    task_type: str
    notes: str = ""
    id: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[int] = None

    def validate(self) -> None:
        if self.patient_id <= 0:
            raise ValueError("Paciente da gravacao deve ser valido.")
        if not self.filename or not self.filename.strip():
            raise ValueError("Arquivo da gravacao eh obrigatorio.")
        if not self.task_type or not self.task_type.strip():
            raise ValueError("Tipo de tarefa da gravacao eh obrigatorio.")
        if self.duration is not None and self.duration < 0:
            raise ValueError("Duracao da gravacao nao pode ser negativa.")
