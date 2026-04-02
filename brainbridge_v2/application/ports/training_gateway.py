"""
Port definition for model training workflows.
"""

from typing import Callable, Optional, Protocol

from brainbridge_v2.domain.entities.training_result import TrainingResult


class TrainingGateway(Protocol):
    def train(
        self,
        csv_file_path: str,
        patient_id: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TrainingResult:
        ...
