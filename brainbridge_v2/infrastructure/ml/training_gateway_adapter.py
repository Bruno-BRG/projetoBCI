"""
Training gateway backed by the existing ML training pipeline.
"""

from pathlib import Path
from typing import Callable, Optional

from brainbridge_v2.domain.entities.training_result import TrainingResult
from brainbridge_v2.infrastructure.config.settings import MODELS_DIR
from brainbridge_v2.infrastructure.ml import trainer as ml_trainer


class ModelTrainingGatewayAdapter:
    """
    Runs model training using the existing TensorFlow/stub pipeline.
    """

    def __init__(
        self,
        trainer_module=ml_trainer,
        models_dir: Optional[Path] = None,
    ):
        self._trainer_module = trainer_module
        self._models_dir = models_dir or MODELS_DIR
        self._models_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        csv_file_path: str,
        patient_id: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TrainingResult:
        csv_path = Path(csv_file_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV de treinamento nao encontrado: {csv_file_path}")

        emit = progress_callback or (lambda _: None)

        emit("Preparando dados...")
        try:
            import tensorflow  # noqa: F401

            use_real_training = True
        except Exception:
            use_real_training = False

        if use_real_training:
            emit("Iniciando treinamento real (Keras)...")
            result = self._trainer_module.train_from_csvs(
                [str(csv_path)],
                model_name=f"patient_{patient_id}",
            )
            emit("Modelo salvo com sucesso.")
            training_result = TrainingResult(
                model_path=result.model_path,
                training_time_seconds=float(result.training_time),
                final_accuracy=result.final_accuracy,
                final_loss=result.final_loss,
                val_accuracy=result.val_accuracy,
                val_loss=result.val_loss,
            )
            training_result.validate()
            return training_result

        emit("TensorFlow nao disponivel. Executando modo simulado...")
        for step in (
            "Carregando CSV e validando formato...",
            "Extraindo janelas...",
            "(stub) Treinando modelo...",
            "Salvando modelo (stub)...",
        ):
            emit(step)

        model_path = self._models_dir / f"patient_{patient_id}.keras"
        model_path.write_text("stub model file", encoding="utf-8")
        emit("Modelo salvo com sucesso.")

        training_result = TrainingResult(
            model_path=str(model_path),
            training_time_seconds=0.0,
        )
        training_result.validate()
        return training_result
