import tempfile
import os
from pathlib import Path

from brainbridge_v2.infrastructure.ml.training_gateway_adapter import (
    ModelTrainingGatewayAdapter,
)


class FakeTrainResult:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.training_time = 4.0
        self.final_accuracy = 0.77
        self.final_loss = 0.3
        self.val_accuracy = 0.71
        self.val_loss = 0.4


class FakeTrainerModule:
    def __init__(self):
        self.calls = []

    def train_from_csvs(self, csv_files, model_name, base_model_path=None):
        self.calls.append((csv_files, model_name, base_model_path))
        return FakeTrainResult(f"C:/models/{model_name}.keras")


def test_training_gateway_adapter_runs_real_training_when_tensorflow_is_available(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("dummy", encoding="utf-8")
        trainer_module = FakeTrainerModule()
        monkeypatch.setitem(__import__("sys").modules, "tensorflow", object())

        adapter = ModelTrainingGatewayAdapter(
            trainer_module=trainer_module,
            models_dir=tmp_path,
        )
        progress_messages = []
        result = adapter.train(
            str(csv_path),
            5,
            progress_callback=progress_messages.append,
        )

        assert trainer_module.calls == [([str(csv_path)], "patient_5", None)]
        assert result.model_path.endswith("patient_5.keras")
        assert progress_messages[:3] == [
            "Preparando dados...",
            "Iniciando treinamento real (Keras)...",
            "Nenhum modelo base encontrado. Treinando modelo novo.",
        ]


def test_training_gateway_adapter_continues_existing_patient_model(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("dummy", encoding="utf-8")
        patient_model = tmp_path / "patient_5.keras"
        patient_model.write_text("existing patient model", encoding="utf-8")
        trainer_module = FakeTrainerModule()
        monkeypatch.setitem(__import__("sys").modules, "tensorflow", object())

        adapter = ModelTrainingGatewayAdapter(
            trainer_module=trainer_module,
            models_dir=tmp_path,
        )
        progress_messages = []
        adapter.train(
            str(csv_path),
            5,
            progress_callback=progress_messages.append,
        )

        assert trainer_module.calls == [([str(csv_path)], "patient_5", str(patient_model))]
        assert progress_messages[2] == "Continuando treino do modelo do paciente: patient_5.keras"


def test_training_gateway_adapter_uses_latest_non_patient_model_as_initial_base(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("dummy", encoding="utf-8")
        older_model = tmp_path / "generalized_old.keras"
        newer_model = tmp_path / "generalized_new.keras"
        older_model.write_text("older model", encoding="utf-8")
        newer_model.write_text("newer model", encoding="utf-8")
        os.utime(older_model, (100, 100))
        os.utime(newer_model, (200, 200))
        trainer_module = FakeTrainerModule()
        monkeypatch.setitem(__import__("sys").modules, "tensorflow", object())

        adapter = ModelTrainingGatewayAdapter(
            trainer_module=trainer_module,
            models_dir=tmp_path,
        )
        adapter.train(str(csv_path), 7)

        assert trainer_module.calls == [([str(csv_path)], "patient_7", str(newer_model))]
