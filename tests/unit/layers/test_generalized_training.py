import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

from brainbridge_v2.infrastructure.ml import trainer


class FakeHistory:
    history = {
        "accuracy": [0.8],
        "loss": [0.2],
        "val_accuracy": [0.7],
        "val_loss": [0.3],
    }


class FakeGeneralizedModel:
    def __init__(self):
        self.fit_calls = []
        self.saved_path = None
        self.input_shape = (None, 4, 3)

    def fit(self, *args, **kwargs):
        self.fit_calls.append((args, kwargs))
        return FakeHistory()

    def evaluate(self, X, y, verbose=0):
        return 0.31, 0.69

    def predict(self, X, verbose=0):
        probabilities = np.zeros((len(X), 2), dtype=np.float32)
        probabilities[:, 0] = 0.8
        probabilities[:, 1] = 0.2
        return probabilities

    def save(self, path):
        self.saved_path = Path(path)
        self.saved_path.write_text("fake model", encoding="utf-8")


def test_infer_group_id_from_path_prefers_patient_or_subject_token():
    assert trainer._infer_group_id_from_path("data/P002_Murilo/session.csv") == "P002"
    assert trainer._infer_group_id_from_path("data/other/S058R04.csv") == "S058"
    assert trainer._infer_group_id_from_path("data/unknown/session.csv") == "unknown"


def test_collect_windowed_dataset_preserves_groups(monkeypatch):
    def fake_load(path):
        return np.ones((250, 16), dtype=np.float32), [""] * 250

    def fake_windows(data, markers, **kwargs):
        return np.ones((2, 4, 3), dtype=np.float32), np.array([0, 1], dtype=np.int32)

    monkeypatch.setattr(trainer, "_load_openbci_csv", fake_load)
    monkeypatch.setattr(trainer, "_create_windows_ht", fake_windows)

    X, y, groups, summaries = trainer.load_generalized_windowed_dataset(
        ["root/P001/a.csv", "root/P002/b.csv"],
        window_size=4,
        group_resolver=lambda path: Path(path).parent.name,
    )

    assert X.shape == (4, 4, 3)
    assert y.tolist() == [0, 1, 0, 1]
    assert groups.tolist() == ["P001", "P001", "P002", "P002"]
    assert {summary.group_id: summary.windows for summary in summaries} == {
        "P001": 2,
        "P002": 2,
    }


def test_train_generalized_from_csvs_uses_group_holdout(monkeypatch):
    X = np.arange(16 * 4 * 3, dtype=np.float32).reshape(16, 4, 3)
    y = np.array([0, 1] * 8, dtype=np.int32)
    groups = np.array(["P001"] * 4 + ["P002"] * 4 + ["P003"] * 4 + ["P004"] * 4)
    summaries = [
        trainer.SubjectWindowSummary("P001", ["a.csv"], 4, {0: 2, 1: 2}),
        trainer.SubjectWindowSummary("P002", ["b.csv"], 4, {0: 2, 1: 2}),
        trainer.SubjectWindowSummary("P003", ["c.csv"], 4, {0: 2, 1: 2}),
        trainer.SubjectWindowSummary("P004", ["d.csv"], 4, {0: 2, 1: 2}),
    ]
    fake_model = FakeGeneralizedModel()

    monkeypatch.setitem(sys.modules, "tensorflow", object())
    monkeypatch.setattr(
        trainer,
        "_collect_windowed_dataset",
        lambda *args, **kwargs: (X, y, groups, summaries),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setattr(trainer, "MODELS_DIR", Path(temp_dir))

        result = trainer.train_generalized_from_csvs(
            ["a.csv", "b.csv", "c.csv", "d.csv"],
            window_size=4,
            epochs=1,
            batch_size=2,
            model_name="dev_general",
            model_builder=lambda input_shape, num_classes: fake_model,
            validation_size=0.25,
        )

        assert result.model_path.endswith("dev_general.keras")
        assert result.val_accuracy == 0.69
        assert result.heldout_groups
        assert result.train_groups
        assert set(result.heldout_groups).isdisjoint(set(result.train_groups))
        assert result.group_summaries == summaries
        assert fake_model.fit_calls
        assert fake_model.saved_path.exists()


def test_train_generalized_requires_multiple_groups(monkeypatch):
    X = np.ones((4, 4, 3), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    groups = np.array(["P001"] * 4)

    monkeypatch.setitem(sys.modules, "tensorflow", object())
    monkeypatch.setattr(
        trainer,
        "_collect_windowed_dataset",
        lambda *args, **kwargs: (X, y, groups, []),
    )

    with pytest.raises(ValueError, match="dois grupos"):
        trainer.train_generalized_from_csvs(["a.csv"], window_size=4)


def test_train_from_csvs_continues_from_base_model(monkeypatch):
    X = np.ones((4, 4, 3), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    fake_model = FakeGeneralizedModel()
    loaded_paths = []

    monkeypatch.setitem(sys.modules, "tensorflow", object())
    monkeypatch.setattr(
        trainer,
        "_load_openbci_csv",
        lambda path: (np.ones((250, 16), dtype=np.float32), [""] * 250),
    )
    monkeypatch.setattr(trainer, "_create_windows_ht", lambda *args, **kwargs: (X, y))

    with tempfile.TemporaryDirectory() as temp_dir:
        base_model = Path(temp_dir) / "patient_9.keras"
        base_model.write_text("existing model", encoding="utf-8")
        monkeypatch.setattr(trainer, "MODELS_DIR", Path(temp_dir))

        result = trainer.train_from_csvs(
            ["session.csv"],
            window_size=4,
            epochs=1,
            batch_size=2,
            model_name="patient_9",
            base_model_path=str(base_model),
            model_loader=lambda path: loaded_paths.append(path) or fake_model,
        )

        assert loaded_paths == [str(base_model)]
        assert fake_model.fit_calls
        assert fake_model.saved_path == Path(result.model_path)
        assert result.model_path.endswith("patient_9.keras")


def test_train_from_csvs_rejects_incompatible_base_model(monkeypatch):
    X = np.ones((4, 4, 3), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    fake_model = FakeGeneralizedModel()
    fake_model.input_shape = (None, 250, 16)

    monkeypatch.setitem(sys.modules, "tensorflow", object())
    monkeypatch.setattr(
        trainer,
        "_load_openbci_csv",
        lambda path: (np.ones((250, 16), dtype=np.float32), [""] * 250),
    )
    monkeypatch.setattr(trainer, "_create_windows_ht", lambda *args, **kwargs: (X, y))

    with tempfile.TemporaryDirectory() as temp_dir:
        base_model = Path(temp_dir) / "generalized.keras"
        base_model.write_text("base model", encoding="utf-8")

        with pytest.raises(ValueError, match="incompativel"):
            trainer.train_from_csvs(
                ["session.csv"],
                window_size=4,
                base_model_path=str(base_model),
                model_loader=lambda path: fake_model,
            )
