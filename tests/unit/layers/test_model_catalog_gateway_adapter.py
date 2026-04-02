import os
import tempfile
from pathlib import Path

from brainbridge_v2.infrastructure.ml.model_catalog_gateway_adapter import (
    FileSystemModelCatalogGatewayAdapter,
)


def test_model_catalog_gateway_adapter_lists_supported_files_sorted_by_mtime():
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        older_model = tmp_path / "older.h5"
        latest_model = tmp_path / "latest.keras"
        ignored_file = tmp_path / "notes.txt"

        older_model.write_text("older", encoding="utf-8")
        latest_model.write_text("latest", encoding="utf-8")
        ignored_file.write_text("ignore", encoding="utf-8")

        os.utime(older_model, (10, 10))
        os.utime(latest_model, (20, 20))

        adapter = FileSystemModelCatalogGatewayAdapter(search_roots=[tmp_path])
        models = adapter.list_models()

        assert [model.name for model in models] == ["latest.keras", "older.h5"]
        assert models[0].path.endswith("latest.keras")
