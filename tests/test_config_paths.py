import os
from pathlib import Path


def test_data_directories_exist():
    from brainbridge_v2.infrastructure.config.settings import DATA_DIR, RECORDINGS_DIR, DATABASE_DIR, MODELS_DIR, LOGS_DIR

    assert DATA_DIR.exists() and DATA_DIR.is_dir()
    assert RECORDINGS_DIR.exists() and RECORDINGS_DIR.is_dir()
    assert DATABASE_DIR.exists() and DATABASE_DIR.is_dir()
    assert MODELS_DIR.exists() and MODELS_DIR.is_dir()
    assert LOGS_DIR.exists() and LOGS_DIR.is_dir()
