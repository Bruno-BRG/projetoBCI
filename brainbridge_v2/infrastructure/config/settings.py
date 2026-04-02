"""
Configuração central de caminhos do BrainBridge v2.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
DATABASE_DIR = DATA_DIR / "database"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"

# Compatibilidade com nomes legados ainda aceitos localmente.
RECORDINGS_PATH = RECORDINGS_DIR
FOLDERS = {
    "recordings": RECORDINGS_DIR,
    "database": DATABASE_DIR,
    "models": MODELS_DIR,
    "logs": LOGS_DIR,
}

DATABASE_PATH = DATABASE_DIR / "bci_patients.db"


def ensure_folders_exist() -> None:
    """Garante que todas as pastas necessárias existem."""
    for folder_path in FOLDERS.values():
        folder_path.mkdir(parents=True, exist_ok=True)


def get_database_path() -> Path:
    return DATABASE_PATH


def get_recording_path(filename: str) -> Path:
    return RECORDINGS_DIR / filename


ensure_folders_exist()
