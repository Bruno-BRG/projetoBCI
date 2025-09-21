"""
Configurações centralizadas do sistema HardThinking
Refatoração do sistema BrainBridge com arquitetura hexagonal
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class DataConfiguration:
    """Configurações relacionadas aos dados EEG"""
    sample_rate: int = 125  # Hz
    channels: int = 16      # Canais EEG (0-15)
    window_size: int = 250  # 2 segundos a 125Hz
    overlap: int = 125      # 50% sobreposição
    classes: List[str] = None
    markers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.classes is None:
            self.classes = ['left', 'right']
        if self.markers is None:
            self.markers = {
                'left_start': 'T1',
                'right_start': 'T2', 
                'end': 'T0'
            }

@dataclass
class ModelConfiguration:
    """Configurações do modelo de ML"""
    architecture: str = 'CNN_1D'
    conv_filters: List[int] = None
    conv_kernels: List[int] = None
    dense_units: int = 512
    dropout_rates: List[float] = None
    activation: str = 'relu'
    output_activation: str = 'softmax'
    optimizer: str = 'adam'
    loss: str = 'categorical_crossentropy'
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.conv_filters is None:
            self.conv_filters = [64, 64, 128]
        if self.conv_kernels is None:
            self.conv_kernels = [3, 3, 3]
        if self.dropout_rates is None:
            self.dropout_rates = [0.25, 0.25, 0.5]
        if self.metrics is None:
            self.metrics = ['accuracy']

@dataclass
class TrainingConfiguration:
    """Configurações de treinamento"""
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class PredictionConfiguration:
    """Configurações de predição"""
    min_confidence: float = 0.65
    default_action: str = "rest"
    smoothing_window: int = 3

@dataclass
class DirectoryConfiguration:
    """Configurações de diretórios"""
    project_root: Path
    data_dir: Path
    models_dir: Path
    logs_dir: Path
    results_dir: Path
    temp_dir: Path
    
    def __post_init__(self):
        # Cria diretórios se não existirem
        for attr_name, directory in self.__dict__.items():
            if isinstance(directory, Path):
                directory.mkdir(parents=True, exist_ok=True)

@dataclass
class CLIConfiguration:
    """Configurações da interface CLI"""
    show_progress: bool = True
    verbose: bool = True
    colors_enabled: bool = True
    banner_enabled: bool = True

class SystemConfiguration:
    """Configuração centralizada do sistema"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # project_root should be the repository/project folder (two levels above src)
            # original code used three levels which pointed to the parent of the project when
            # the repository is nested under another folder. Use two levels to locate the
            # HardThinking project root correctly.
            project_root = Path(__file__).parent.parent
        
        self.data = DataConfiguration()
        self.model = ModelConfiguration()
        self.training = TrainingConfiguration()
        self.prediction = PredictionConfiguration()
        self.cli = CLIConfiguration()
        
        # Configuração de diretórios
        # Define diretórios. Preferência: se existir, usa resources/eeg_data (caminho relativo ao projeto),
        # caso contrário usa o diretório 'data' para compatibilidade com instalações legadas.
        preferred_data_dir = project_root / "resources" / "eeg_data"
        default_data_dir = project_root / "data"

        data_dir = preferred_data_dir if preferred_data_dir.exists() else default_data_dir

        self.directories = DirectoryConfiguration(
            project_root=project_root,
            data_dir=data_dir,
            models_dir=project_root / "models",
            logs_dir=project_root / "logs",
            results_dir=project_root / "results",
            temp_dir=project_root / "temp"
        )
    
    def validate_data_directory(self, data_path: Optional[str] = None) -> bool:
        """Valida se o diretório de dados existe e tem estrutura adequada"""
        if data_path:
            data_dir = Path(data_path)
        else:
            data_dir = self.directories.data_dir
        
        if not data_dir.exists():
            return False
        
        # Verifica se há pelo menos um subdiretório com arquivos CSV
        for item in data_dir.iterdir():
            if item.is_dir():
                csv_files = list(item.glob("*.csv"))
                if csv_files:
                    return True
        
        return False
    
    def get_system_info(self) -> Dict[str, str]:
        """Retorna informações do sistema"""
        import platform
        import sys
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "project_root": str(self.directories.project_root),
            "data_dir": str(self.directories.data_dir),
            "models_dir": str(self.directories.models_dir)
        }

# Instância global da configuração (singleton pattern)
_config_instance = None

def get_config() -> SystemConfiguration:
    """Retorna a instância singleton da configuração"""
    global _config_instance
    if _config_instance is None:
        _config_instance = SystemConfiguration()
    return _config_instance

def set_config(config: SystemConfiguration):
    """Define uma nova instância de configuração (para testes)"""
    global _config_instance
    _config_instance = config

# Compatibilidade com código legado
DATA_CONFIG = get_config().data.__dict__
MODEL_CONFIG = get_config().model.__dict__
TRAINING_CONFIG = get_config().training.__dict__
PREDICTION_CONFIG = get_config().prediction.__dict__
CLI_CONFIG = get_config().cli.__dict__
DIRECTORIES = {
    'data': str(get_config().directories.data_dir),
    'models': str(get_config().directories.models_dir),
    'logs': str(get_config().directories.logs_dir),
    'results': str(get_config().directories.results_dir),
    'temp': str(get_config().directories.temp_dir)
}

# Funções de compatibilidade
def validate_data_directory(data_path: Optional[str] = None) -> bool:
    return get_config().validate_data_directory(data_path)

def get_system_info() -> Dict[str, str]:
    return get_config().get_system_info()
