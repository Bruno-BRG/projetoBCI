"""
Ports de saída da aplicação (Secondary Ports)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import numpy as np

class MLModelPort(ABC):
    """Port para modelos de Machine Learning"""
    
    @abstractmethod
    def create_model(self, architecture: str, hyperparameters: Dict[str, Any]) -> Any:
        """Cria modelo de ML"""
        pass
    
    @abstractmethod
    def train_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Treina modelo"""
        pass
    
    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Faz predições"""
        pass
    
    @abstractmethod
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Avalia modelo"""
        pass
    
    @abstractmethod
    def save_model(self, model: Any, file_path: str) -> bool:
        """Salva modelo em arquivo"""
        pass
    
    @abstractmethod
    def load_model(self, file_path: str) -> Any:
        """Carrega modelo de arquivo"""
        pass

class FileSystemPort(ABC):
    """Port para sistema de arquivos"""
    
    @abstractmethod
    def read_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Lê arquivo CSV"""
        pass
    
    @abstractmethod
    def write_csv_file(self, file_path: str, data: Dict[str, Any]) -> bool:
        """Escreve arquivo CSV"""
        pass
    
    @abstractmethod
    def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """Lista arquivos em diretório"""
        pass
    
    @abstractmethod
    def create_directory(self, directory_path: str) -> bool:
        """Cria diretório"""
        pass
    
    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        """Verifica se arquivo existe"""
        pass
    
    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """Remove arquivo"""
        pass

class LoggingPort(ABC):
    """Port para logging"""
    
    @abstractmethod
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log de informação"""
        pass
    
    @abstractmethod
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log de aviso"""
        pass
    
    @abstractmethod
    def log_error(self, message: str, error: Optional[Exception] = None, 
                  context: Optional[Dict[str, Any]] = None):
        """Log de erro"""
        pass
    
    @abstractmethod
    def log_debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log de debug"""
        pass

class NotificationPort(ABC):
    """Port para notificações"""
    
    @abstractmethod
    def notify_training_started(self, model_id: str, subject_id: str):
        """Notifica início de treinamento"""
        pass
    
    @abstractmethod
    def notify_training_completed(self, model_id: str, metrics: Dict[str, float]):
        """Notifica conclusão de treinamento"""
        pass
    
    @abstractmethod
    def notify_training_failed(self, model_id: str, error: str):
        """Notifica falha no treinamento"""
        pass
    
    @abstractmethod
    def notify_progress(self, process_id: str, progress: float, message: str):
        """Notifica progresso"""
        pass

class CachePort(ABC):
    """Port para cache"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Obtém valor do cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Define valor no cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove valor do cache"""
        pass
    
    @abstractmethod
    def clear(self):
        """Limpa cache"""
        pass

class ConfigurationPort(ABC):
    """Port para configurações"""
    
    @abstractmethod
    def get_config(self, key: str, default: Optional[Any] = None) -> Any:
        """Obtém configuração"""
        pass
    
    @abstractmethod
    def set_config(self, key: str, value: Any):
        """Define configuração"""
        pass
    
    @abstractmethod
    def get_all_configs(self) -> Dict[str, Any]:
        """Obtém todas as configurações"""
        pass
