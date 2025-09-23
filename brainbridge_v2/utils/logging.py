"""
Sistema de logging centralizado
"""
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "data/logs") -> logging.Logger:
    """
    Configura o sistema de logging
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Diretório para salvar logs
        
    Returns:
        Logger configurado
    """
    # Criar diretório de logs se não existir
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"brainbridge_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Configurar formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configurar logger principal
    logger = logging.getLogger('brainbridge')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# Logger global
logger = setup_logging()