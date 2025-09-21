"""
Adapter para logging
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from ...application.ports.secondary_ports import LoggingPort

class PythonLoggingAdapter(LoggingPort):
    """Adapter para logging usando Python logging module"""
    
    def __init__(self, log_directory: str = "logs", log_level: str = "INFO"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Configura logger
        self.logger = logging.getLogger("HardThinking")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Configura formatador
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler para arquivo
        log_file = self.log_directory / f"hardthinking_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log de informação"""
        full_message = self._format_message(message, context)
        self.logger.info(full_message)
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log de aviso"""
        full_message = self._format_message(message, context)
        self.logger.warning(full_message)
    
    def log_error(self, message: str, error: Optional[Exception] = None, 
                  context: Optional[Dict[str, Any]] = None):
        """Log de erro"""
        full_message = self._format_message(message, context)
        if error:
            full_message += f" | Error: {str(error)}"
            self.logger.error(full_message, exc_info=error)
        else:
            self.logger.error(full_message)
    
    def log_debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log de debug"""
        full_message = self._format_message(message, context)
        self.logger.debug(full_message)
    
    def _format_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Formata mensagem com contexto"""
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            return f"{message} | Context: {context_str}"
        return message
    
    def log_training_event(self, event_type: str, model_id: str, details: Dict[str, Any]):
        """Log específico para eventos de treinamento"""
        context = {"model_id": model_id, "event": event_type}
        context.update(details)
        self.log_info(f"Training event: {event_type}", context)
    
    def log_prediction_event(self, model_id: str, prediction_details: Dict[str, Any]):
        """Log específico para eventos de predição"""
        context = {"model_id": model_id}
        context.update(prediction_details)
        self.log_info("Prediction made", context)
    
    def log_performance_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Log de métricas de performance"""
        context = {"model_id": model_id}
        context.update(metrics)
        self.log_info("Performance metrics", context)

class FileLoggingAdapter(LoggingPort):
    """Adapter simples para logging em arquivo"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_directory / f"hardthinking_{datetime.now().strftime('%Y%m%d')}.log"
    
    def _write_log(self, level: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Escreve log no arquivo"""
        timestamp = datetime.now().isoformat()
        full_message = f"[{timestamp}] {level}: {message}"
        
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            full_message += f" | Context: {context_str}"
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(full_message + "\n")
        except Exception as e:
            print(f"Erro ao escrever log: {e}")
    
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log de informação"""
        self._write_log("INFO", message, context)
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log de aviso"""
        self._write_log("WARNING", message, context)
    
    def log_error(self, message: str, error: Optional[Exception] = None, 
                  context: Optional[Dict[str, Any]] = None):
        """Log de erro"""
        full_message = message
        if error:
            full_message += f" | Error: {str(error)}"
        self._write_log("ERROR", full_message, context)
    
    def log_debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log de debug"""
        self._write_log("DEBUG", message, context)
