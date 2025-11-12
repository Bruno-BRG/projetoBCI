"""
Adaptador para carregamento e predição com modelos TensorFlow/Keras

Compatibilidade com o esperado pela GUI e por legado HardThinking.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np


class TensorFlowMLAdapter:
    """
    Adaptador para carregar modelos .keras/.h5 e realizar predições.
    Compatível com a interface esperada pelo GUI de streaming.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o adaptador TensorFlow.
        
        Args:
            config: Dicionário opcional de configurações (não usado atualmente)
        """
        self.config = config or {}
        self.model = None

    def load_model(self, model_path: str):
        """
        Carrega um modelo Keras (.keras ou .h5).
        
        Args:
            model_path: Caminho para o arquivo do modelo
            
        Returns:
            O modelo Keras carregado
            
        Raises:
            FileNotFoundError: Se o arquivo não existir
            ImportError: Se TensorFlow não estiver disponível
        """
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")
        
        try:
            from tensorflow.keras.models import load_model
        except ImportError as e:
            raise ImportError(
                "TensorFlow não está disponível. Instale com: pip install tensorflow"
            ) from e
        
        self.model = load_model(str(path))
        return self.model

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Realiza predição em um lote de dados.
        
        Args:
            data: Array de entrada, shape (batch, timesteps, channels)
            
        Returns:
            Predictions array
            
        Raises:
            RuntimeError: Se modelo não foi carregado
        """
        if self.model is None:
            raise RuntimeError("Modelo não foi carregado. Chame load_model() primeiro.")
        
        return self.model.predict(data, verbose=0)

    def predict_on_window(self, window: np.ndarray) -> Dict[str, Any]:
        """
        Prediz classe para uma janela EEG (compatibilidade com predictor.py).
        
        Args:
            window: Array shape (timesteps, channels)
            
        Returns:
            dict com 'probs' e 'label' ('left' ou 'right')
        """
        if self.model is None:
            raise RuntimeError("Modelo não foi carregado. Chame load_model() primeiro.")
        
        if window.ndim != 2:
            raise ValueError(f"Esperado window shape (timesteps, channels), got {window.shape}")
        
        # Adiciona batch dimension
        x = window.astype('float32')[None, ...]  # (1, T, C)
        probs = self.model.predict(x, verbose=0)[0]
        
        idx = int(np.argmax(probs))
        label = 'left' if idx == 0 else 'right'
        
        return {
            'probs': probs.tolist(),
            'label': label,
            'confidence': float(probs[idx])
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo carregado."""
        if self.model is None:
            return {'loaded': False}
        
        info = {'loaded': True}
        
        try:
            if hasattr(self.model, 'input_shape'):
                info['input_shape'] = tuple(self.model.input_shape)
            if hasattr(self.model, 'output_shape'):
                info['output_shape'] = tuple(self.model.output_shape)
            if hasattr(self.model, 'summary'):
                info['name'] = getattr(self.model, 'name', 'unknown')
        except Exception:
            pass
        
        return info
