"""
Pré-processamento de sinais EEG
"""
import numpy as np
from typing import Tuple, Optional

from .filters import FilterBank, create_standard_filters


class EEGPreprocessor:
    """Classe para pré-processamento de sinais EEG"""
    
    def __init__(self, fs: float, enable_filtering: bool = True):
        self.fs = fs
        self.enable_filtering = enable_filtering
        
        if enable_filtering:
            self.filter_bank = create_standard_filters(fs)
        else:
            self.filter_bank = None
    
    def preprocess(self, data: np.ndarray, apply_car: bool = True, 
                   normalize: bool = True) -> np.ndarray:
        """
        Aplica pré-processamento completo aos dados
        
        Args:
            data: Array 2D (samples x channels)
            apply_car: Se deve aplicar Common Average Reference
            normalize: Se deve normalizar os dados
            
        Returns:
            Dados pré-processados
        """
        processed_data = data.copy()
        
        # 1. Filtragem
        if self.enable_filtering and self.filter_bank:
            processed_data = self.filter_bank.filter_signal(processed_data)
        
        # 2. Common Average Reference (CAR)
        if apply_car:
            processed_data = self.apply_car(processed_data)
        
        # 3. Normalização
        if normalize:
            processed_data = self.normalize_data(processed_data)
        
        return processed_data
    
    def apply_car(self, data: np.ndarray) -> np.ndarray:
        """
        Aplica Common Average Reference
        
        Args:
            data: Array 2D (samples x channels)
            
        Returns:
            Dados com CAR aplicado
        """
        # Calcular média de todos os canais
        average_ref = np.mean(data, axis=1, keepdims=True)
        
        # Subtrair a referência média de cada canal
        return data - average_ref
    
    def normalize_data(self, data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normaliza os dados
        
        Args:
            data: Array 2D (samples x channels)
            method: 'zscore', 'minmax', 'robust'
            
        Returns:
            Dados normalizados
        """
        if method == 'zscore':
            # Z-score normalization
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            # Evitar divisão por zero
            std[std == 0] = 1
            return (data - mean) / std
            
        elif method == 'minmax':
            # Min-max normalization
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            range_val = max_val - min_val
            # Evitar divisão por zero
            range_val[range_val == 0] = 1
            return (data - min_val) / range_val
            
        elif method == 'robust':
            # Robust normalization usando mediana e MAD
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            # Evitar divisão por zero
            mad[mad == 0] = 1
            return (data - median) / mad
            
        else:
            raise ValueError(f"Método de normalização inválido: {method}")
    
    def segment_data(self, data: np.ndarray, window_size: int, 
                     overlap: float = 0.5) -> np.ndarray:
        """
        Segmenta dados em janelas
        
        Args:
            data: Array 2D (samples x channels)
            window_size: Tamanho da janela em amostras
            overlap: Sobreposição entre janelas (0-1)
            
        Returns:
            Array 3D (windows x samples x channels)
        """
        step = int(window_size * (1 - overlap))
        n_samples, n_channels = data.shape
        
        # Calcular número de janelas
        n_windows = (n_samples - window_size) // step + 1
        
        if n_windows <= 0:
            return np.array([])
        
        # Criar array de saída
        segments = np.zeros((n_windows, window_size, n_channels))
        
        for i in range(n_windows):
            start = i * step
            end = start + window_size
            segments[i] = data[start:end]
        
        return segments
    
    def extract_epochs(self, data: np.ndarray, events: list, 
                       epoch_start: float = -0.5, epoch_end: float = 2.0) -> Tuple[np.ndarray, list]:
        """
        Extrai épocas baseadas em eventos
        
        Args:
            data: Array 2D (samples x channels)
            events: Lista de (timestamp, marker)
            epoch_start: Início da época em segundos (relativo ao evento)
            epoch_end: Fim da época em segundos
            
        Returns:
            Tuple (epochs, labels)
        """
        epochs = []
        labels = []
        
        start_samples = int(epoch_start * self.fs)
        end_samples = int(epoch_end * self.fs)
        epoch_length = end_samples - start_samples
        
        for event_sample, marker in events:
            epoch_start_idx = event_sample + start_samples
            epoch_end_idx = event_sample + end_samples
            
            # Verificar se a época está dentro dos dados
            if epoch_start_idx >= 0 and epoch_end_idx < data.shape[0]:
                epoch = data[epoch_start_idx:epoch_end_idx]
                epochs.append(epoch)
                labels.append(marker)
        
        if epochs:
            return np.array(epochs), labels
        else:
            return np.array([]), []
    
    def remove_artifacts(self, data: np.ndarray, threshold: float = 100.0) -> np.ndarray:
        """
        Remove artefatos simples baseado em limiar
        
        Args:
            data: Array 2D (samples x channels)
            threshold: Limiar de amplitude
            
        Returns:
            Dados com artefatos removidos
        """
        # Marcar amostras com amplitude muito alta
        artifact_mask = np.abs(data) > threshold
        
        # Interpolar valores artefactuais
        cleaned_data = data.copy()
        
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            artifact_samples = artifact_mask[:, ch]
            
            if np.any(artifact_samples):
                # Interpolação linear simples
                valid_indices = np.where(~artifact_samples)[0]
                artifact_indices = np.where(artifact_samples)[0]
                
                if len(valid_indices) > 1:
                    cleaned_data[artifact_indices, ch] = np.interp(
                        artifact_indices, valid_indices, channel_data[valid_indices]
                    )
        
        return cleaned_data