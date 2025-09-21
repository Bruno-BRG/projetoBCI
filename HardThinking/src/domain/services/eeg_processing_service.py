"""
Serviço de processamento de dados EEG
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ..entities.eeg_data import EEGData
from ..value_objects.eeg_types import EEGSegment, TimeWindow, MotorImageryClass, DataQuality
from scipy import signal
from sklearn.preprocessing import StandardScaler

class EEGProcessingService:
    """Serviço para processamento de dados EEG"""
    
    def __init__(self, sample_rate: int = 125, window_size: int = 250, overlap: int = 125):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
    
    def extract_motor_imagery_segments(self, eeg_data: EEGData) -> List[EEGSegment]:
        """Extrai segmentos de imagética motora baseado nos marcadores T1, T2, T0"""
        segments = []
        
        # Encontra índices dos marcadores
        marker_indices = self._find_marker_indices(eeg_data.annotations)
        
        # Extrai segmentos T1->T0 (esquerda)
        left_segments = self._extract_segments_between_markers(
            eeg_data.raw_data, 
            marker_indices.get('T1', []), 
            marker_indices.get('T0', []),
            MotorImageryClass.LEFT
        )
        segments.extend(left_segments)
        
        # Extrai segmentos T2->T0 (direita)
        right_segments = self._extract_segments_between_markers(
            eeg_data.raw_data,
            marker_indices.get('T2', []),
            marker_indices.get('T0', []),
            MotorImageryClass.RIGHT
        )
        segments.extend(right_segments)
        
        return segments
    
    def _find_marker_indices(self, annotations: List[str]) -> Dict[str, List[int]]:
        """Encontra índices dos marcadores nas anotações"""
        marker_indices = {'T0': [], 'T1': [], 'T2': []}
        
        for i, annotation in enumerate(annotations):
            if annotation in marker_indices:
                marker_indices[annotation].append(i)
        
        return marker_indices
    
    def _extract_segments_between_markers(self, 
                                        raw_data: np.ndarray,
                                        start_markers: List[int],
                                        end_markers: List[int],
                                        label: MotorImageryClass) -> List[EEGSegment]:
        """Extrai segmentos entre marcadores de início e fim"""
        segments = []
        
        for start_idx in start_markers:
            # Encontra próximo marcador de fim
            end_idx = None
            for end_marker_idx in end_markers:
                if end_marker_idx > start_idx:
                    end_idx = end_marker_idx
                    break
            
            if end_idx is not None:
                # Extrai dados entre marcadores
                segment_data = raw_data[start_idx:end_idx]
                
                # Divide em janelas se longo o suficiente
                if len(segment_data) >= self.window_size:
                    windowed_segments = self._create_sliding_windows(segment_data, label)
                    segments.extend(windowed_segments)
        
        return segments
    
    def _create_sliding_windows(self, data: np.ndarray, label: MotorImageryClass) -> List[EEGSegment]:
        """Cria janelas deslizantes dos dados"""
        segments = []
        
        for start_idx in range(0, len(data) - self.window_size + 1, self.overlap):
            window_data = data[start_idx:start_idx + self.window_size]
            segment = EEGSegment(data=window_data, label=label)
            segments.append(segment)
        
        return segments
    
    def preprocess_segment(self, segment: EEGSegment) -> EEGSegment:
        """Pré-processa um segmento EEG (filtros, normalização)"""
        # Aplicar filtro passa-banda (8-30 Hz)
        filtered_data = self._apply_bandpass_filter(segment.data)
        
        # Normalizar dados
        normalized_data = self._normalize_data(filtered_data)
        
        return EEGSegment(
            data=normalized_data,
            label=segment.label,
            confidence=segment.confidence
        )
    
    def _apply_bandpass_filter(self, data: np.ndarray, 
                              low_freq: float = 8.0, 
                              high_freq: float = 30.0) -> np.ndarray:
        """Aplica filtro passa-banda"""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = np.zeros_like(data)
        
        # Aplica filtro em cada canal
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normaliza dados usando StandardScaler"""
        scaler = StandardScaler()
        normalized_data = np.zeros_like(data)
        
        # Normaliza cada canal independentemente
        for i in range(data.shape[1]):
            normalized_data[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
        
        return normalized_data
    
    def calculate_data_quality(self, eeg_data: EEGData) -> DataQuality:
        """Calcula qualidade dos dados EEG"""
        # Calcula SNR (aproximado)
        signal_power = np.mean(np.var(eeg_data.raw_data, axis=0))
        noise_power = np.mean(np.var(np.diff(eeg_data.raw_data, axis=0), axis=0))
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        # Calcula nível de artefatos (baseado em outliers)
        outlier_threshold = 3 * np.std(eeg_data.raw_data)
        outliers = np.abs(eeg_data.raw_data) > outlier_threshold
        artifact_level = np.mean(outliers)
        
        # Calcula completude (dados não-nulos)
        valid_data = ~np.isnan(eeg_data.raw_data)
        completeness = np.mean(valid_data)
        
        return DataQuality(
            signal_noise_ratio=snr,
            artifact_level=artifact_level,
            completeness=completeness
        )
    
    def extract_features(self, segment: EEGSegment) -> np.ndarray:
        """Extrai features do segmento EEG"""
        features = []
        
        for channel in range(segment.channels):
            channel_data = segment.get_channel(channel)
            
            # Features estatísticas
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.var(channel_data),
                np.min(channel_data),
                np.max(channel_data)
            ])
            
            # Features espectrais
            fft = np.fft.fft(channel_data)
            power_spectrum = np.abs(fft) ** 2
            
            # Bandas de frequência
            alpha_power = np.mean(power_spectrum[8:13])  # 8-13 Hz
            beta_power = np.mean(power_spectrum[13:30])  # 13-30 Hz
            
            features.extend([alpha_power, beta_power])
        
        return np.array(features)
