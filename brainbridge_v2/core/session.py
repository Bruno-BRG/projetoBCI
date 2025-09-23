"""
Sessões de gravação/streaming
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from .patient import Patient
from .eeg_data import EEGSample, EEGSession


class SessionState(Enum):
    """Estados possíveis de uma sessão"""
    IDLE = "idle"
    RECORDING = "recording" 
    STREAMING = "streaming"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class SessionConfig:
    """Configuração de uma sessão"""
    session_type: str = "motor_imagery"
    duration_seconds: Optional[int] = None
    trials_per_class: int = 10
    rest_duration: int = 3
    trial_duration: int = 4
    auto_save: bool = True


class Session:
    """Gerencia uma sessão de coleta/streaming de dados"""
    
    def __init__(self, patient: Patient, config: SessionConfig = None):
        self.patient = patient
        self.config = config or SessionConfig()
        self.state = SessionState.IDLE
        self.eeg_session = EEGSession(patient.id, self.config.session_type)
        self.trial_count = 0
        self.current_class = None
        self.start_time: Optional[datetime] = None
        
    def start(self):
        """Inicia a sessão"""
        if self.state != SessionState.IDLE:
            raise ValueError(f"Cannot start session in state {self.state}")
        
        self.state = SessionState.RECORDING
        self.start_time = datetime.now()
        
    def pause(self):
        """Pausa a sessão"""
        if self.state != SessionState.RECORDING:
            raise ValueError(f"Cannot pause session in state {self.state}")
        
        self.state = SessionState.PAUSED
        
    def resume(self):
        """Resume a sessão"""
        if self.state != SessionState.PAUSED:
            raise ValueError(f"Cannot resume session in state {self.state}")
        
        self.state = SessionState.RECORDING
        
    def stop(self):
        """Para a sessão"""
        if self.state in [SessionState.STOPPED, SessionState.IDLE]:
            return
        
        self.state = SessionState.STOPPED
        
        if self.config.auto_save:
            self.save()
    
    def add_sample(self, sample: EEGSample):
        """Adiciona uma amostra à sessão"""
        if self.state == SessionState.RECORDING:
            self.eeg_session.add_sample(sample)
    
    def set_marker(self, marker: str):
        """Define o marcador atual"""
        # Implementar lógica de marcadores
        pass
    
    def next_trial(self):
        """Avança para o próximo trial"""
        self.trial_count += 1
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Salva a sessão em arquivo"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"data/recordings/{self.patient.id}_{timestamp}.csv"
        
        # Implementar salvamento em CSV
        # Por enquanto, apenas retorna o caminho
        return filepath
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas da sessão"""
        return {
            'patient_id': self.patient.id,
            'duration': self.eeg_session.get_duration(),
            'samples_count': len(self.eeg_session.samples),
            'trials_completed': self.trial_count,
            'state': self.state.value,
            'session_type': self.config.session_type
        }