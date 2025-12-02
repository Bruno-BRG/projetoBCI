"""
Gestão de pacientes
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import uuid


@dataclass
class Patient:
    """Representa um paciente no sistema"""
    id: str
    name: str
    age: int
    gender: str
    created_at: datetime
    notes: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def create_new(cls, name: str, age: int, gender: str, notes: Optional[str] = None) -> 'Patient':
        """Cria um novo paciente com ID único"""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            age=age,
            gender=gender,
            created_at=datetime.now(),
            notes=notes
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'created_at': self.created_at.isoformat(),
            'notes': self.notes,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Patient':
        """Cria instância a partir de dicionário"""
        return cls(
            id=data['id'],
            name=data['name'],
            age=data['age'],
            gender=data['gender'],
            created_at=datetime.fromisoformat(data['created_at']),
            notes=data.get('notes'),
            metadata=data.get('metadata', {})
        )