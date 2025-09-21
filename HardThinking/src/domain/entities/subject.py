"""
Entidade Subject - Representa um sujeito de pesquisa
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

@dataclass
class Subject:
    """Entidade que representa um sujeito de pesquisa"""
    
    id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    sessions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    @classmethod
    def create(cls, 
               name: str,
               age: Optional[int] = None,
               gender: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> 'Subject':
        """Factory method para criar instância de Subject"""
        
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            age=age,
            gender=gender,
            sessions=[],
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def add_session(self, session_id: str):
        """Adiciona uma sessão ao sujeito"""
        if session_id not in self.sessions:
            self.sessions.append(session_id)
            self.updated_at = datetime.now()
    
    def remove_session(self, session_id: str):
        """Remove uma sessão do sujeito"""
        if session_id in self.sessions:
            self.sessions.remove(session_id)
            self.updated_at = datetime.now()
    
    def get_session_count(self) -> int:
        """Retorna número de sessões"""
        return len(self.sessions)
    
    def update_metadata(self, key: str, value: Any):
        """Atualiza metadados"""
        self.metadata[key] = value
        self.updated_at = datetime.now()
    
    def validate(self) -> bool:
        """Valida se os dados do sujeito estão consistentes"""
        if not self.name or not self.name.strip():
            return False
        if self.age is not None and (self.age < 0 or self.age > 150):
            return False
        return True
