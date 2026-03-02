"""
Port definition for patient persistence.
"""

from typing import List, Protocol
from brainbridge_v2.domain.entities.patient import Patient


class PatientRepository(Protocol):
    def add(self, patient: Patient) -> int:
        """
        Persists a patient and returns the generated identifier.
        """
        ...

    def list_all(self) -> List[Patient]:
        """
        Returns all patients.
        """
        ...

