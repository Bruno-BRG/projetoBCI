"""
Patient-related use cases.
"""

from typing import List

from brainbridge_v2.application.ports.patient_repository import PatientRepository
from brainbridge_v2.domain.entities.patient import Patient


class RegisterPatientUseCase:
    """
    Registers a new patient after domain validation.
    """

    def __init__(self, repository: PatientRepository):
        self._repository = repository

    def execute(self, patient: Patient) -> int:
        patient.validate()
        return self._repository.add(patient)


class ListPatientsUseCase:
    """
    Returns all registered patients.
    """

    def __init__(self, repository: PatientRepository):
        self._repository = repository

    def execute(self) -> List[Patient]:
        return self._repository.list_all()

