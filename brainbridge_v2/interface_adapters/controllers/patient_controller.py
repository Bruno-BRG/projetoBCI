"""
Controller that adapts presentation requests to patient use cases.
"""

from typing import Any, Dict, List

from brainbridge_v2.application.ports.patient_repository import PatientRepository
from brainbridge_v2.application.use_cases.patient_use_cases import (
    ListPatientsUseCase,
    RegisterPatientUseCase,
)
from brainbridge_v2.domain.entities.patient import Patient


class PatientController:
    """
    Presentation-facing controller for patient workflows.
    """

    def __init__(
        self,
        register_patient_use_case: RegisterPatientUseCase,
        list_patients_use_case: ListPatientsUseCase,
    ):
        self._register_patient_use_case = register_patient_use_case
        self._list_patients_use_case = list_patients_use_case

    @classmethod
    def from_repository(cls, repository: PatientRepository) -> "PatientController":
        return cls(
            register_patient_use_case=RegisterPatientUseCase(repository),
            list_patients_use_case=ListPatientsUseCase(repository),
        )

    def register_patient(self, patient_data: Dict[str, Any]) -> int:
        patient = Patient(
            name=str(patient_data.get("name", "")).strip(),
            age=int(patient_data.get("age", 0)),
            sex=str(patient_data.get("sex", "")).strip(),
            affected_hand=str(patient_data.get("affected_hand", "")).strip(),
            time_since_event=int(patient_data.get("time_since_event", 0)),
            notes=str(patient_data.get("notes", "")).strip(),
        )
        return self._register_patient_use_case.execute(patient)

    def list_patients(self) -> List[Dict[str, Any]]:
        patients = self._list_patients_use_case.execute()
        return [self._to_view_model(patient) for patient in patients]

    @staticmethod
    def _to_view_model(patient: Patient) -> Dict[str, Any]:
        return {
            "id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "sex": patient.sex,
            "affected_hand": patient.affected_hand,
            "time_since_event": patient.time_since_event,
            "created_at": patient.created_at or "",
            "notes": patient.notes,
        }

