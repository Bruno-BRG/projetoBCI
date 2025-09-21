"""
Implementação do repositório de dados EEG
"""

import json
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from ...domain.entities.eeg_data import EEGData
from ...domain.entities.subject import Subject
from ...domain.repositories.eeg_repository import EEGDataRepository, SubjectRepository
from ...domain.value_objects.eeg_types import EEGSegment, MotorImageryClass
from ...domain.services.eeg_processing_service import EEGProcessingService
from ..adapters.filesystem_adapter import LocalFileSystemAdapter

class FileSystemEEGRepository(EEGDataRepository):
    """Repositório de dados EEG baseado em sistema de arquivos"""
    
    def __init__(self, data_directory: str, filesystem_adapter: LocalFileSystemAdapter):
        self.data_directory = Path(data_directory)
        self.filesystem = filesystem_adapter
        self.processing_service = EEGProcessingService()
        
        # Cria diretórios se não existem
        self.data_directory.mkdir(parents=True, exist_ok=True)
        (self.data_directory / "metadata").mkdir(parents=True, exist_ok=True)
    
    def save_eeg_data(self, eeg_data: EEGData) -> bool:
        """Salva dados EEG"""
        try:
            # Salva metadados
            metadata_path = f"metadata/{eeg_data.id}.json"
            metadata = {
                'id': eeg_data.id,
                'subject_id': eeg_data.subject_id,
                'session_id': eeg_data.session_id,
                'sample_rate': eeg_data.sample_rate,
                'channels': eeg_data.channels,
                'timestamp': eeg_data.timestamp.isoformat(),
                'metadata': eeg_data.metadata,
                'shape': eeg_data.raw_data.shape
            }
            
            success = self.filesystem.write_json_file(metadata_path, metadata)
            if not success:
                return False
            
            # Salva dados EEG em formato CSV
            csv_data = {
                'eeg_data': eeg_data.raw_data,
                'annotations': eeg_data.annotations,
                'header_info': {
                    'subject_id': eeg_data.subject_id,
                    'sample_rate': str(eeg_data.sample_rate),
                    'channels': str(eeg_data.channels),
                    'timestamp': eeg_data.timestamp.isoformat()
                }
            }
            
            csv_path = f"raw_data/{eeg_data.subject_id}/{eeg_data.id}.csv"
            return self.filesystem.write_csv_file(csv_path, csv_data)
            
        except Exception as e:
            print(f"Erro ao salvar dados EEG: {str(e)}")
            return False
    
    def get_eeg_data_by_id(self, data_id: str) -> Optional[EEGData]:
        """Obtém dados EEG por ID"""
        try:
            metadata_path = f"metadata/{data_id}.json"
            
            if not self.filesystem.file_exists(metadata_path):
                return None
            
            metadata = self.filesystem.read_json_file(metadata_path)
            
            # Carrega dados CSV
            csv_path = f"raw_data/{metadata['subject_id']}/{data_id}.csv"
            if not self.filesystem.file_exists(csv_path):
                return None
            
            csv_data = self.filesystem.read_csv_file(csv_path)
            
            return EEGData(
                id=metadata['id'],
                subject_id=metadata['subject_id'],
                session_id=metadata.get('session_id'),
                raw_data=csv_data['eeg_data'],
                annotations=csv_data['annotations'],
                sample_rate=metadata['sample_rate'],
                channels=metadata['channels'],
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                metadata=metadata.get('metadata', {})
            )
            
        except Exception as e:
            print(f"Erro ao carregar dados EEG: {str(e)}")
            return None
    
    def get_eeg_data_by_subject(self, subject_id: str) -> List[EEGData]:
        """Obtém todos os dados EEG de um sujeito"""
        eeg_data_list = []
        
        try:
            # Lista arquivos de metadados
            metadata_files = self.filesystem.list_files("metadata", "*.json")
            for metadata_file in metadata_files:
                metadata = self.filesystem.read_json_file(metadata_file)

                if metadata.get('subject_id') == subject_id:
                    eeg_data = self.get_eeg_data_by_id(metadata['id'])
                    if eeg_data:
                        eeg_data_list.append(eeg_data)

            # Se nenhum metadado foi encontrado, provavelmente temos apenas CSVs brutos
            # (formato resources/eeg_data/SXXX/*.csv). Nesse caso, procura arquivos CSV
            # em raw_data/<subject_id> e em <subject_id> e carrega diretamente.
            if not eeg_data_list:
                # Tentativa 1: raw_data/<subject_id>
                csv_files = self.filesystem.list_files(f"raw_data/{subject_id}", "*.csv")
                for csv_file in csv_files:
                    eeg = self.load_from_csv(csv_file, subject_id)
                    if eeg:
                        eeg_data_list.append(eeg)

                # Tentativa 2: diretório direto do sujeito under data root (ex: resources/eeg_data/S001)
                csv_files = self.filesystem.list_files(f"{subject_id}", "*.csv")
                for csv_file in csv_files:
                    eeg = self.load_from_csv(csv_file, subject_id)
                    if eeg:
                        eeg_data_list.append(eeg)
            
            return eeg_data_list
            
        except Exception as e:
            print(f"Erro ao carregar dados do sujeito {subject_id}: {str(e)}")
            return []
    
    def get_all_subjects(self) -> List[str]:
        """Obtém lista de todos os sujeitos"""
        subjects = set()
        
        try:
            # Primeira tentativa: estrutura com subdiretório 'raw_data/<subject>' (compatibilidade)
            raw_data_dirs = self.filesystem.list_directories("raw_data")
            if raw_data_dirs:
                for dir_path in raw_data_dirs:
                    subject_id = Path(dir_path).name
                    subjects.add(subject_id)
            else:
                # Segunda tentativa: diretórios direto no data root (ex: resources/eeg_data/S001/...)
                root_dirs = self.filesystem.list_directories(".")
                for dir_path in root_dirs:
                    # Ignora metadados e outros diretórios conhecidos
                    name = Path(dir_path).name
                    if name.lower() in {"metadata", "subjects", "models", "logs", "results", "temp"}:
                        continue
                    subjects.add(name)
            
            return list(subjects)
            
        except Exception as e:
            print(f"Erro ao listar sujeitos: {str(e)}")
            return []
    
    def delete_eeg_data(self, data_id: str) -> bool:
        """Remove dados EEG"""
        try:
            # Remove metadados
            metadata_path = f"metadata/{data_id}.json"
            metadata_deleted = self.filesystem.delete_file(metadata_path)
            
            # Busca e remove arquivo CSV
            metadata = self.filesystem.read_json_file(metadata_path)
            if metadata:
                csv_path = f"raw_data/{metadata['subject_id']}/{data_id}.csv"
                csv_deleted = self.filesystem.delete_file(csv_path)
                return metadata_deleted and csv_deleted
            
            return metadata_deleted
            
        except Exception as e:
            print(f"Erro ao deletar dados EEG: {str(e)}")
            return False
    
    def load_from_csv(self, file_path: str, subject_id: str) -> Optional[EEGData]:
        """Carrega dados EEG de arquivo CSV"""
        try:
            csv_data = self.filesystem.read_csv_file(file_path)
            
            eeg_data = EEGData.create(
                subject_id=subject_id,
                raw_data=csv_data['eeg_data'],
                annotations=csv_data['annotations'],
                sample_rate=125,  # Padrão do sistema
                channels=16,      # Padrão do sistema
                metadata={'source_file': file_path}
            )
            
            return eeg_data
            
        except Exception as e:
            print(f"Erro ao carregar CSV {file_path}: {str(e)}")
            return None
    
    def extract_segments(self, eeg_data: EEGData) -> List[EEGSegment]:
        """Extrai segmentos de dados EEG baseados em marcadores"""
        return self.processing_service.extract_motor_imagery_segments(eeg_data)
    
    def get_data_statistics(self, subject_id: str) -> Dict[str, Any]:
        """Obtém estatísticas dos dados de um sujeito"""
        try:
            eeg_data_list = self.get_eeg_data_by_subject(subject_id)
            
            if not eeg_data_list:
                return {}
            
            total_samples = sum(len(data.raw_data) for data in eeg_data_list)
            total_duration = sum(data.get_duration_seconds() for data in eeg_data_list)
            
            # Conta marcadores
            marker_counts = {'T1': 0, 'T2': 0, 'T0': 0}
            for data in eeg_data_list:
                for annotation in data.annotations:
                    if annotation in marker_counts:
                        marker_counts[annotation] += 1
            
            return {
                'subject_id': subject_id,
                'total_sessions': len(eeg_data_list),
                'total_samples': total_samples,
                'total_duration_seconds': total_duration,
                'marker_counts': marker_counts,
                'average_session_duration': total_duration / len(eeg_data_list) if eeg_data_list else 0
            }
            
        except Exception as e:
            print(f"Erro ao calcular estatísticas: {str(e)}")
            return {}

class FileSystemSubjectRepository(SubjectRepository):
    """Repositório de sujeitos baseado em sistema de arquivos"""
    
    def __init__(self, data_directory: str, filesystem_adapter: LocalFileSystemAdapter):
        self.data_directory = Path(data_directory)
        self.filesystem = filesystem_adapter
        
        # Cria diretório de sujeitos
        (self.data_directory / "subjects").mkdir(parents=True, exist_ok=True)
    
    def save_subject(self, subject: Subject) -> bool:
        """Salva sujeito"""
        try:
            subject_data = {
                'id': subject.id,
                'name': subject.name,
                'age': subject.age,
                'gender': subject.gender,
                'sessions': subject.sessions,
                'metadata': subject.metadata,
                'created_at': subject.created_at.isoformat(),
                'updated_at': subject.updated_at.isoformat()
            }
            
            file_path = f"subjects/{subject.id}.json"
            return self.filesystem.write_json_file(file_path, subject_data)
            
        except Exception as e:
            print(f"Erro ao salvar sujeito: {str(e)}")
            return False
    
    def get_subject_by_id(self, subject_id: str) -> Optional[Subject]:
        """Obtém sujeito por ID"""
        try:
            file_path = f"subjects/{subject_id}.json"
            
            if not self.filesystem.file_exists(file_path):
                return None
            
            data = self.filesystem.read_json_file(file_path)
            
            return Subject(
                id=data['id'],
                name=data['name'],
                age=data.get('age'),
                gender=data.get('gender'),
                sessions=data.get('sessions', []),
                metadata=data.get('metadata', {}),
                created_at=datetime.fromisoformat(data['created_at']),
                updated_at=datetime.fromisoformat(data['updated_at'])
            )
            
        except Exception as e:
            print(f"Erro ao carregar sujeito: {str(e)}")
            return None
    
    def get_all_subjects(self) -> List[Subject]:
        """Obtém todos os sujeitos"""
        subjects = []
        
        try:
            subject_files = self.filesystem.list_files("subjects", "*.json")
            
            for file_path in subject_files:
                data = self.filesystem.read_json_file(file_path)
                
                subject = Subject(
                    id=data['id'],
                    name=data['name'],
                    age=data.get('age'),
                    gender=data.get('gender'),
                    sessions=data.get('sessions', []),
                    metadata=data.get('metadata', {}),
                    created_at=datetime.fromisoformat(data['created_at']),
                    updated_at=datetime.fromisoformat(data['updated_at'])
                )
                subjects.append(subject)
            
            return subjects
            
        except Exception as e:
            print(f"Erro ao carregar sujeitos: {str(e)}")
            return []
    
    def update_subject(self, subject: Subject) -> bool:
        """Atualiza sujeito"""
        subject.updated_at = datetime.now()
        return self.save_subject(subject)
    
    def delete_subject(self, subject_id: str) -> bool:
        """Remove sujeito"""
        try:
            file_path = f"subjects/{subject_id}.json"
            return self.filesystem.delete_file(file_path)
        except Exception as e:
            print(f"Erro ao deletar sujeito: {str(e)}")
            return False
    
    def find_subjects_by_criteria(self, criteria: Dict[str, Any]) -> List[Subject]:
        """Busca sujeitos por critérios"""
        all_subjects = self.get_all_subjects()
        filtered_subjects = []
        
        for subject in all_subjects:
            match = True
            
            for key, value in criteria.items():
                if key == 'name' and value.lower() not in subject.name.lower():
                    match = False
                    break
                elif key == 'age' and subject.age != value:
                    match = False
                    break
                elif key == 'gender' and subject.gender != value:
                    match = False
                    break
                elif key in subject.metadata and subject.metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered_subjects.append(subject)
        
        return filtered_subjects
