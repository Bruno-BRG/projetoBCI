"""
Implementação do repositório de modelos
"""

import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from ...domain.entities.model import Model, ModelStatus, ModelArchitecture
from ...domain.repositories.model_repository import ModelRepository
from ...domain.value_objects.training_types import ModelPerformance
from ..adapters.filesystem_adapter import LocalFileSystemAdapter

class FileSystemModelRepository(ModelRepository):
    """Repositório de modelos baseado em sistema de arquivos"""
    
    def __init__(self, models_directory: str, filesystem_adapter: LocalFileSystemAdapter):
        self.models_directory = Path(models_directory)
        self.filesystem = filesystem_adapter
        
        # Cria diretórios se não existem
        self.models_directory.mkdir(parents=True, exist_ok=True)
        (self.models_directory / "metadata").mkdir(parents=True, exist_ok=True)
        (self.models_directory / "files").mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Model) -> bool:
        """Salva modelo"""
        try:
            # Salva metadados
            metadata_path = f"metadata/{model.id}.json"
            metadata = {
                'id': model.id,
                'name': model.name,
                'architecture': model.architecture.value,
                'status': model.status.value,
                'file_path': model.file_path,
                'subject_ids': model.subject_ids,
                'training_metrics': self._serialize_metrics(model.training_metrics),
                'hyperparameters': model.hyperparameters,
                'metadata': model.metadata,
                'created_at': model.created_at.isoformat(),
                'updated_at': model.updated_at.isoformat(),
                'trained_at': model.trained_at.isoformat() if model.trained_at else None
            }
            
            return self.filesystem.write_json_file(metadata_path, metadata)
            
        except Exception as e:
            print(f"Erro ao salvar modelo: {str(e)}")
            return False
    
    def get_model_by_id(self, model_id: str) -> Optional[Model]:
        """Obtém modelo por ID"""
        try:
            metadata_path = f"metadata/{model_id}.json"
            
            if not self.filesystem.file_exists(metadata_path):
                return None
            
            metadata = self.filesystem.read_json_file(metadata_path)
            
            return Model(
                id=metadata['id'],
                name=metadata['name'],
                architecture=ModelArchitecture(metadata['architecture']),
                status=ModelStatus(metadata['status']),
                file_path=metadata.get('file_path'),
                subject_ids=metadata.get('subject_ids', []),
                training_metrics=self._deserialize_metrics(metadata.get('training_metrics')),
                hyperparameters=metadata.get('hyperparameters', {}),
                metadata=metadata.get('metadata', {}),
                created_at=datetime.fromisoformat(metadata['created_at']),
                updated_at=datetime.fromisoformat(metadata['updated_at']),
                trained_at=datetime.fromisoformat(metadata['trained_at']) if metadata.get('trained_at') else None
            )
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {str(e)}")
            return None
    
    def get_models_by_subject(self, subject_id: str) -> List[Model]:
        """Obtém modelos treinados para um sujeito"""
        models = []
        
        try:
            metadata_files = self.filesystem.list_files("metadata", "*.json")
            
            for metadata_file in metadata_files:
                metadata = self.filesystem.read_json_file(metadata_file)
                
                if subject_id in metadata.get('subject_ids', []):
                    model = self.get_model_by_id(metadata['id'])
                    if model:
                        models.append(model)
            
            return models
            
        except Exception as e:
            print(f"Erro ao buscar modelos por sujeito: {str(e)}")
            return []
    
    def get_models_by_status(self, status: ModelStatus) -> List[Model]:
        """Obtém modelos por status"""
        models = []
        
        try:
            metadata_files = self.filesystem.list_files("metadata", "*.json")
            
            for metadata_file in metadata_files:
                metadata = self.filesystem.read_json_file(metadata_file)
                
                if metadata.get('status') == status.value:
                    model = self.get_model_by_id(metadata['id'])
                    if model:
                        models.append(model)
            
            return models
            
        except Exception as e:
            print(f"Erro ao buscar modelos por status: {str(e)}")
            return []
    
    def get_models_by_architecture(self, architecture: ModelArchitecture) -> List[Model]:
        """Obtém modelos por arquitetura"""
        models = []
        
        try:
            metadata_files = self.filesystem.list_files("metadata", "*.json")
            
            for metadata_file in metadata_files:
                metadata = self.filesystem.read_json_file(metadata_file)
                
                if metadata.get('architecture') == architecture.value:
                    model = self.get_model_by_id(metadata['id'])
                    if model:
                        models.append(model)
            
            return models
            
        except Exception as e:
            print(f"Erro ao buscar modelos por arquitetura: {str(e)}")
            return []
    
    def get_best_model_for_subject(self, subject_id: str) -> Optional[Model]:
        """Obtém melhor modelo para um sujeito"""
        models = self.get_models_by_subject(subject_id)
        
        if not models:
            return None
        
        # Filtra apenas modelos treinados
        trained_models = [m for m in models if m.status in [ModelStatus.TRAINED, ModelStatus.VALIDATED, ModelStatus.DEPLOYED]]
        
        if not trained_models:
            return None
        
        # Retorna o modelo com maior acurácia
        best_model = max(trained_models, key=lambda m: m.get_accuracy() or 0.0)
        return best_model
    
    def update_model(self, model: Model) -> bool:
        """Atualiza modelo"""
        model.updated_at = datetime.now()
        return self.save_model(model)
    
    def delete_model(self, model_id: str) -> bool:
        """Remove modelo"""
        try:
            # Remove metadados
            metadata_path = f"metadata/{model_id}.json"
            metadata_deleted = self.filesystem.delete_file(metadata_path)
            
            # Remove arquivo do modelo se existir - prefere o file_path registrado
            file_deleted = True
            model_obj = self.get_model_by_id(model_id)
            if model_obj and model_obj.file_path:
                if self.filesystem.file_exists(model_obj.file_path):
                    file_deleted = self.filesystem.delete_file(model_obj.file_path)
            else:
                # Fallback para a convenção legacy -> usa .keras por padrão
                model_file_path = f"files/{model_id}.keras"
                if self.filesystem.file_exists(model_file_path):
                    file_deleted = self.filesystem.delete_file(model_file_path)
            
            return metadata_deleted and file_deleted
            
        except Exception as e:
            print(f"Erro ao deletar modelo: {str(e)}")
            return False
    
    def load_model_file(self, model_id: str) -> Optional[Any]:
        """Carrega arquivo do modelo para predição"""
        try:
            model = self.get_model_by_id(model_id)
            if not model or not model.file_path:
                return None
            
            # Tenta carregar modelo TensorFlow/Keras
            import tensorflow as tf

            # Primeiro tenta a convenção atual (.keras)
            model_path_default = f"files/{model_id}.keras"
            if self.filesystem.file_exists(model_path_default):
                return tf.keras.models.load_model(model_path_default)

            # Depois tenta o caminho registrado no metadata (model.file_path)
            if model.file_path and self.filesystem.file_exists(model.file_path):
                return tf.keras.models.load_model(model.file_path)

            return None
            
        except Exception as e:
            print(f"Erro ao carregar arquivo do modelo: {str(e)}")
            return None
    
    def save_model_file(self, model: Model, model_object: Any) -> bool:
        """Salva arquivo do modelo"""
        try:
            # Usa o file_path registrado se houver, senão define a convenção .keras
            model_path = model.file_path if model.file_path else f"files/{model.id}.keras"

            # Salva modelo TensorFlow/Keras
            if hasattr(model_object, 'save'):
                model_object.save(model_path)
                model.file_path = model_path
                return True

            return False
            
        except Exception as e:
            print(f"Erro ao salvar arquivo do modelo: {str(e)}")
            return False
    
    def get_model_performance_history(self, model_id: str) -> List[ModelPerformance]:
        """Obtém histórico de performance do modelo"""
        # Para implementação futura - histórico de métricas
        return []
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Compara performance de múltiplos modelos"""
        comparison = {}
        
        for model_id in model_ids:
            model = self.get_model_by_id(model_id)
            if model and model.training_metrics:
                comparison[model_id] = {
                    'accuracy': model.training_metrics.accuracy,
                    'precision': model.training_metrics.precision,
                    'recall': model.training_metrics.recall,
                    'f1_score': model.training_metrics.f1_score,
                    'loss': model.training_metrics.loss
                }
        
        return comparison
    
    def _serialize_metrics(self, metrics) -> Optional[Dict[str, Any]]:
        """Serializa métricas para JSON"""
        if not metrics:
            return None
        # Usa getattr para ser resiliente a diferentes tipos de objetos de métricas
        return {
            'accuracy': getattr(metrics, 'accuracy', None),
            'precision': getattr(metrics, 'precision', None),
            'recall': getattr(metrics, 'recall', None),
            'f1_score': getattr(metrics, 'f1_score', None),
            'loss': getattr(metrics, 'loss', None),
            'val_accuracy': getattr(metrics, 'val_accuracy', None),
            'val_loss': getattr(metrics, 'val_loss', None),
            'training_time': getattr(metrics, 'training_time', None),
            'epochs_trained': getattr(metrics, 'epochs_trained', None)
        }
    
    def _deserialize_metrics(self, data) -> Optional[Any]:
        """Deserializa métricas do JSON"""
        if not data:
            return None
        
        from ...domain.entities.model import TrainingMetrics
        
        return TrainingMetrics(
            accuracy=data['accuracy'],
            precision=data['precision'],
            recall=data['recall'],
            f1_score=data['f1_score'],
            loss=data['loss'],
            val_accuracy=data.get('val_accuracy'),
            val_loss=data.get('val_loss'),
            training_time=data.get('training_time'),
            epochs_trained=data.get('epochs_trained')
        )
