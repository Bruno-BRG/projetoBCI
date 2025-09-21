"""
Caso de uso: Treinar modelo de imagética motora
"""

from typing import List, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass

# Temporarily use relative imports that will work once the full structure is in place
from ...domain.entities.eeg_data import EEGData
from ...domain.entities.model import Model, ModelArchitecture, ModelStatus, TrainingMetrics
from ...domain.entities.subject import Subject
from ...domain.value_objects.eeg_types import EEGSegment, MotorImageryClass
from ...domain.value_objects.training_types import TrainingStrategy, ModelPerformance, ValidationResult
from ...domain.services.eeg_processing_service import EEGProcessingService
from ...domain.services.model_validation_service import ModelValidationService
from ..ports.secondary_ports import MLModelPort, LoggingPort, NotificationPort

@dataclass
class TrainModelRequest:
    """Request para treinamento de modelo"""
    subject_ids: List[str]
    strategy: TrainingStrategy
    model_architecture: ModelArchitecture
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2
    cv_folds: int = 5

@dataclass
class TrainModelResponse:
    """Response do treinamento de modelo"""
    model: Model
    validation_result: ValidationResult
    success: bool
    error_message: Optional[str] = None

class TrainModelUseCase:
    """Caso de uso para treinamento de modelos"""
    
    def __init__(self,
                 eeg_repository,
                 model_repository,
                 subject_repository,
                 ml_port: MLModelPort,
                 logging_port: LoggingPort,
                 notification_port: NotificationPort):
        self.eeg_repository = eeg_repository
        self.model_repository = model_repository
        self.subject_repository = subject_repository
        self.ml_port = ml_port
        self.logging_port = logging_port
        self.notification_port = notification_port
        self.eeg_service = EEGProcessingService()
        self.validation_service = ModelValidationService()
    
    def execute(self, request: TrainModelRequest) -> TrainModelResponse:
        """Executa o caso de uso de treinamento"""
        try:
            # Log início do treinamento
            self.logging_port.log_info(
                f"Iniciando treinamento - Estratégia: {request.strategy.value}, Sujeitos: {request.subject_ids}"
            )
            
            # Cria modelo
            model = Model.create(
                name=f"model_{request.strategy.value}_{len(request.subject_ids)}_subjects",
                architecture=request.model_architecture,
                hyperparameters=request.hyperparameters or {},
                metadata={"training_strategy": request.strategy.value}
            )
            
            # Adiciona sujeitos ao modelo
            for subject_id in request.subject_ids:
                model.add_subject(subject_id)
            
            # Notifica início
            self.notification_port.notify_training_started(model.id, ",".join(request.subject_ids))
            
            # Carrega e processa dados
            all_segments = self._load_and_process_data(request.subject_ids)
            
            if not all_segments:
                raise ValueError("Nenhum segmento de dados encontrado")
            
            # Executa treinamento baseado na estratégia
            validation_result = self._execute_training_strategy(
                model, all_segments, request
            )
            
            # Salva modelo
            self.model_repository.save_model(model)
            
            # Notifica conclusão
            metrics = {
                "accuracy": validation_result.validation_performance.accuracy,
                "f1_score": validation_result.validation_performance.f1_score
            }
            self.notification_port.notify_training_completed(model.id, metrics)
            
            self.logging_port.log_info(
                f"Treinamento concluído - Modelo: {model.id}, Acurácia: {validation_result.validation_performance.accuracy:.3f}"
            )
            
            return TrainModelResponse(
                model=model,
                validation_result=validation_result,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Erro durante treinamento: {str(e)}"
            self.logging_port.log_error(error_msg, e)
            
            if 'model' in locals():
                model.set_error(error_msg)
                self.model_repository.save_model(model)
                self.notification_port.notify_training_failed(model.id, error_msg)
            
            return TrainModelResponse(
                model=model if 'model' in locals() else None,
                validation_result=None,
                success=False,
                error_message=error_msg
            )
    
    def _load_and_process_data(self, subject_ids: List[str]) -> List[EEGSegment]:
        """Carrega e processa dados dos sujeitos"""
        all_segments = []
        
        for subject_id in subject_ids:
            # Obtém dados EEG do sujeito
            eeg_data_list = self.eeg_repository.get_eeg_data_by_subject(subject_id)
            
            for eeg_data in eeg_data_list:
                # Extrai segmentos de imagética motora
                segments = self.eeg_service.extract_motor_imagery_segments(eeg_data)
                
                # Pré-processa segmentos
                for segment in segments:
                    processed_segment = self.eeg_service.preprocess_segment(segment)
                    all_segments.append(processed_segment)
        
        return all_segments
    
    def _execute_training_strategy(self, 
                                  model: Model,
                                  segments: List[EEGSegment],
                                  request: TrainModelRequest) -> ValidationResult:
        """Executa estratégia de treinamento específica"""
        
        # Prepara dados para ML
        X, y = self._prepare_ml_data(segments)
        
        # Cria modelo de ML
        ml_model = self.ml_port.create_model(
            architecture=request.model_architecture.value,
            hyperparameters=request.hyperparameters or {}
        )
        
        if request.strategy == TrainingStrategy.SINGLE_SUBJECT:
            return self._train_single_subject(ml_model, X, y, model, request.validation_split)
        
        elif request.strategy == TrainingStrategy.CROSS_VALIDATION:
            return self._train_cross_validation(ml_model, X, y, model, request.cv_folds)
        
        elif request.strategy == TrainingStrategy.LEAVE_ONE_OUT:
            return self._train_leave_one_out(ml_model, X, y, model)
        
        elif request.strategy == TrainingStrategy.MULTI_SUBJECT:
            return self._train_multi_subject(ml_model, X, y, model, request.validation_split)
        
        else:
            raise ValueError(f"Estratégia não suportada: {request.strategy}")
    
    def _prepare_ml_data(self, segments: List[EEGSegment]) -> tuple:
        """Prepara dados para treinamento de ML"""
        # Mapeamento de classes
        class_map = {
            MotorImageryClass.LEFT: 0,
            MotorImageryClass.RIGHT: 1
        }

        # Esperamos que cada segmento tenha forma (window_size, channels).
        # Empacotamos como um array 3D (n_samples, window_size, channels) para compatibilidade
        # com modelos Keras que recebem entrada (batch, timesteps, channels).
        X = np.stack([segment.data for segment in segments]).astype('float32')
        y = np.array([class_map[segment.label] for segment in segments], dtype='int32')

        return X, y
    
    def _train_single_subject(self, ml_model, X, y, model: Model, validation_split: float) -> ValidationResult:
        """Treina modelo para um único sujeito"""
        validation_result = self.validation_service.validate_single_subject(
            ml_model, X, y, test_size=validation_split
        )
        
        # Treina modelo final com todos os dados
        training_result = self.ml_port.train_model(ml_model, X, y)

        # Constrói TrainingMetrics compatível com a entidade Model
        history = training_result.get('history', {})
        epochs_trained = 0
        try:
            # history é um dict com listas por métrica
            epochs_trained = len(next(iter(history.values())))
        except Exception:
            epochs_trained = None

        training_metrics = TrainingMetrics(
            accuracy=training_result.get('final_accuracy', validation_result.validation_performance.accuracy),
            loss=training_result.get('final_loss', 1.0 - training_result.get('final_accuracy', 0.0)),
            val_accuracy=training_result.get('val_accuracy', validation_result.validation_performance.accuracy),
            val_loss=training_result.get('val_loss', None),
            training_time=training_result.get('training_time', None),
            epochs_trained=epochs_trained
        )

        # Salva modelo treinado (usar extensão moderna .keras ao invés de legacy .h5)
        model_path = f"files/{model.id}.keras"
        self.ml_port.save_model(ml_model, model_path)
        model.set_training_completed(
            metrics=training_metrics,
            file_path=model_path
        )

        return validation_result
    
    def _train_cross_validation(self, ml_model, X, y, model: Model, cv_folds: int) -> ValidationResult:
        """Treina modelo com validação cruzada"""
        validation_result = self.validation_service.cross_validate_model(
            ml_model, X, y, cv_folds=cv_folds
        )
        
        # Treina modelo final
        training_result = self.ml_port.train_model(ml_model, X, y)

        # Constrói TrainingMetrics
        history = training_result.get('history', {})
        epochs_trained = 0
        try:
            epochs_trained = len(next(iter(history.values())))
        except Exception:
            epochs_trained = None

        training_metrics = TrainingMetrics(
            accuracy=training_result.get('final_accuracy', validation_result.validation_performance.accuracy),
            loss=training_result.get('final_loss', 1.0 - training_result.get('final_accuracy', 0.0)),
            val_accuracy=training_result.get('val_accuracy', validation_result.validation_performance.accuracy),
            val_loss=training_result.get('val_loss', None),
            training_time=training_result.get('training_time', None),
            epochs_trained=epochs_trained
        )

        # Salva modelo (arquivo .keras)
        model_path = f"files/{model.id}.keras"
        self.ml_port.save_model(ml_model, model_path)
        model.set_training_completed(
            metrics=training_metrics,
            file_path=model_path
        )

        return validation_result
    
    def _train_leave_one_out(self, ml_model, X, y, model: Model) -> ValidationResult:
        """Treina modelo com leave-one-out validation"""
        from sklearn.model_selection import LeaveOneOut
        
        validation_result = self.validation_service.cross_validate_model(
            ml_model, X, y, strategy=TrainingStrategy.LEAVE_ONE_OUT
        )
        
        # Treina modelo final
        training_result = self.ml_port.train_model(ml_model, X, y)

        # Constrói TrainingMetrics
        history = training_result.get('history', {})
        epochs_trained = 0
        try:
            epochs_trained = len(next(iter(history.values())))
        except Exception:
            epochs_trained = None

        training_metrics = TrainingMetrics(
            accuracy=training_result.get('final_accuracy', validation_result.validation_performance.accuracy),
            loss=training_result.get('final_loss', 1.0 - training_result.get('final_accuracy', 0.0)),
            val_accuracy=training_result.get('val_accuracy', validation_result.validation_performance.accuracy),
            val_loss=training_result.get('val_loss', None),
            training_time=training_result.get('training_time', None),
            epochs_trained=epochs_trained
        )

        # Salva modelo (arquivo .keras)
        model_path = f"files/{model.id}.keras"
        self.ml_port.save_model(ml_model, model_path)
        model.set_training_completed(
            metrics=training_metrics,
            file_path=model_path
        )

        return validation_result
    
    def _train_multi_subject(self, ml_model, X, y, model: Model, validation_split: float) -> ValidationResult:
        """Treina modelo com dados de múltiplos sujeitos"""
        # Para multi-subject, usa validação simples
        validation_result = self.validation_service.validate_single_subject(
            ml_model, X, y, test_size=validation_split
        )
        
        # Treina modelo final
        training_result = self.ml_port.train_model(ml_model, X, y)

        # Constrói TrainingMetrics
        history = training_result.get('history', {})
        epochs_trained = 0
        try:
            epochs_trained = len(next(iter(history.values())))
        except Exception:
            epochs_trained = None

        training_metrics = TrainingMetrics(
            accuracy=training_result.get('final_accuracy', validation_result.validation_performance.accuracy),
            loss=training_result.get('final_loss', 1.0 - training_result.get('final_accuracy', 0.0)),
            val_accuracy=training_result.get('val_accuracy', validation_result.validation_performance.accuracy),
            val_loss=training_result.get('val_loss', None),
            training_time=training_result.get('training_time', None),
            epochs_trained=epochs_trained
        )

        # Salva modelo (arquivo .keras)
        model_path = f"files/{model.id}.keras"
        self.ml_port.save_model(ml_model, model_path)
        model.set_training_completed(
            metrics=training_metrics,
            file_path=model_path
        )

        return validation_result
