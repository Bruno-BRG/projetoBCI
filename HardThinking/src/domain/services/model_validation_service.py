"""
Serviço de validação de modelos
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ..entities.model import Model
from ..value_objects.training_types import ModelPerformance, ValidationResult, TrainingStrategy

class ModelValidationService:
    """Serviço para validação de modelos de ML"""
    
    def __init__(self):
        pass
    
    def validate_model_performance(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray] = None,
                                 loss: Optional[float] = None) -> ModelPerformance:
        """Calcula métricas de performance do modelo"""
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Se loss não fornecido, calcula baseado na acurácia
        if loss is None:
            loss = 1.0 - accuracy
        
        return ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            loss=loss
        )
    
    def cross_validate_model(self, 
                           model_func,
                           X: np.ndarray,
                           y: np.ndarray,
                           cv_folds: int = 5,
                           strategy: TrainingStrategy = TrainingStrategy.CROSS_VALIDATION) -> ValidationResult:
        """Realiza validação cruzada do modelo"""
        
        if strategy == TrainingStrategy.CROSS_VALIDATION:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        elif strategy == TrainingStrategy.LEAVE_ONE_OUT:
            cv = LeaveOneOut()
        else:
            raise ValueError(f"Estratégia {strategy} não suportada para validação cruzada")
        
        # Realiza validação cruzada
        cv_scores = cross_val_score(model_func, X, y, cv=cv, scoring='accuracy')
        
        # Treina modelo com todos os dados para obter métricas detalhadas
        model_func.fit(X, y)
        y_pred = model_func.predict(X)
        
        train_performance = self.validate_model_performance(y, y_pred)
        
        # Para validação, usa média dos scores de CV
        val_accuracy = np.mean(cv_scores)
        val_performance = ModelPerformance(
            accuracy=val_accuracy,
            precision=val_accuracy,  # Aproximação
            recall=val_accuracy,     # Aproximação
            f1_score=val_accuracy,   # Aproximação
            loss=1.0 - val_accuracy
        )
        
        return ValidationResult(
            train_performance=train_performance,
            validation_performance=val_performance,
            cross_validation_scores=cv_scores.tolist()
        )
    
    def validate_single_subject(self,
                               model_func,
                               X: np.ndarray,
                               y: np.ndarray,
                               test_size: float = 0.2) -> ValidationResult:
        """Valida modelo para um único sujeito"""
        from sklearn.model_selection import train_test_split
        import numpy as _np
        try:
            import tensorflow as _tf
        except Exception:
            _tf = None
        
        # Divide dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Se for um modelo Keras (tf.keras.Model), trate labels como categorical e use API Keras
        if _tf is not None and isinstance(model_func, _tf.keras.Model):
            # Converte labels para one-hot
            num_classes = int(_np.max(y) + 1)
            y_train_cat = _tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
            y_test_cat = _tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

            # Treina com poucas épocas para validação (config leve)
            model_func.fit(X_train, y_train_cat, epochs=10, batch_size=32, verbose=0)

            # Predições (retorna índices de classe)
            y_train_pred = _np.argmax(model_func.predict(X_train, verbose=0), axis=1)
            y_test_pred = _np.argmax(model_func.predict(X_test, verbose=0), axis=1)
        else:
            # Treina modelo (sklearn-like)
            model_func.fit(X_train, y_train)

            # Predições
            y_train_pred = model_func.predict(X_train)
            y_test_pred = model_func.predict(X_test)
        
        # Calcula métricas
        train_performance = self.validate_model_performance(y_train, y_train_pred)
        test_performance = self.validate_model_performance(y_test, y_test_pred)
        
        return ValidationResult(
            train_performance=train_performance,
            validation_performance=test_performance,
            test_performance=test_performance
        )
    
    def compare_models(self, 
                      models: List[Tuple[str, object]], 
                      X: np.ndarray, 
                      y: np.ndarray) -> Dict[str, ModelPerformance]:
        """Compara performance de múltiplos modelos"""
        results = {}
        
        for model_name, model_func in models:
            try:
                validation_result = self.cross_validate_model(model_func, X, y)
                results[model_name] = validation_result.validation_performance
            except Exception as e:
                print(f"Erro ao validar modelo {model_name}: {str(e)}")
                # Cria performance com valores baixos em caso de erro
                results[model_name] = ModelPerformance(
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    loss=1.0
                )
        
        return results
    
    def evaluate_generalization(self, 
                               model_func,
                               train_subjects_data: List[Tuple[np.ndarray, np.ndarray]],
                               test_subject_data: Tuple[np.ndarray, np.ndarray]) -> ValidationResult:
        """Avalia generalização do modelo entre sujeitos"""
        
        # Combina dados de treinamento de múltiplos sujeitos
        X_train_combined = np.vstack([data[0] for data in train_subjects_data])
        y_train_combined = np.hstack([data[1] for data in train_subjects_data])
        
        X_test, y_test = test_subject_data
        
        # Treina modelo
        model_func.fit(X_train_combined, y_train_combined)
        
        # Predições
        y_train_pred = model_func.predict(X_train_combined)
        y_test_pred = model_func.predict(X_test)
        
        # Calcula métricas
        train_performance = self.validate_model_performance(y_train_combined, y_train_pred)
        test_performance = self.validate_model_performance(y_test, y_test_pred)
        
        return ValidationResult(
            train_performance=train_performance,
            validation_performance=test_performance,
            test_performance=test_performance
        )
    
    def calculate_model_stability(self, 
                                 model_func,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 n_iterations: int = 10) -> Dict[str, float]:
        """Calcula estabilidade do modelo através de múltiplas execuções"""
        accuracies = []
        
        for _ in range(n_iterations):
            validation_result = self.validate_single_subject(model_func, X, y)
            accuracies.append(validation_result.validation_performance.accuracy)
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'stability_coefficient': 1.0 - (np.std(accuracies) / np.mean(accuracies))
        }
