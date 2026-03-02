"""
Avaliação de modelos (compatível com funcionalidades do HardThinking)

Inclui:
- compute_metrics: calcula acurácia/F1 de vetores de rótulos
- validate_single_subject: split treino/teste
- cross_validate_accuracy: validação cruzada estratificada (k-fold)
"""

from typing import Dict, Optional
import numpy as np


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
	try:
		from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
	except Exception as e:
		raise ImportError("scikit-learn não está instalado. 'pip install scikit-learn'") from e

	return {
		'accuracy': float(accuracy_score(y_true, y_pred)),
		'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
		'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
		'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
	}


def validate_single_subject(model, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, float]:
	"""Divide em treino/teste estratificado e retorna métricas no teste.

	Se 'model' for um tf.keras.Model, assume y como rótulos inteiros e usa API Keras.
	"""
	try:
		from sklearn.model_selection import train_test_split
	except Exception as e:
		raise ImportError("scikit-learn não está instalado. 'pip install scikit-learn'") from e

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=42, stratify=y
	)

	# Detectar se é Keras model
	is_keras = False
	try:
		import tensorflow as tf  # noqa
		is_keras = hasattr(model, 'fit') and hasattr(model, 'predict') and hasattr(model, 'compile')
	except Exception:
		pass

	if is_keras:
		# treino rápido para validação
		model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
		y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
	else:
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)

	return compute_metrics(y_test, y_pred)


def cross_validate_accuracy(model_builder, X: np.ndarray, y: np.ndarray, k_folds: int = 5) -> Dict[str, float]:
	"""Validação cruzada estratificada. model_builder deve retornar um modelo novo a cada fold.
	Retorna média e desvio padrão da acurácia.
	"""
	try:
		from sklearn.model_selection import StratifiedKFold
	except Exception as e:
		raise ImportError("scikit-learn não está instalado. 'pip install scikit-learn'") from e

	skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
	accuracies = []
	for train_idx, test_idx in skf.split(X, y):
		X_train, X_test = X[train_idx], X[test_idx]
		y_train, y_test = y[train_idx], y[test_idx]
		model = model_builder()
		try:
			model.fit(X_train, y_train)
			y_pred = model.predict(X_test)
		except Exception:
			# Keras
			model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
			y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
		acc = (y_pred == y_test).mean()
		accuracies.append(float(acc))

	return {
		'cv_mean_accuracy': float(np.mean(accuracies)),
		'cv_std_accuracy': float(np.std(accuracies)),
		'cv_k_folds': int(k_folds),
	}

