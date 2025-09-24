"""
Definições de modelos de ML para BrainBridge v2

Inclui construção de um modelo CNN 1D simples para classificação de MI
e utilitários para carregar/salvar modelos Keras.
"""

from typing import Tuple, Optional
import importlib


def build_cnn_1d(input_shape: Tuple[int, int] = (250, 16), num_classes: int = 2):
	"""Constroi um modelo CNN 1D simples em Keras.

	Args:
		input_shape: (timesteps, channels)
		num_classes: número de classes (2 para T1/T2)

	Returns:
		tf.keras.Model
	"""
	
	try:
		layers = importlib.import_module('tensorflow.keras.layers')
		models = importlib.import_module('tensorflow.keras.models')
	except Exception as e:
		raise ImportError(
			"TensorFlow não está disponível. Instale com 'pip install tensorflow'"
		) from e

	inputs = layers.Input(shape=input_shape)
	x = inputs

	# Conv1D espera (timesteps, features). Aqui features=canais.
	x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling1D(pool_size=2)(x)

	x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPooling1D(pool_size=2)(x)

	x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.GlobalAveragePooling1D()(x)

	x = layers.Dropout(0.3)(x)
	x = layers.Dense(64, activation='relu')(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)

	model = models.Model(inputs=inputs, outputs=outputs, name='cnn1d_mi')
	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
	return model


def load_keras_model(model_path: str):
	"""Carrega um modelo Keras salvo (.keras ou .h5)."""
	try:
		keras_models = importlib.import_module('tensorflow.keras.models')
	except Exception as e:
		raise ImportError(
			"TensorFlow não está disponível. Instale com 'pip install tensorflow'"
		) from e
	return keras_models.load_model(model_path)


def build_simple_eegnet(input_shape: Tuple[int, int] = (250, 16), num_classes: int = 2):
	"""Variante simplificada tipo EEGNet, compatível com dados (T,C).

	Útil para experimentos rápidos mantendo a mesma assinatura do CNN_1D.
	"""
	try:
		layers = importlib.import_module('tensorflow.keras.layers')
		models = importlib.import_module('tensorflow.keras.models')
	except Exception as e:
		raise ImportError("TensorFlow não está disponível. Instale com 'pip install tensorflow'") from e

	inputs = layers.Input(shape=input_shape)
	x = layers.Conv1D(16, 64, padding='same')(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.Conv1D(32, 1)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('elu')(x)
	x = layers.MaxPooling1D(pool_size=4)(x)
	x = layers.Dropout(0.25)(x)

	x = layers.Conv1D(32, 16, padding='same', groups=1)(x)
	x = layers.Conv1D(32, 1)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('elu')(x)
	x = layers.MaxPooling1D(pool_size=8)(x)
	x = layers.Dropout(0.25)(x)

	x = layers.Flatten()(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)
	model = models.Model(inputs, outputs, name='eegnet_simple')
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model


def save_model(model, path: str) -> None:
	"""Salva modelo Keras em path (.keras recomendado)."""
	import os
	os.makedirs(os.path.dirname(path), exist_ok=True)
	model.save(path)
