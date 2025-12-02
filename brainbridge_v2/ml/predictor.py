"""
Preditor em tempo real para BrainBridge v2

Carrega um modelo Keras (.keras) e realiza predições em janelas
no formato (timesteps, channels) = (250, 16) por padrão.
"""

from typing import Tuple, Dict, Any
import numpy as np

from . import models as _models


class Predictor:
	"""Wrapper simples para predição de janelas EEG."""

	def __init__(self, model_path: str):
		# Carrega via módulo para permitir monkeypatch em testes
		self.model = _models.load_keras_model(model_path)

	def predict_window(self, window: np.ndarray) -> Dict[str, Any]:
		"""Prediz classe para uma janela EEG.

		Args:
			window: array shape (timesteps, channels)
		Returns:
			dict: { 'probs': [p_left, p_right], 'label': 'left'|'right' }
		"""
		if window.ndim != 2:
			raise ValueError("Esperado window com shape (timesteps, channels)")

		x = window.astype('float32')[None, ...]  # (1, T, C)
		probs = self.model.predict(x, verbose=0)[0]
		idx = int(np.argmax(probs))
		label = 'left' if idx == 0 else 'right'
		return {
			'probs': probs.tolist(),
			'label': label
		}

