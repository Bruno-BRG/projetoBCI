"""Smoke test rápido para TensorFlow adapter do HardThinking.

Objetivo: verificar que, com TensorFlow instalado, conseguimos criar um modelo via
`TensorFlowMLAdapter`, treinar 1 época em dados aleatórios e avaliar sem erros.

Como usar (PowerShell):
  # ative seu venv (opcional)
  python -m venv .venv; .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  python tools\smoke_tf_test.py

O script imprime progresso e resultados (final accuracy/loss).
"""

import sys
from pathlib import Path
import numpy as np
import os

# Tenta localizar 'HardThinking/src' subindo a árvore de diretórios
hardthinking_src = None
for parent in Path(__file__).resolve().parents:
    candidate = parent / 'HardThinking' / 'src'
    if candidate.exists():
        hardthinking_src = candidate
        break

if hardthinking_src is None:
    print("ERRO: não encontrou 'HardThinking/src' na árvore de pastas. Certifique-se que a pasta HardThinking está ao lado de BrainBridge.")
    raise SystemExit(1)

# Adiciona o diretório pai (HardThinking) para que 'src' seja importável como pacote
hardthinking_root = hardthinking_src.parent
if str(hardthinking_root) not in sys.path:
    sys.path.insert(0, str(hardthinking_root))

try:
    from src.infrastructure.adapters.tensorflow_ml_adapter import TensorFlowMLAdapter
except Exception as e:
    print("ERRO: import padrão falhou:", e)
    print("----- Debug: sys.path -----")
    import sys as _sys
    for p in _sys.path:
        print(p)
    raise

print("Adapter importado com sucesso. Criando modelo...")
adapter = TensorFlowMLAdapter(config={})

# Parâmetros de teste pequenos
time_steps = 250  # corresponde a 2s @125Hz (convenção do projeto)
channels = 16
samples = 32
num_classes = 2

# Criar modelo EEGNET simplificado via adapter
hyperparameters = {
    'input_shape': (time_steps, channels),
    'num_classes': num_classes,
    'learning_rate': 0.001
}
model = adapter.create_model('EEGNET', hyperparameters)
print("Modelo criado:", model.summary())

# Dados falsos
X = np.random.randn(samples, time_steps, channels).astype(np.float32)
y = np.random.randint(0, num_classes, size=(samples,))

# Treinar por 1 época para smoke test
print("Iniciando treino (1 época)...")
history = model.fit(X, (np.eye(num_classes)[y]), epochs=1, batch_size=8, validation_split=0.2, verbose=1)
print("Treino finalizado.")

# Avaliar via adapter
metrics = adapter.evaluate_model(model, X, y)
print("Avaliação:", metrics)

print("Smoke test concluído com sucesso.")
