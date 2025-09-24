"""
Configurações auxiliares globais para BrainBridge v2

Expõe utilitários de caminho e injeta caminhos externos (como HardThinking/src)
quando disponíveis, para evitar erros de import durante desenvolvimento.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent

def ensure_external_paths():
	"""Adiciona HardThinking/src ao sys.path se existir no projeto."""
	ht_src = ROOT.parent / 'HardThinking' / 'src'
	if ht_src.exists():
		for p in (str(ht_src), str(ht_src.parent)):
			if p not in sys.path:
				sys.path.insert(0, p)

# Executa ao importar
ensure_external_paths()

