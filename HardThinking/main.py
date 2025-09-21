#!/usr/bin/env python3
"""
HardThinking - Sistema de Treinamento EEG Motor Imagery
Ponto de entrada principal - Mantém compatibilidade com sistema original
"""

import os
import sys
from pathlib import Path

# Adiciona caminhos necessários
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

def main():
    """Função principal de entrada"""
    try:
        # Importa e executa a interface CLI
        from src.interfaces.cli.main_cli import main as cli_main
        cli_main()
        
    except ImportError as e:
        print(f"Erro ao importar módulos: {e}")
        print("Certifique-se de que todas as dependências estão instaladas.")
        sys.exit(1)
    except Exception as e:
        print(f"Erro durante execução: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
