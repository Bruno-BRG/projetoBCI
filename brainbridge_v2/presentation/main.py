#!/usr/bin/env python3
"""
BrainBridge - Sistema BCI para classificação de Motor Imagery
Ponto de entrada principal único

Uso:
    python main.py              # Inicia interface gráfica
    python main.py --cli         # Interface de linha de comando
    python main.py --simulate    # Modo simulação
    python main.py --train       # Modo treinamento
"""

import sys
import os
import argparse
from pathlib import Path

current_dir = Path(__file__).resolve().parent
package_root = current_dir.parent
repo_root = package_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Use local writable config/cache for matplotlib in restricted Windows environments.
mpl_config_dir = repo_root / ".mplconfig"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

# 1) Configurar TensorFlow para quando ele for carregado sob demanda.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # reduzir logs do TF
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
HAS_ML = None

# 2) Verificar dependências de GUI depois do TF
try:
    import numpy as np  # noqa: F401
    import PyQt5  # noqa: F401
    HAS_GUI = True
except Exception as e:
    print(f"Aviso: Dependências da GUI não encontradas: {e}")
    HAS_GUI = False


def run_gui():
    """Executa a interface gráfica"""
    if not HAS_GUI:
        print("Erro: PyQt5 não está instalado. Execute: pip install PyQt5")
        return 1
    
    try:
        from PyQt5.QtWidgets import QApplication
        from brainbridge_v2.bootstrap.container import build_app_container
        from brainbridge_v2.presentation.gui.main_window import MainWindow
        from brainbridge_v2.presentation.gui.styles import apply_theme
        
        app = QApplication(sys.argv)
        app.setApplicationName("BrainBridge")
        app.setApplicationVersion("2.0.0")
        
        # Aplicar tema visual
        apply_theme(app)
        
        container = build_app_container()
        window = MainWindow(container)
        window.show()
        
        return app.exec_()
        
    except ImportError as e:
        print(f"Erro ao importar GUI: {e}")
        return 1
    except Exception as e:
        print(f"Erro na interface gráfica: {e}")
        return 1


def check_environment():
    """Verifica o ambiente e dependências"""
    print("=== Verificação do Ambiente ===")
    
    # Verificar Python
    print(f"Python: {sys.version}")
    
    # Verificar dependências principais
    dependencies = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'PyQt5': 'PyQt5',
        'tensorflow': 'tensorflow'
    }
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}: Instalado")
        except Exception as exc:
            print(f"✗ {name}: Não instalado ou falhou ao inicializar ({exc})")
    
    # Verificar estrutura de diretórios
    print("\n=== Estrutura de Diretórios ===")
    required_dirs = [
        'application',
        'domain',
        'infrastructure',
        'infrastructure/acquisition',
        'infrastructure/communication',
        'infrastructure/config',
        'infrastructure/database',
        'infrastructure/ml',
        'infrastructure/signal_processing',
        'interface_adapters',
        'presentation',
        'presentation/gui',
        'data/recordings', 'data/models', 'data/database', 'data/logs'
    ]

    for dir_path in required_dirs:
        full_path = package_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (criando...)")
            full_path.mkdir(parents=True, exist_ok=True)


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="BrainBridge - Sistema BCI para classificação de Motor Imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python main.py                    # Interface gráfica
    python main.py --cli              # Linha de comando
    python main.py --simulate         # Simulação de dados
    python main.py --train            # Treinamento de modelos
    python main.py --check-env        # Verificar ambiente
        """
    )
    
    parser.add_argument('--check-env', action='store_true',
                       help='Verificar ambiente e dependências')
    parser.add_argument('--version', action='version', version='BrainBridge 2.0.0')
    
    args = parser.parse_args()
    
    # Verificar ambiente se solicitado
    if args.check_env:
        check_environment()
        return 0
    
    # Executar modo apropriado

    else:
        # Modo padrão: GUI
        return run_gui()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"Erro fatal: {e}")
        sys.exit(1)
