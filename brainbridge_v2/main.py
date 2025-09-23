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

# Adicionar diretório atual ao path para imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Verificar e importar dependências
try:
    import numpy as np
    import PyQt5
    HAS_GUI = True
except ImportError as e:
    print(f"Aviso: Dependências da GUI não encontradas: {e}")
    HAS_GUI = False

try:
    import tensorflow as tf
    HAS_ML = True
except ImportError as e:
    print(f"Aviso: TensorFlow não encontrado: {e}")
    HAS_ML = False


def run_gui():
    """Executa a interface gráfica"""
    if not HAS_GUI:
        print("Erro: PyQt5 não está instalado. Execute: pip install PyQt5")
        return 1
    
    try:
        from PyQt5.QtWidgets import QApplication
        from gui.main_window import MainWindow
        
        app = QApplication(sys.argv)
        app.setApplicationName("BrainBridge")
        app.setApplicationVersion("2.0.0")
        
        window = MainWindow()
        window.show()
        
        return app.exec_()
        
    except ImportError as e:
        print(f"Erro ao importar GUI: {e}")
        return 1
    except Exception as e:
        print(f"Erro na interface gráfica: {e}")
        return 1


def run_cli():
    """Executa interface de linha de comando"""
    print("=== BrainBridge CLI ===")
    print("Interface de linha de comando em desenvolvimento...")
    
    # TODO: Implementar CLI completa
    print("\nOpções disponíveis:")
    print("1. Listar pacientes")
    print("2. Iniciar sessão de gravação")
    print("3. Treinar modelo")
    print("4. Simular dados")
    
    return 0


def run_simulation():
    """Executa modo simulação"""
    print("=== Modo Simulação ===")
    
    try:
        from acquisition.simulators import EEGSimulator
        from core.eeg_data import EEGBuffer
        
        def sample_callback(sample):
            print(f"Amostra recebida: {len(sample.channels)} canais, marker: {sample.marker}")
        
        buffer = EEGBuffer()
        simulator = EEGSimulator(sample_callback)
        
        print("Iniciando simulação... (Ctrl+C para parar)")
        simulator.start()
        
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nParando simulação...")
            simulator.stop()
        
        return 0
        
    except ImportError as e:
        print(f"Erro ao importar simulador: {e}")
        return 1
    except Exception as e:
        print(f"Erro na simulação: {e}")
        return 1


def run_training():
    """Executa modo treinamento"""
    print("=== Modo Treinamento ===")
    
    if not HAS_ML:
        print("Erro: TensorFlow não está instalado. Execute: pip install tensorflow")
        return 1
    
    print("Sistema de treinamento em desenvolvimento...")
    
    # TODO: Implementar interface de treinamento
    try:
        from ml.trainer import ModelTrainer
        print("Trainer importado com sucesso")
        return 0
    except ImportError as e:
        print(f"Erro ao importar trainer: {e}")
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
        except ImportError:
            print(f"✗ {name}: Não instalado")
    
    # Verificar estrutura de diretórios
    print("\n=== Estrutura de Diretórios ===")
    required_dirs = [
        'config', 'core', 'acquisition', 'processing',
        'ml', 'gui', 'database', 'communication', 'utils',
        'data/recordings', 'data/models', 'data/database', 'data/logs'
    ]
    
    for dir_path in required_dirs:
        full_path = current_dir / dir_path
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
    
    parser.add_argument('--cli', action='store_true',
                       help='Executar interface de linha de comando')
    parser.add_argument('--simulate', action='store_true',
                       help='Executar modo simulação')
    parser.add_argument('--train', action='store_true',
                       help='Executar modo treinamento')
    parser.add_argument('--check-env', action='store_true',
                       help='Verificar ambiente e dependências')
    parser.add_argument('--version', action='version', version='BrainBridge 2.0.0')
    
    args = parser.parse_args()
    
    # Verificar ambiente se solicitado
    if args.check_env:
        check_environment()
        return 0
    
    # Executar modo apropriado
    if args.cli:
        return run_cli()
    elif args.simulate:
        return run_simulation()
    elif args.train:
        return run_training()
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