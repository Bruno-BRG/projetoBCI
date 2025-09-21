"""
Interface PyQt5 para Sistema BCI
Funcionalidades:
- Cadastro de pacientes
- Streaming de dados em tempo real com visualização
- Gravação de dados atrelada ao paciente com marcadores T1, T2 e Baseline
- Predição em tempo real durante o modo jogo
"""

import sys
import os
import time
import random
import sqlite3
import os
import threading
import time
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import deque
# Preload TensorFlow runtime (if installed) before importing any GUI or
# other native-extension libraries. This reduces the chance of Windows
# DLL initialization conflicts (common when mixing TF and PyQt).
try:
    import tensorflow as _tf
    print(f'Preloaded TensorFlow: {_tf.__version__}')
except Exception:
    pass

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QLabel, QLineEdit, 
                           QPushButton, QComboBox, QSpinBox, QTextEdit,
                           QTableWidget, QTableWidgetItem, QGroupBox,
                           QGridLayout, QMessageBox, QProgressBar, QCheckBox,
                           QSplitter, QFrame, QDateEdit, QDoubleSpinBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QDate
from PyQt5.QtGui import QFont, QPixmap, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
import traceback
# Detectar se estamos executando como módulo ou script direto
if __name__ == "__main__" and __package__ is None:
    # Executando como script direto, adicionar o diretório pai ao path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    # Usar imports absolutos
    from bci.configs.config import get_recording_path, get_database_path, ensure_folders_exist
else:
    # Executando como módulo, usar imports relativos
    from .configs.config import get_recording_path, get_database_path, ensure_folders_exist

# Importar configuração de caminhos

# Importar o logger OpenBCI
if __name__ == "__main__" and __package__ is None:
    # Executando como script direto
    try:
        from bci.network.openbci_csv_logger import OpenBCICSVLogger
        USE_OPENBCI_LOGGER = True
    except ImportError:
        USE_OPENBCI_LOGGER = False
        print("OpenBCI Logger não encontrado, usando logger simples")
else:
    # Executando como módulo
    try:
        from .network.openbci_csv_logger import OpenBCICSVLogger
        USE_OPENBCI_LOGGER = True
    except ImportError:
        USE_OPENBCI_LOGGER = False
        print("OpenBCI Logger não encontrado, usando logger simples")


# Importar módulos do sistema existente
import csv

# Não precisa adicionar ao path pois os módulos estão na mesma pasta agora
if __name__ == "__main__" and __package__ is None:
    try:
        from bci.network.udp_receiver_BCI import UDPReceiver
        from bci.network.realtime_udp_converter import RealTimeUDPConverter
        print("Módulos do sistema carregados com sucesso")
    except ImportError as e:
        print(f"Erro ao importar módulos: {e}")
        print("Usando modo de simulação")
else:
    try:
        from .network.udp_receiver_BCI import UDPReceiver
        from .network.realtime_udp_converter import RealTimeUDPConverter
        print("Módulos do sistema carregados com sucesso")
    except ImportError as e:
        print(f"Erro ao importar módulos: {e}")
        print("Usando modo de simulação")
    # Criar classes mock para desenvolvimento
    class UDPReceiver:
        def __init__(self, host, port): 
            self.host = host
            self.port = port
            self.callback = None
            
        def set_callback(self, callback): 
            self.callback = callback
            
        def start(self): 
            # Simular erro de porta ocupada ocasionalmente
            import random
            if random.random() < 0.3:
                raise Exception("[WinError 10048] Porta já em uso")
            
        def stop(self): 
            pass
    
    class RealTimeUDPConverter:
        def __init__(self): pass
        def process_udp_data(self, data): 
            # Simular dados EEG para desenvolvimento
            return np.random.randn(16) * 50
    
# Importar a janela principal
if __name__ == "__main__" and __package__ is None:
    try:
        from bci.ui.BCI_main_window import BCIMainWindow
    except ImportError as e:
        print(f"Erro ao importar BCIMainWindow: {e}")
        BCIMainWindow = None
else:
    try:
        from .ui.BCI_main_window import BCIMainWindow
    except ImportError as e:
        print(f"Erro ao importar BCIMainWindow: {e}")
        BCIMainWindow = None


def main():
    """Função principal do sistema BCI"""
    # Garantir que as pastas necessárias existam
    ensure_folders_exist()
    
    # Criar aplicação PyQt5
    app = QApplication(sys.argv)
    # Inicializar captura/log de erros do PyQt -> faz com que qualquer QMessageBox
    # mostrado também seja impresso no terminal (stderr) e que exceções não tratadas
    # sejam logadas e exibidas em um QMessageBox.
    try:
        setup_qt_error_logging()
    except Exception:
        # Não falhar a inicialização da aplicação se a instrumentação falhar
        print("Aviso: não foi possível habilitar o logging de QMessageBox.", file=sys.stderr)
    
    if BCIMainWindow:
        # Usar a interface principal modularizada
        window = BCIMainWindow()
        window.show()
    else:
        # Mostrar erro se não conseguir carregar a interface
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Erro")
        msg.setText("Não foi possível carregar a interface principal do sistema BCI.")
        msg.setDetailedText("Verifique se todos os módulos estão instalados corretamente.")
        msg.exec_()
        return
    
    # Executar aplicação
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


def setup_qt_error_logging():
    """Monkeypatch QMessageBox to also print its messages to stderr and
    install an excepthook that logs unhandled exceptions to stderr and shows
    a message box (which will be logged by the patched exec_).

    This is safe to call after PyQt5 is imported and a QApplication exists.
    """
    # Local import to avoid issues if PyQt isn't available at import time
    from PyQt5.QtWidgets import QMessageBox

    # Patch instance exec_ to log dialog contents
    orig_exec = QMessageBox.exec_

    def exec_with_log(self, *args, **kwargs):
        try:
            title = self.windowTitle() if hasattr(self, 'windowTitle') else ''
            text = self.text() if hasattr(self, 'text') else ''
            info = self.informativeText() if hasattr(self, 'informativeText') else ''
            detail = self.detailedText() if hasattr(self, 'detailedText') else ''
        except Exception:
            title = text = info = detail = '<unavailable>'
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] PyQt5 QMessageBox - {title}: {text}", file=sys.stderr)
        if info:
            print(f"  Info: {info}", file=sys.stderr)
        if detail:
            print(f"  Details: {detail}", file=sys.stderr)
        return orig_exec(self, *args, **kwargs)

    QMessageBox.exec_ = exec_with_log

    # Patch convenience/static methods like QMessageBox.critical(...)
    for name in ('critical', 'warning', 'information', 'question'):
        orig = getattr(QMessageBox, name, None)
        if orig is None:
            continue

        def make_wrapper(orig_func):
            def wrapper(*args, **kwargs):
                try:
                    # signature: (parent, title, text, ...)
                    title = args[1] if len(args) > 1 else kwargs.get('title', '')
                    text = args[2] if len(args) > 2 else kwargs.get('text', '')
                except Exception:
                    title = text = '<unavailable>'
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{ts}] PyQt5 QMessageBox.{orig_func.__name__} - {title}: {text}", file=sys.stderr)
                return orig_func(*args, **kwargs)
            return wrapper

        setattr(QMessageBox, name, make_wrapper(orig))

    # Install a excepthook that logs to stderr and shows a message box
    def qt_excepthook(exc_type, exc_value, exc_tb):
        tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] Unhandled exception:\n{tb}", file=sys.stderr)
        try:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Unhandled Exception")
            msg.setText(str(exc_value))
            msg.setDetailedText(tb)
            # exec_ is patched and will log this dialog too
            msg.exec_()
        except Exception:
            # Se algo falhar ao mostrar o dialog, apenas continue
            pass
        # Chamar o excepthook original para manter comportamento padrão
        try:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    sys.excepthook = qt_excepthook





