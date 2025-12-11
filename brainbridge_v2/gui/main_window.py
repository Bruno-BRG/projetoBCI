from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QTabWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtCore import QSize

# Imports ajustados para nova estrutura
from database.manager import DatabaseManager
from gui.widgets.patient_form import PatientRegistrationWidget
from gui.widgets.streaming import StreamingWidget
from gui.styles import Theme


class MainWindow(QMainWindow):
    """Janela principal da aplicação BCI (v2)"""

    def __init__(self):
        super().__init__()

        # Inicializar gerenciador de banco de dados
        try:
            self.db_manager = DatabaseManager()
            # Testar conexão
            if self.db_manager.test_connection():
                print("Sistema BCI inicializado com banco de dados funcionando")
            else:
                print("Aviso: Problemas com o banco de dados")
        except Exception as e:
            print(f"Erro ao inicializar banco de dados: {e}")
            # Tentar criar novamente
            try:
                self.db_manager = DatabaseManager()
            except Exception as e2:
                print(f"Falha crítica no banco de dados: {e2}")

        self.setup_ui()

    def setup_ui(self):
        """Configura a interface principal"""
        self.setWindowTitle("BrainBridge - Sistema BCI")
        self.setGeometry(100, 100, 1400, 900)
        
        # Aplicar tema
        self.setStyleSheet(Theme.get_stylesheet())

        # Widget central com abas
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Título compacto
        title_label = QLabel("🧠 BrainBridge")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 12, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"""
            color: {Theme.PRIMARY_DARK_GREEN};
            background-color: {Theme.VERY_LIGHT_GREEN};
            padding: 6px;
            border-radius: 4px;
            border: 1px solid {Theme.LIGHT_GREEN};
        """)
        layout.addWidget(title_label)

        # Abas com estilo
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(Theme.get_stylesheet())
        self.tabs.setTabPosition(QTabWidget.North)

        # Aba de pacientes
        self.patient_widget = PatientRegistrationWidget(self.db_manager)
        self.tabs.addTab(self.patient_widget, "👥 Cadastro de Pacientes")

        # Aba de streaming
        self.streaming_widget = StreamingWidget(self.db_manager)
        self.tabs.addTab(self.streaming_widget, "📊 Streaming e Gravação")

        layout.addWidget(self.tabs, 1)
        central_widget.setLayout(layout)

        # Barra de status com estilo
        self.statusBar().showMessage("✓ Sistema BCI inicializado")
        self.statusBar().setStyleSheet(f"""
            background-color: {Theme.PRIMARY_DARK_GREEN};
            color: {Theme.WHITE};
            border-top: 2px solid {Theme.LIGHT_GREEN};
            padding: 5px;
        """)

    def closeEvent(self, event):
        """Evento de fechamento da aplicação"""
        # Parar streaming se estiver rodando
        if hasattr(self.streaming_widget, 'streaming_thread') and \
           self.streaming_widget.streaming_thread is not None:
            try:
                self.streaming_widget.streaming_thread.stop_streaming()
            except Exception:
                pass

        event.accept()