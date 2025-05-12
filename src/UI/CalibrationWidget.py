# Importações da biblioteca padrão
import os

# Importações de terceiros
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QInputDialog,
    QMessageBox, QGroupBox
)

# Importações locais
from model.BCISystem import create_bci_system
from model.EEGAugmentation import load_local_eeg_data

class CalibrationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Container principal com padding
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Grupo de plotagem
        plot_group = QGroupBox("Visualização do Sinal EEG")
        plot_layout = QVBoxLayout()
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        main_layout.addWidget(plot_group)
        
        # Grupo de navegação
        nav_group = QGroupBox("Navegação")
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("◀ Anterior")
        self.next_button = QPushButton("Próximo ▶")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_group.setLayout(nav_layout)
        main_layout.addWidget(nav_group)
        
        # Grupo de controle
        control_group = QGroupBox("Controles")
        control_layout = QVBoxLayout()
        self.load_button = QPushButton("Carregar Dados EEG")
        self.add_button = QPushButton("Adicionar à Calibração")
        self.train_button = QPushButton("Treinar Modelo")
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.add_button)
        control_layout.addWidget(self.train_button)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        self.setLayout(main_layout)
        
        # Estilo do plot
        plt.style.use('default')
        self.figure.patch.set_facecolor('white')
        
        # Armazenamento de dados
        self.data = None
        self.labels = None
        self.idx = 0
        self.eeg_channel = None
        self.bci = create_bci_system()
        
        # Conectar sinais
        self.load_button.clicked.connect(self.load_data)
        self.prev_button.clicked.connect(self.prev_sample)
        self.next_button.clicked.connect(self.next_sample)
        self.add_button.clicked.connect(self.add_to_calibration)
        self.train_button.clicked.connect(self.train_model)
        
        # Estados iniciais dos botões
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.train_button.setEnabled(False)

    def load_data(self):
        subject_id, ok = QInputDialog.getInt(self, "ID do Sujeito", "Digite o ID do Sujeito:", 1, 1, 109)
        if not ok:
            return
        X, y, ch = load_local_eeg_data(subject_id, augment=False)  # Alterado para sem aumento
        self.data, self.labels, self.eeg_channel = X, y, ch
        self.idx = 0
        
        # Inicializar modelo BCI com contagem de canais
        self.bci.initialize_model(self.eeg_channel)
        
        # Habilitar botões de navegação e controle
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)
        self.add_button.setEnabled(True)
        
        self.update_plot()

    def update_plot(self):
        if self.data is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        sample = self.data[self.idx]
        for ch in range(sample.shape[0]):
            ax.plot(sample[ch], alpha=0.5, linewidth=0.5)
        ax.set_title(f"Amostra {self.idx+1}/{len(self.data)} - Rótulo: {'Esquerda' if self.labels[self.idx]==0 else 'Direita'}")
        ax.set_xlabel("Tempo")
        ax.set_ylabel("Amplitude")
        ax.set_facecolor('lightgray')
        self.canvas.draw()

    def prev_sample(self):
        if self.data is None:
            return
        self.idx = max(0, self.idx - 1)
        self.update_plot()

    def next_sample(self):
        if self.data is None:
            return
        self.idx = min(len(self.data)-1, self.idx + 1)
        self.update_plot()

    def add_to_calibration(self):
        if self.data is None:
            return
        self.bci.add_calibration_sample(self.data[self.idx], int(self.labels[self.idx]))
        self.train_button.setEnabled(True)  # Habilitar botão de treino após adicionar amostras
        QMessageBox.information(self, "Sucesso", "Amostra adicionada ao conjunto de calibração")

    def train_model(self):
        if self.bci and self.eeg_channel:
            # Perguntar ao usuário o nome do modelo
            name, ok = QInputDialog.getText(self, "Nome do Modelo", "Digite o nome para o modelo de calibração:")
            if not ok or not name.strip():
                QMessageBox.warning(self, "Cancelado", "Treinamento cancelado: nenhum nome de modelo fornecido")
                return
            # Garantir que o diretório de checkpoints exista e definir o caminho do modelo
            os.makedirs('checkpoints', exist_ok=True)
            model_path = os.path.join('checkpoints', f"{name}.pth")
            self.bci.model_path = model_path
            try:
                self.bci.train_calibration(num_epochs=50, batch_size=10, learning_rate=5e-4)
                QMessageBox.information(self, "Sucesso", f"Treinamento do modelo concluído e salvo em {model_path}")
            except Exception as e:
                QMessageBox.warning(self, "Erro", f"Falha no treinamento: {str(e)}")
