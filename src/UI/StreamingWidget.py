# Importações da biblioteca padrão
import os
import numpy as np
from model.EEGFilter import EEGFilter
import logging
from datetime import datetime
from collections import deque

# Importações de terceiros
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox,
    QScrollArea, QCheckBox, QGroupBox, QComboBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer

# Importações locais
from model.BCISystem import create_bci_system
from pylsl import StreamInlet, resolve_streams

class StreamingWidget(QWidget):
    """Widget para transmissão LSL ao vivo e plotagem de EEG em tempo real"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Layout principal com padding
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Grupo de plotagem
        plot_group = QGroupBox("Monitor de Sinal EEG")
        plot_layout = QVBoxLayout()
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.canvas)
        plot_layout.addWidget(self.scroll)
        plot_group.setLayout(plot_layout)
        main_layout.addWidget(plot_group)
        
        # Grupo de status
        status_group = QGroupBox("Status da Predição")
        status_layout = QHBoxLayout()
        self.pred_label = QLabel("Pred: N/A")
        self.conf_label = QLabel("Conf: N/A")
        self.pred_label.setStyleSheet("font-size: 14px; padding: 5px;")
        self.conf_label.setStyleSheet("font-size: 14px; padding: 5px;")
        status_layout.addWidget(self.pred_label)
        status_layout.addWidget(self.conf_label)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Grupo de controle
        control_group = QGroupBox("Controles de Transmissão")
        control_layout = QVBoxLayout()
        
        # Linha de controles de transmissão
        stream_layout = QHBoxLayout()
        self.start_btn = QPushButton("Iniciar Transmissão")
        self.stop_btn = QPushButton("Parar Transmissão")
        self.stop_btn.setEnabled(False)
        stream_layout.addWidget(self.start_btn)
        stream_layout.addWidget(self.stop_btn)
        control_layout.addLayout(stream_layout)
        
        # Linha de controles de processamento
        process_layout = QHBoxLayout()
        self.process_check = QCheckBox("Habilitar Processamento de Sinal")
        self.process_check.setStyleSheet("font-size: 12px; padding: 5px;")
        self.capture_button = QPushButton("Capturar Janela de 5s")
        process_layout.addWidget(self.process_check)
        process_layout.addWidget(self.capture_button)
        control_layout.addLayout(process_layout)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Dropdown de seleção de modelo
        model_group = QGroupBox("Selecionar Modelo")
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        os.makedirs('checkpoints', exist_ok=True)
        models = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        self.model_combo.addItems(models)
        self.model_combo.currentIndexChanged.connect(self.on_model_change)
        self.browse_model_btn = QPushButton("Procurar Modelo...")
        self.browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.browse_model_btn)
        model_group.setLayout(model_layout)
        main_layout.insertWidget(2, model_group)  # após controles
        
        self.setLayout(main_layout)
        
        # Configuração do inlet LSL e buffer
        self.inlet = None
        self.buffer = None
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        
        # Variáveis de captura
        self.capturing = False
        self.capture_buffer = []
        self.sample_count = 0
        self.capture_needed = 0
        
        # Conectar sinais
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)
        self.timer.timeout.connect(self.update_plot)
        self.capture_button.clicked.connect(self.start_capture)
        
        # Configuração do sistema BCI
        self.bci = create_bci_system()
        self.filter = None  # will hold EEGFilter for streaming
        # Carregar seleção inicial do modelo
        if models:
            self.on_model_change(0)

    def start_stream(self):
        logging.info("Iniciando transmissão LSL EEG")
        streams = resolve_streams(wait_time=1.0)
        eeg_streams = [s for s in streams if s.type() == 'EEG']
        if eeg_streams:
            self.inlet = StreamInlet(eeg_streams[0])
            info = self.inlet.info()
            n_ch = info.channel_count()
            # ajustar tamanho da figura para contagem de canais (altura em polegadas por canal)
            self.figure.set_size_inches(10, max(4, n_ch * 1.5))
            sr = int(info.nominal_srate())
            self.sr = sr  # armazenar taxa de amostragem para captura
            # Initialize streaming filter with correct sampling rate
            self.filter = EEGFilter(sfreq=self.sr)
            # Inicializar sistema BCI com modelo selecionado
            selected = self.model_combo.currentText() if hasattr(self, 'model_combo') else None
            model_path = os.path.join('checkpoints', selected) if selected else None
            self.bci = create_bci_system(model_path=model_path)
            self.bci.initialize_model(n_ch)
            if not self.bci.is_calibrated:
                QMessageBox.warning(self, "Aviso do Modelo",
                    "Checkpoint incompatível ou não carregado. Classificação desativada até calibração.")
            
            # definir janela deslizante de 1s e predição a cada 1s
            # usar janela deslizante de 400 amostras e passo de 50 amostras
            self.window_size = 400
            self.window_step = 50
            # atualizar temporizador para disparar com base no passo de amostra na taxa de amostragem atual
            self.timer.setInterval(int(self.window_step / self.sr * 1000))
            

            self.sample_since_last = 0
            buf_len = 500  # número fixo de amostras para exibir
            self.buffer = [deque(maxlen=buf_len) for _ in range(n_ch)]
            # Preparar figura para atualização eficiente em tempo real
            self.figure.clear()
            self.axes = []
            self.lines = []
            for idx in range(n_ch):
                ax = self.figure.add_subplot(n_ch, 1, idx+1)
                ax.set_ylim(-100, 100)  # faixa de amplitude estática
                ax.set_xlim(0, buf_len)  # janela de amostra fixa no eixo x
                ax.set_ylabel(f"Ch {idx+1}")
                if idx < n_ch - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Amostras")
                line, = ax.plot([], [], color='blue')
                self.lines.append(line)
                self.axes.append(ax)
            self.figure.tight_layout()
            self.canvas.draw()
            # garantir que a tela seja alta o suficiente e habilitar rolagem
            height_px = int(self.figure.get_figheight() * self.figure.get_dpi())
            self.canvas.setMinimumHeight(height_px)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.timer.start()
            logging.info(f"Transmissão iniciada: {n_ch} canais a {self.sr} Hz")

    def start_capture(self):
        """Iniciar captura dos próximos 5 segundos de dados EEG"""
        if not self.inlet or not hasattr(self, 'sr'):
            return
        self.capture_buffer = []
        self.sample_count = 0
        self.capture_needed = int(self.sr * 5)
        self.capturing = True
        self.capture_button.setEnabled(False)
        logging.info("Captura de dados de 5s iniciada")

    def stop_stream(self):
        self.timer.stop()
        self.inlet = None
        self.buffer = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        logging.info("Transmissão EEG parada")

    def update_plot(self):
        if not self.inlet:
            return
        chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=32)
        if chunk:
            # Always filter incoming chunk and append to buffers
            arr = np.array(chunk).T  # shape (n_channels, n_samples)
            filtered_chunk = self.filter.filter_stream(arr)
            for ch_idx, channel_samples in enumerate(filtered_chunk):
                for sample_val in channel_samples:
                    self.buffer[ch_idx].append(sample_val)
            
            # atualizar cada objeto de linha com novos dados do buffer
            for idx, line in enumerate(self.lines):
                data = list(self.buffer[idx])
                line.set_data(range(len(data)), data)
            
            # Atualizar eixos
            max_samples = self.buffer[0].maxlen
            for ax in self.axes:
                ax.set_xlim(0, max_samples)
                if not self.process_check.isChecked():
                    # Manter eixo y fixo para dados brutos
                    ax.set_ylim(-100, 100)
                else:
                    # Permitir autoescalonamento se o processamento estiver habilitado
                    ax.relim()
                    ax.autoscale_view(scaley=True)
            
            # redesenhar tela de forma eficiente
            self.canvas.draw_idle()

        # Lógica de captura: acumular dados por 5 segundos
        if self.capturing and chunk:
            for sample in chunk:
                self.capture_buffer.append(sample)
            self.sample_count += len(chunk)
            if self.sample_count >= self.capture_needed:
                self.capturing = False
                self.capture_button.setEnabled(True)
                data_arr = np.array(self.capture_buffer)
                os.makedirs('captured_data', exist_ok=True)
                filename = datetime.now().strftime("captured_data/capture_%Y%m%d_%H%M%S.npy")
                np.save(filename, data_arr)
                logging.info(f"Dados capturados salvos em {filename}")

        # Classificação em tempo real com janela deslizante
        if hasattr(self, 'bci') and self.bci.is_calibrated and self.buffer and self.window_size:
            if len(self.buffer[0]) >= self.window_size:
                self.sample_since_last += len(chunk)
                if self.sample_since_last >= self.window_step:
                    self.sample_since_last = 0                    # extrair últimas amostras de window_size
                    window_data = np.array([list(self.buffer[i])[-self.window_size:] for i in range(len(self.buffer))])
                    
                    # Get prediction
                    pred, conf = self.bci.predict_movement(window_data)
                    
                    # Atualizar rótulos da GUI
                    self.pred_label.setText(f"Pred: {pred}")
                    self.conf_label.setText(f"Conf: {conf:.2%}")
                    logging.info(f"Predição do modelo: {pred} (confiança {conf:.2%}) na janela de {self.window_size} amostras")

    def on_model_change(self, index):
        """Lidar com seleção de checkpoint"""
        name = self.model_combo.currentText()
        if name:
            path = os.path.join('checkpoints', name)
            self.bci = create_bci_system(model_path=path)
            # se já estiver transmitindo canais conhecidos, inicializar modelo para carregar estado
            # será carregado em start_stream

    def browse_model(self):
        """Permitir que o usuário escolha um arquivo de modelo do disco"""
        start_dir = os.getcwd() + os.sep + 'checkpoints'
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar arquivo de modelo", start_dir, "Modelo PyTorch (*.pth)")
        if path:
            name = os.path.basename(path)
            if self.model_combo.findText(name) == -1:
                self.model_combo.addItem(name)
            self.model_combo.setCurrentText(name)
