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
        
        # Configure matplotlib for high performance
        plt.style.use('fast')  # Use fast style
        self.figure = plt.figure(facecolor='white')
        self.figure.set_dpi(80)  # Lower DPI for better performance
        
        # Use new Qt5 backend
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
        self.timer.setInterval(16)  # ~60fps update rate
        
        # Buffer and plot settings
        self.buffer_size = 400  # Fixed buffer size of 400 samples
        self.max_update_rate = 30  # Frame rate limit in Hz
        self.last_update_time = 0
        self.background = None
        self.need_redraw = True
        self.lines = []
        self.axes = []
        
        # Chunk processing settings
        self.max_chunk_size = 32
        self.min_update_interval = 1.0 / self.max_update_rate
        
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
            # Initialize LSL inlet with minimal buffering
            self.inlet = StreamInlet(eeg_streams[0], max_buflen=1, processing_flags=0)
            info = self.inlet.info()
            n_ch = info.channel_count()
            sr = int(info.nominal_srate())
            self.sr = sr
            
            # Buffer size is fixed at 400 samples
            self.buffer_size = 400  # Fixed window size
            
            # Setup optimized figure size
            self.figure.set_size_inches(10, max(4, n_ch * 1.5))
            
            # Initialize streaming filter and BCI system
            self.filter = EEGFilter(sfreq=self.sr)
            selected = self.model_combo.currentText() if hasattr(self, 'model_combo') else None
            model_path = os.path.join('checkpoints', selected) if selected else None
            self.bci = create_bci_system(model_path=model_path)
            self.bci.initialize_model(n_ch)
            
            # Configure windows and timing
            self.window_size = 400
            self.window_step = 50
            update_interval = max(16, int(1000 / self.max_update_rate))
            self.timer.setInterval(update_interval)
            
            # Initialize optimized buffers with fixed size
            self.sample_since_last = 0
            self.buffer = [deque(maxlen=400) for _ in range(n_ch)]  # Fixed size of 400
            self.buffer_x = np.arange(400)  # Pre-allocate x data for 400 samples
            
            # Setup optimized plotting
            self.figure.clear()
            self.axes = []
            self.lines = []
            
            # Create subplots with optimized settings
            for idx in range(n_ch):
                ax = self.figure.add_subplot(n_ch, 1, idx+1)
                ax.set_ylim(-100, 100)
                ax.set_xlim(0, 400)  # Fixed x-axis range for 400 samples
                ax.set_ylabel(f"Ch {idx+1}")
                ax.grid(False)
                ax.set_facecolor('white')
                ax.minorticks_off()
                if idx < n_ch - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Amostras")
                
                # Create animated line with minimal properties
                line, = ax.plot([], [], 'b-', animated=True, linewidth=1.0)
                self.lines.append(line)
                self.axes.append(ax)
            
            # Initial draw to setup background
            self.figure.tight_layout()
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.figure.bbox)
            
            # Set canvas height
            height_px = int(self.figure.get_figheight() * self.figure.get_dpi())
            self.canvas.setMinimumHeight(height_px)
            
            # Enable UI elements
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.timer.start()
            
            logging.info(f"Transmissão iniciada: {n_ch} canais a {self.sr} Hz, buffer fixo de 400 amostras")

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
            
        # Frame rate limiting
        current_time = datetime.now().timestamp()
        if current_time - self.last_update_time < self.min_update_interval:
            return
            
        # Get all available chunks efficiently
        chunks = []
        total_samples = 0
        while total_samples < self.max_chunk_size:
            chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=self.max_chunk_size - total_samples)
            if not chunk:
                break
            chunks.extend(chunk)
            total_samples += len(chunk)
            
        if not chunks:
            return
            
        # Process all chunks at once
        arr = np.array(chunks).T
        filtered_chunk = self.filter.filter_stream(arr)
        
        # Update buffers with vectorized operations
        for ch_idx, channel_samples in enumerate(filtered_chunk):
            self.buffer[ch_idx].extend(channel_samples)
        
        # Only update plot if enough time has passed
        self.last_update_time = current_time
        
        # Fast plot update using blitting
        if self.background is not None:
            self.canvas.restore_region(self.background)
            
            # Get all channel data at once
            buffer_data = [np.array(list(b)) for b in self.buffer]
            
            # Update all lines efficiently
            for idx, (line, data) in enumerate(zip(self.lines, buffer_data)):
                line.set_data(self.buffer_x[:len(data)], data)
                self.axes[idx].draw_artist(line)
                
                # Update y-limits if needed (less frequently)
                if self.process_check.isChecked() and idx % 3 == current_time % 3:  # Stagger updates
                    data_min, data_max = data.min(), data.max()
                    current_ymin, current_ymax = self.axes[idx].get_ylim()
                    if data_min < current_ymin or data_max > current_ymax:
                        margin = (data_max - data_min) * 0.1
                        self.axes[idx].set_ylim(data_min - margin, data_max + margin)
                        self.need_redraw = True
            
            # Efficient canvas updates
            self.canvas.blit(self.figure.bbox)
            
            # Full redraw only when necessary
            if self.need_redraw:
                self.canvas.draw()
                self.background = self.canvas.copy_from_bbox(self.figure.bbox)
                self.need_redraw = False
        
        # Handle data capture if active
        if self.capturing and chunks:
            self.capture_buffer.extend(chunks)
            self.sample_count += len(chunks)
            if self.sample_count >= self.capture_needed:
                self.capturing = False
                self.capture_button.setEnabled(True)
                data_arr = np.array(self.capture_buffer)
                os.makedirs('captured_data', exist_ok=True)
                filename = datetime.now().strftime("captured_data/capture_%Y%m%d_%H%M%S.npy")
                np.save(filename, data_arr)
                logging.info(f"Dados capturados salvos em {filename}")

        # Classification with sliding window
        if hasattr(self, 'bci') and self.bci.is_calibrated and self.buffer:
            self.sample_since_last += len(chunks)
            if self.sample_since_last >= self.window_step and len(self.buffer[0]) >= self.window_size:
                self.sample_since_last = 0
                # Extract window data efficiently
                window_data = np.array([list(b)[-self.window_size:] for b in self.buffer])
                
                # Get prediction
                pred, conf = self.bci.predict_movement(window_data)
                
                # Update UI labels
                self.pred_label.setText(f"Pred: {pred}")
                self.conf_label.setText(f"Conf: {conf:.2%}")
                logging.info(f"Predição do modelo: {pred} (confiança {conf:.2%})")

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
