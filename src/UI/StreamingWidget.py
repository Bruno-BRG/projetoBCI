# Importações da biblioteca padrão
import os
import numpy as np
import logging
from datetime import datetime
from collections import deque
import time

# Performance optimizations for matplotlib
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend for better performance
import matplotlib.pyplot as plt
plt.style.use('fast')  # Use fast style
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000

# Importações de terceiros
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox,
    QScrollArea, QCheckBox, QGroupBox, QComboBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer

# Importações locais
from model.BCISystem import create_bci_system
from model.EEGAugmentation import EEGAugmentation
from pylsl import StreamInlet, resolve_streams

class StreamingWidget(QWidget):
    """Widget para transmissão LSL ao vivo e plotagem de EEG em tempo real"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Layout principal com padding
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Grupo de plotagem com otimizações
        plot_group = QGroupBox("Monitor de Sinal EEG")
        plot_layout = QVBoxLayout()
        
        # Create optimized figure
        self.figure = plt.figure(facecolor='white', figsize=(10, 6), dpi=80)  # Lower DPI for better performance
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)
        
        # Enable widget attributes for better performance
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground)
        
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
        # Optimized timer configuration
        self.timer = QTimer(self)
        self.timer.setInterval(10)  # 100 Hz update rate for graph
        
        # Variáveis de captura
        self.capturing = False
        self.capture_buffer = []
        self.sample_count = 0
        self.capture_needed = 0
        
        # Initialize optimization variables
        self.last_plot_time = 0
        self.last_prediction_time = 0
        self.prediction_interval = 50  # 50ms between predictions
        
        # Conectar sinais
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)
        self.timer.timeout.connect(self.update_plot)
        self.capture_button.clicked.connect(self.start_capture)
        
        # Configuração do sistema BCI
        self.bci = create_bci_system()
        # Carregar seleção inicial do modelo
        if models:
            self.on_model_change(0)

    def start_stream(self):
        logging.info("Iniciando transmissão LSL EEG")
        streams = resolve_streams(wait_time=1.0)
        eeg_streams = [s for s in streams if s.type() == 'EEG']
        if eeg_streams:
            # Setup inlet for real-time processing
            self.inlet = StreamInlet(eeg_streams[0], max_buflen=1, max_chunklen=1, processing_flags=1)
            info = self.inlet.info()
            n_ch = info.channel_count()
            
            self.figure.set_size_inches(10, max(4, n_ch * 1.5))
            sr = int(info.nominal_srate())
            self.sr = sr
            
            # Initialize BCI system
            selected = self.model_combo.currentText() if hasattr(self, 'model_combo') else None
            model_path = os.path.join('checkpoints', selected) if selected else None
            self.bci = create_bci_system(model_path=model_path)
            self.bci.initialize_model(n_ch)
            if not self.bci.is_calibrated:
                QMessageBox.warning(self, "Aviso do Modelo",
                    "Checkpoint incompatível ou não carregado. Classificação desativada até calibração.")
              # Configure buffer sizes and display parameters
            self.window_size = 400  # window for predictions
            self.window_step = 50   # step size for predictions
            display_samples = 500    # number of samples to display
            
            # Initialize buffers with zeros
            self.buffer = [deque(maxlen=display_samples) for _ in range(n_ch)]
            for buf in self.buffer:
                for _ in range(display_samples):
                    buf.append(0)
                    
            self.sample_since_last = 0
            
            # Setup optimized plotting
            self.figure.clear()
            plt.style.use('fast')
            self.axes = []
            self.lines = []
            # Create background for blitting
            self.backgrounds = []
            for idx in range(n_ch):
                ax = self.figure.add_subplot(n_ch, 1, idx+1)
                ax.set_ylim(-100, 100)
                ax.set_xlim(0, display_samples)
                ax.set_ylabel(f"Canal {idx+1}")
                if idx < n_ch - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Amostras")
                # Initialize with zeros to avoid empty plot
                initial_data = [0] * display_samples
                line, = ax.plot(range(display_samples), initial_data, 'b-', animated=True, linewidth=1)
                self.lines.append(line)
                self.axes.append(ax)
                self.figure.tight_layout()
            
            # Configure for maximum responsiveness
            self.timer.setInterval(20)  # 50 Hz refresh rate
            self.canvas.draw()
            
            # Save backgrounds for blitting
            for ax in self.axes:
                self.backgrounds.append(self.canvas.copy_from_bbox(ax.bbox))
            
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
        
        # Get samples efficiently with a small chunk size
        chunk, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=32)
        if not chunk:
            return
              # Process samples
        for sample in chunk:
            if self.process_check.isChecked():
                sample = EEGAugmentation.process_sample(sample)
            
            # Add new samples to buffer
            for i, value in enumerate(sample):
                self.buffer[i].append(value)
        
        # Efficient plotting update
        for i, (line, ax, background) in enumerate(zip(self.lines, self.axes, self.backgrounds)):
            # Get data from buffer
            data = list(self.buffer[i])
            x = np.arange(len(data))
            
            # Restore the background and update the line
            self.canvas.restore_region(background)
            line.set_data(x, data)
            
            # Draw just the line
            ax.draw_artist(line)
            
            # Update y-axis scale if processing is enabled
            if self.process_check.isChecked():
                ymin, ymax = min(data), max(data)
                margin = (ymax - ymin) * 0.1
                ax.set_ylim(ymin - margin, ymax + margin)
            
            # Blit just this axis area
            self.canvas.blit(ax.bbox)
            
        # Ensure smooth animation
        self.canvas.flush_events()
        
        # Model prediction handling
        if hasattr(self, 'bci') and self.bci.is_calibrated:
            self.sample_since_last += len(chunk)
            current_time = time.time() * 1000
            
            # Check if we should make a prediction
            if (current_time - getattr(self, 'last_prediction_time', 0) >= 50 and 
                self.sample_since_last >= self.window_step):
                
                # Get latest window for prediction
                window_data = np.array([list(buf)[-self.window_size:] for buf in self.buffer])
                self.process_model_prediction(window_data)
                self.last_prediction_time = current_time
                self.sample_since_last = 0
        
        # Handle capture if active
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

    def process_model_prediction(self, window_data):
        """Process model prediction in an optimized way"""
        try:
            # Apply data augmentation more efficiently
            aug_data = window_data.copy()  # Start with a copy to preserve original
            if self.process_check.isChecked():
                aug_data = EEGAugmentation.time_shift(aug_data)
                aug_data = EEGAugmentation.add_gaussian_noise(aug_data)
                aug_data = EEGAugmentation.scale_amplitude(aug_data)
            
            # Get prediction
            pred, conf = self.bci.predict_movement(aug_data)
            
            # Update GUI labels
            self.pred_label.setText(f"Pred: {pred}")
            self.conf_label.setText(f"Conf: {conf:.2%}")
            
            # Log prediction asynchronously
            logging.info(f"Predição do modelo: {pred} (confiança {conf:.2%})")
            
            return pred, conf
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return None, None

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
