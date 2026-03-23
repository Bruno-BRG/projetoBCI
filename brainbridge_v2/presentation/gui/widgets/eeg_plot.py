import numpy as np
from collections import deque
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class EEGPlotWidget(QWidget):
    """Widget para plotar dados EEG em tempo real"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_plot()
        
        # Buffer para dados
        self.data_buffer = deque(maxlen=1000)  # 8 segundos a 125 Hz
        self.time_buffer = deque(maxlen=1000)
        self.current_time = 0
        
        # Timer para atualizar plot
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 20 FPS
        
    def setup_ui(self):
        """Configura a interface do widget - Escala automática, todos os canais"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Área do plot - preenche todo o espaço
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)

        self.setLayout(layout)
        
    def setup_plot(self):
        """Configura o plot inicial"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, 8)  # 8 segundos
        self.ax.set_ylim(-100, 100)
        self.ax.set_xlabel('Tempo (s)')
        self.ax.set_ylabel('Amplitude (µV)')
        self.ax.set_title('Dados EEG em Tempo Real')
        self.ax.grid(True, alpha=0.3)
        
        # Linhas para cada canal
        self.lines = []
        colors = plt.cm.tab10(np.linspace(0, 1, 16))
        
        for i in range(16):
            line, = self.ax.plot([], [], color=colors[i], linewidth=0.8, 
                               label=f'Canal {i}', alpha=0.7)
            self.lines.append(line)
            
        self.canvas.draw()
        
    def add_data(self, eeg_data: np.ndarray):
        """Adiciona novos dados EEG"""
        if len(eeg_data) == 16:  # 16 canais
            self.data_buffer.append(eeg_data)
            self.time_buffer.append(self.current_time)
            self.current_time += 1/125  # 125 Hz
            
    def update_plot(self):
        """Atualiza o plot com novos dados - Todos os canais com escala automática"""
        if len(self.data_buffer) == 0:
            return

        times = np.array(self.time_buffer)
        data = np.array(self.data_buffer)

        if len(times) < 2:
            return

        # Janela de 8 segundos
        current_time = times[-1]
        window_start = max(0, current_time - 8)
        mask = times >= window_start
        windowed_times = times[mask] - window_start
        windowed_data = data[mask]

        # Mostrar todos os canais com offset vertical
        if len(windowed_data) > 0:
            for i in range(16):
                y_data = windowed_data[:, i] + i * 100
                self.lines[i].set_data(windowed_times, y_data)
                self.lines[i].set_visible(True)

            self.ax.set_ylim(-100, 1600)
            self.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)

        self.ax.set_xlim(0, 8)
        self.canvas.draw()