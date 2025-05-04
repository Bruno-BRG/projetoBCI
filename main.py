import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QStackedWidget, QProgressBar, QSlider, QCheckBox)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QPointF
from PySide6.QtGui import QPainter, QColor, QPen
import numpy as np
from collections import deque
import queue
from pylsl import StreamInlet, resolve_streams
import time
from ML1 import create_bci_system, load_local_eeg_data

class EEGDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 800)  # Increased height to accommodate more channels
        self.channel_data = [deque(maxlen=1000) for _ in range(16)]  # 16 channels
        self.time_window = 5.0  # 5 seconds window
        self.update_interval = 50  # 50ms = 20Hz update rate
        self.pixels_per_second = 200  # Pixels per second for x-axis
        self.scale_factor = 0.5  # Initial scale factor for y-axis
        self.auto_scale = True  # Enable auto-scaling by default
        self.channel_spacing = 100  # Base amplitude spacing between channels
        
        # Extended color palette for 16 channels
        self.channel_colors = [
            QColor("#4B0082"),  # Purple
            QColor("#0000FF"),  # Blue
            QColor("#00FF00"),  # Green
            QColor("#FFFF00"),  # Yellow
            QColor("#FFA500"),  # Orange
            QColor("#FF0000"),  # Red
            QColor("#FF69B4"),  # Pink
            QColor("#808080"),  # Gray
            QColor("#800080"),  # Deep Purple
            QColor("#000080"),  # Navy Blue
            QColor("#008000"),  # Dark Green
            QColor("#FFD700"),  # Gold
            QColor("#FF4500"),  # Orange Red
            QColor("#DC143C"),  # Crimson
            QColor("#FF1493"),  # Deep Pink
            QColor("#A9A9A9"),  # Dark Gray
        ]
        
        # Set black background
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(palette)
        
        # Update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_interval)

    def auto_scale_signals(self):
        """Calculate appropriate scaling based on current signal amplitudes"""
        if not any(self.channel_data):
            return
            
        max_amplitudes = []
        for channel_data in self.channel_data:
            if channel_data:
                amplitude = max(abs(min(channel_data)), abs(max(channel_data)))
                max_amplitudes.append(amplitude)
            else:
                max_amplitudes.append(1)
                
        if not max_amplitudes:
            return
            
        max_amp = max(max_amplitudes)
        if max_amp > 0:
            # Scale to fit within 80% of the channel height
            channel_height = self.height() / 8
            desired_peak = channel_height * 0.4  # Use 40% of channel height
            self.scale_factor = desired_peak / max_amp
    
    def add_samples(self, samples):
        """Add new samples for each channel"""
        if not samples:
            return
            
        for ch in range(min(len(self.channel_data), len(samples[0]))):
            for sample in samples:
                self.channel_data[ch].append(sample[ch])
    
    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Calculate dimensions
            w = self.width()
            h = self.height()
            channel_height = h / 16  # Adjusted for 16 channels
            
            if self.auto_scale:
                self.auto_scale_signals()
            
            # Draw time grid
            painter.setPen(QPen(QColor("#333333"), 1))  # Dark gray grid
            time_step = int(self.pixels_per_second)  # Convert to int for grid spacing
            for x in range(w, 0, -time_step):
                painter.drawLine(x, 0, x, h)
            
            # Draw zero lines and channel labels
            for ch in range(16):  # Updated to 16 channels
                y_base = (ch + 0.5) * channel_height
                painter.setPen(QPen(Qt.white, 1))
                painter.drawLine(0, y_base, w, y_base)
                painter.drawText(10, int(y_base - 10), f"Ch{ch+1}")
            
            # Draw EEG signals
            for ch in range(16):  # Updated to 16 channels
                if not self.channel_data[ch]:
                    continue
                    
                painter.setPen(QPen(self.channel_colors[ch], 2))
                y_base = (ch + 0.5) * channel_height
                
                points = []
                data = list(self.channel_data[ch])
                for i, value in enumerate(data):
                    x = w - (len(data) - i)
                    # Apply scaling factor to keep signals within their channels
                    y = y_base - (value * self.scale_factor)
                    points.append(QPointF(x, y))
                
                # Draw the signal
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        painter.drawLine(points[i], points[i + 1])
            
            # Draw time labels
            painter.setPen(Qt.white)
            for i in range(6):
                x = w - (i * time_step)
                if x > 0:
                    painter.drawText(x - 20, h - 5, f"-{i}s")
                    
        finally:
            painter.end()
class LSLThread(QThread):
    data_ready = Signal(object, object)
    connection_status = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.inlet = None
        self.running = False
    
    def init_stream(self):
        try:
            print("Looking for an EEG stream...")
            streams = resolve_streams()
            
            if not streams:
                self.connection_status.emit("No LSL streams found! Is OpenBCI GUI running?")
                return False
            
            # Filter for EEG streams
            eeg_streams = [stream for stream in streams if stream.type() == 'EEG']
            
            if not eeg_streams:
                self.connection_status.emit("No EEG streams found! Check OpenBCI GUI settings.")
                return False
            
            selected_stream = eeg_streams[0]
            self.inlet = StreamInlet(selected_stream)
            
            stream_info = (
                f"Connected to LSL stream:\n"
                f"Name: {selected_stream.name()}\n"
                f"Type: {selected_stream.type()}\n"
                f"Channels: {selected_stream.channel_count()}\n"
                f"Rate: {selected_stream.nominal_srate()} Hz"
            )
            self.connection_status.emit(stream_info)
            return True
            
        except Exception as e:
            self.connection_status.emit(f"Error connecting to LSL stream: {str(e)}")
            return False
    
    def run(self):
        self.running = True
        while self.running and self.inlet:
            chunk, timestamps = self.inlet.pull_chunk(timeout=0.1)
            if chunk:
                self.data_ready.emit(chunk, timestamps)
            time.sleep(0.001)  # Small sleep to prevent CPU overuse
    
    def stop(self):
        self.running = False
        self.wait()

class BCIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI System")
        self.setMinimumSize(1200, 800)
        
        # Initialize system components
        self.bci_system = create_bci_system()
        self.lsl_thread = LSLThread()
        self.lsl_thread.data_ready.connect(self.handle_eeg_data)
        self.lsl_thread.connection_status.connect(self.update_status)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Add controls at the top
        controls = QHBoxLayout()
        self.connect_btn = QPushButton("Start Streaming")
        self.connect_btn.clicked.connect(self.toggle_streaming)
        controls.addWidget(self.connect_btn)
        
        # Scale controls
        scale_controls = QHBoxLayout()
        self.auto_scale_cb = QCheckBox("Auto Scale")
        self.auto_scale_cb.setChecked(True)
        self.auto_scale_cb.stateChanged.connect(self.toggle_auto_scale)
        scale_controls.addWidget(self.auto_scale_cb)
        
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(1, 1000)  # Wider range for manual scaling
        self.scale_slider.setValue(500)
        self.scale_slider.valueChanged.connect(self.update_scale)
        self.scale_slider.setEnabled(not self.auto_scale_cb.isChecked())
        scale_controls.addWidget(QLabel("Manual Scale:"))
        scale_controls.addWidget(self.scale_slider)
        controls.addLayout(scale_controls)
        
        self.time_window_slider = QSlider(Qt.Horizontal)
        self.time_window_slider.setRange(1, 10)
        self.time_window_slider.setValue(5)
        self.time_window_slider.valueChanged.connect(self.update_time_window)
        controls.addWidget(QLabel("Time Window (s):"))
        controls.addWidget(self.time_window_slider)
        
        layout.addLayout(controls)
        
        # Add EEG display
        self.eeg_display = EEGDisplay()
        layout.addWidget(self.eeg_display)
        
        # Add status label
        self.status_label = QLabel("Not connected")
        layout.addWidget(self.status_label)
    
    def toggle_streaming(self):
        if not self.lsl_thread.running:
            if self.lsl_thread.init_stream():
                self.connect_btn.setText("Stop Streaming")
                self.lsl_thread.start()
            else:
                self.status_label.setText("Failed to connect to LSL stream")
        else:
            self.lsl_thread.stop()
            self.connect_btn.setText("Start Streaming")
            self.status_label.setText("Disconnected")
    
    @Slot(object, object)
    def handle_eeg_data(self, chunk, timestamps):
        self.eeg_display.add_samples(chunk)
    
    @Slot(str)
    def update_status(self, message):
        self.status_label.setText(message)
    
    def toggle_auto_scale(self, state):
        self.eeg_display.auto_scale = bool(state)
        self.scale_slider.setEnabled(not bool(state))
    
    def update_scale(self, value):
        if not self.eeg_display.auto_scale:
            # Convert slider value to a reasonable scaling factor
            self.eeg_display.scale_factor = value / 1000.0  # Scale from 0.001 to 1.0
    
    def update_time_window(self, value):
        self.eeg_display.time_window = value
        self.eeg_display.pixels_per_second = self.width() / value
    
    def closeEvent(self, event):
        self.lsl_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = BCIMainWindow()
    window.show()
    
    sys.exit(app.exec())