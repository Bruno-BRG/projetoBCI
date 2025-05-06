import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QInputDialog,
    QScrollArea, QCheckBox, QMessageBox, QGroupBox
)
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QStyleFactory
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from ML1 import load_local_eeg_data, create_bci_system, EEGAugmentation
from pylsl import StreamInlet, resolve_streams
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class CalibrationWidget(QWidget):
    """Widget for calibration mode: load data, navigate samples, add to calibration, and train model"""  
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.load_button = QPushButton("Load EEG Data")
        self.prev_button = QPushButton("Previous Sample")
        self.next_button = QPushButton("Next Sample")
        self.add_button = QPushButton("Add to Calibration")
        self.train_button = QPushButton("Train Model")
        # Plot area
        self.figure = plt.Figure(figsize=(5,3)) if False else plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        layout.addWidget(self.load_button)
        layout.addWidget(self.prev_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.add_button)
        layout.addWidget(self.train_button)
        self.setLayout(layout)
        # Data storage
        self.data = None
        self.labels = None
        self.idx = 0
        self.eeg_channel = None
        self.bci = create_bci_system()
        # Connect signals
        self.load_button.clicked.connect(self.load_data)
        self.prev_button.clicked.connect(self.prev_sample)
        self.next_button.clicked.connect(self.next_sample)
        self.add_button.clicked.connect(self.add_to_calibration)
        self.train_button.clicked.connect(self.train_model)

    def load_data(self):
        subject_id, ok = QInputDialog.getInt(self, "Subject ID", "Enter Subject ID:", 1, 1, 109)
        if not ok:
            return
        X, y, ch = load_local_eeg_data(subject_id, augment=False)
        self.data, self.labels, self.eeg_channel = X, y, ch
        self.idx = 0
        self.update_plot()

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        sample = self.data[self.idx]
        for ch in range(sample.shape[0]):
            ax.plot(sample[ch], alpha=0.5)
        ax.set_title(f"Sample {self.idx+1}/{len(self.data)} - Label: {self.labels[self.idx]}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
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

    def train_model(self):
        if self.bci and self.eeg_channel:
            self.bci.initialize_model(self.eeg_channel)
            # Launch training in separate thread or directly
            self.bci.train_calibration(num_epochs=10, batch_size=4, learning_rate=1e-3)
            # Could display training plot

class RealUseWidget(QWidget):
    """Widget for real-use mode: navigate samples, classify, and stream LSL"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.load_button = QPushButton("Load EEG Data")
        self.prev_button = QPushButton("Previous Sample")
        self.next_button = QPushButton("Next Sample")
        self.classify_button = QPushButton("Classify Movement")
        # Plot area
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        # Result labels
        self.true_label_label = QLabel("True: N/A")
        self.pred_label_label = QLabel("Predicted: N/A")
        self.confidence_label = QLabel("Confidence: N/A")
        # Streaming controls (to be implemented)
        self.stream_start_button = QPushButton("Start Stream")
        self.stream_stop_button = QPushButton("Stop Stream")
        # Add widgets
        layout.addWidget(self.load_button)
        layout.addWidget(self.canvas)
        layout.addWidget(self.true_label_label)
        layout.addWidget(self.pred_label_label)
        layout.addWidget(self.confidence_label)
        hl = QHBoxLayout()
        hl.addWidget(self.prev_button)
        hl.addWidget(self.next_button)
        hl.addWidget(self.classify_button)
        layout.addLayout(hl)
        layout.addWidget(self.stream_start_button)
        layout.addWidget(self.stream_stop_button)
        self.setLayout(layout)
        # Data storage
        self.data = None
        self.labels = None
        self.idx = 0
        self.eeg_channel = None
        self.bci = create_bci_system()
        # Connect signals
        self.load_button.clicked.connect(self.load_data)
        self.prev_button.clicked.connect(self.prev_sample)
        self.next_button.clicked.connect(self.next_sample)
        self.classify_button.clicked.connect(self.classify_movement)
        # Stream signals (to implement)

    def load_data(self):
        subject_id, ok = QInputDialog.getInt(self, "Subject ID", "Enter Subject ID:", 1, 1, 109)
        if not ok:
            return
        X, y, ch = load_local_eeg_data(subject_id, augment=False)
        self.data, self.labels, self.eeg_channel = X, y, ch
        self.idx = 0
        # Initialize model with channel count
        self.bci.initialize_model(self.eeg_channel)
        self.update_plot()

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        sample = self.data[self.idx]
        for ch in range(sample.shape[0]):
            ax.plot(sample[ch], alpha=0.5)
        ax.set_title(f"Sample {self.idx+1}/{len(self.data)} - True: {'Left' if self.labels[self.idx]==0 else 'Right'}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
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

    def classify_movement(self):
        if self.data is None or not self.bci.is_calibrated:
            return
        sample = self.data[self.idx]
        pred, conf = self.bci.predict_movement(sample)
        self.true_label_label.setText(f"True: {'Left' if self.labels[self.idx]==0 else 'Right'}")
        self.pred_label_label.setText(f"Predicted: {pred}")
        self.confidence_label.setText(f"Confidence: {conf:.2%}")

class StreamingWidget(QWidget):
    """Widget for live LSL streaming and real-time EEG plotting"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Main panel for streaming widget
        main_group = QGroupBox("Streaming Panel")
        layout = QVBoxLayout(main_group)
        # Plot area with scrollable canvas
        plot_group = QGroupBox("EEG Plot")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.canvas)
        plot_layout.addWidget(self.scroll)
        layout.addWidget(plot_group)
        
        # Control buttons and options
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)
        self.start_btn = QPushButton("Start Streaming")
        self.stop_btn = QPushButton("Stop Streaming")
        self.stop_btn.setEnabled(False)
        self.process_check = QCheckBox("Enable Additional Processing")
        self.process_check.setChecked(False)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.process_check)
        self.capture_button = QPushButton("Capture 5s")
        control_layout.addWidget(self.capture_button)
        layout.addWidget(control_group)
        # Prediction display
        self.pred_label = QLabel("Pred: N/A")
        self.conf_label = QLabel("Conf: N/A")
         
        layout.addWidget(self.pred_label)
        layout.addWidget(self.conf_label)
        self.setLayout(layout)
        # Add main_group as widget layout
        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(main_group)
        self.setLayout(outer_layout)
        
        # LSL inlet and buffer
        self.inlet = None
        self.buffer = None
        self.timer = QTimer(self)
        self.timer.setInterval(20)  # ms for higher frame rate
        # Capture state variables
        self.capturing = False
        self.capture_buffer = []
        self.sample_count = 0
        self.capture_needed = 0
        
        # Signals
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)
        self.timer.timeout.connect(self.update_plot)
        self.capture_button.clicked.connect(self.start_capture)
        # BCISystem for real-time inference
        self.bci = create_bci_system()
        self.window_size = None
        self.window_step = None
        self.sample_since_last = 0
        
    def start_stream(self):
        logging.info("Starting LSL EEG stream")
        streams = resolve_streams(wait_time=1.0)
        eeg_streams = [s for s in streams if s.type() == 'EEG']
        if eeg_streams:
            self.inlet = StreamInlet(eeg_streams[0])
            info = self.inlet.info()
            n_ch = info.channel_count()
            # adjust figure size for channel count (height inches per channel)
            self.figure.set_size_inches(10, max(4, n_ch * 1.5))
            sr = int(info.nominal_srate())
            self.sr = sr  # store sample rate for capture
            # initialize model for real-time classification
            self.bci.initialize_model(n_ch)
            if not self.bci.is_calibrated:
                QMessageBox.warning(self, "Model Warning", \
                    "Checkpoint incompatible or not loaded. Classification disabled until calibration.")
            # set sliding window of 1s and prediction every 1s
            self.window_size = int(self.sr * 1.0)
            self.window_step = int(self.sr * 1.0)
            self.sample_since_last = 0
            buf_len = 500  # fixed number of samples to display
            from collections import deque
            self.buffer = [deque(maxlen=buf_len) for _ in range(n_ch)]
            # Prepare figure for efficient real-time update
            self.figure.clear()
            self.axes = []
            self.lines = []
            for idx in range(n_ch):
                ax = self.figure.add_subplot(n_ch, 1, idx+1)
                ax.set_ylim(-100, 100)  # static amplitude range
                ax.set_xlim(0, buf_len)  # fixed sample window on x-axis
                ax.set_ylabel(f"Ch {idx+1}")
                if idx < n_ch - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("Samples")
                line, = ax.plot([], [], color='blue')
                self.lines.append(line)
                self.axes.append(ax)
            self.figure.tight_layout()
            self.canvas.draw()
            # ensure canvas is tall enough and enable scrolling
            height_px = int(self.figure.get_figheight() * self.figure.get_dpi())
            self.canvas.setMinimumHeight(height_px)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.timer.start()
            logging.info(f"Stream started: {n_ch} channels at {self.sr} Hz")

    def start_capture(self):
        """Start capturing next 5 seconds of EEG data"""
        if not self.inlet or not hasattr(self, 'sr'):
            return
        self.capture_buffer = []
        self.sample_count = 0
        self.capture_needed = int(self.sr * 5)
        self.capturing = True
        self.capture_button.setEnabled(False)
        logging.info("Started 5s data capture")

    def stop_stream(self):
        self.timer.stop()
        self.inlet = None
        self.buffer = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        logging.info("Stopped EEG stream")

    def update_plot(self):
        if not self.inlet:
            return
        chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=32)
        if chunk:
            # append new samples
            for sample in chunk:
                for i, val in enumerate(sample):
                    if self.process_check.isChecked():
                        # Only apply processing if enabled
                        # You can add your processing here if needed in the future
                        pass
                    self.buffer[i].append(val)
                    
            # update each line object with new buffer data
            for idx, line in enumerate(self.lines):
                data = list(self.buffer[idx])
                line.set_data(range(len(data)), data)
            
            # Update axes
            max_samples = self.buffer[0].maxlen
            for ax in self.axes:
                ax.set_xlim(0, max_samples)
                if not self.process_check.isChecked():
                    # Keep fixed y-axis for raw data
                    ax.set_ylim(-100, 100)
                else:
                    # Allow autoscaling if processing is enabled
                    ax.relim()
                    ax.autoscale_view(scaley=True)
            
            # redraw canvas efficiently
            self.canvas.draw_idle()

        # Capture logic: accumulate data for 5 seconds
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
                logging.info(f"Saved captured data to {filename}")

        # Real-time classification with sliding window
        if hasattr(self, 'bci') and self.bci.is_calibrated and self.buffer and self.window_size:
            if len(self.buffer[0]) >= self.window_size:
                self.sample_since_last += len(chunk)
                if self.sample_since_last >= self.window_step:
                    self.sample_since_last = 0
                    # extract last window_size samples
                    window_data = np.array([list(self.buffer[i])[-self.window_size:] for i in range(len(self.buffer))])
                    # apply data augmentation before inference
                    aug_data = EEGAugmentation.time_shift(window_data)
                    aug_data = EEGAugmentation.add_gaussian_noise(aug_data)
                    aug_data = EEGAugmentation.scale_amplitude(aug_data)
                    # Save window plot for analysis
                    os.makedirs('window_plots', exist_ok=True)
                    fig, axs = plt.subplots(len(self.buffer), 1, figsize=(10, len(self.buffer)*2))
                    for i, ax in enumerate(axs):
                        ax.plot(aug_data[i], color='blue')
                        ax.set_ylabel(f'Ch {i+1}')
                    fig.tight_layout()
                    filename = datetime.now().strftime("window_plots/window_%Y%m%d_%H%M%S_%f.png")
                    fig.savefig(filename)
                    plt.close(fig)
                    logging.info(f"Saved window plot to {filename}")
                    pred, conf = self.bci.predict_movement(aug_data)
                    self.pred_label.setText(f"Pred: {pred}")
                    self.conf_label.setText(f"Conf: {conf:.2%}")
                    logging.info(f"Model prediction: {pred} (confidence {conf:.2%}) on window of {self.window_size} samples")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI System for Post-Stroke Rehabilitation")
        self.resize(1200, 800)

        # Central widget with stacked layout for modes
        self.stacked = QStackedWidget()
        self.calib_widget = CalibrationWidget()
        self.real_widget = RealUseWidget()
        self.stream_widget = StreamingWidget()
        self.stacked.addWidget(self.calib_widget)
        self.stacked.addWidget(self.real_widget)
        self.stacked.addWidget(self.stream_widget)
        self.setCentralWidget(self.stacked)

        # Toolbar for mode selection
        self.toolbar = self.addToolBar("Mode")
        self.calib_action = self.toolbar.addAction("Calibration")
        self.real_action = self.toolbar.addAction("Real Use")
        self.stream_action = self.toolbar.addAction("Streaming")
        self.calib_action.triggered.connect(lambda: self.switch_mode(0))
        self.real_action.triggered.connect(lambda: self.switch_mode(1))
        self.stream_action.triggered.connect(lambda: self.switch_mode(2))

    def switch_mode(self, index: int):
        self.stacked.setCurrentIndex(index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Set Fusion style for consistency
    app.setStyle(QStyleFactory.create('Fusion'))
    # Dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53,53,53))
    dark_palette.setColor(QPalette.WindowText, QColor(255,255,255))
    dark_palette.setColor(QPalette.Base, QColor(25,25,25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53,53,53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255,255,255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255,255,255))
    dark_palette.setColor(QPalette.Text, QColor(255,255,255))
    dark_palette.setColor(QPalette.Button, QColor(53,53,53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255,255,255))
    dark_palette.setColor(QPalette.BrightText, QColor(255,0,0))
    app.setPalette(dark_palette)
    # Global style sheet
    app.setStyleSheet("""
        QPushButton {
            min-height: 36px;
            font-size: 14px;
            background-color: #29a19c;
            color: #ffffff;
            border-radius: 6px;
            margin: 4px;
        }
        QPushButton:disabled {
            background-color: #555555;
        }
        QLabel {
            font-size: 14px;
            margin: 2px;
        }
        QScrollArea, QWidget {
            background-color: #2e2e2e;
        }
    """)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())