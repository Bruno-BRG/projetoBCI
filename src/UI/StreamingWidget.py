# Standard library imports
import os
import numpy as np
import logging
from datetime import datetime
from collections import deque

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox,
    QScrollArea, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer

# Local imports
from model.BCISystem import create_bci_system
from model.EEGAugmentation import EEGAugmentation
from pylsl import StreamInlet, resolve_streams

class StreamingWidget(QWidget):
    """Widget for live LSL streaming and real-time EEG plotting"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main layout with padding
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Plot group
        plot_group = QGroupBox("EEG Signal Monitor")
        plot_layout = QVBoxLayout()
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.canvas)
        plot_layout.addWidget(self.scroll)
        plot_group.setLayout(plot_layout)
        main_layout.addWidget(plot_group)
        
        # Status group
        status_group = QGroupBox("Prediction Status")
        status_layout = QHBoxLayout()
        self.pred_label = QLabel("Pred: N/A")
        self.conf_label = QLabel("Conf: N/A")
        self.pred_label.setStyleSheet("font-size: 14px; padding: 5px;")
        self.conf_label.setStyleSheet("font-size: 14px; padding: 5px;")
        status_layout.addWidget(self.pred_label)
        status_layout.addWidget(self.conf_label)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Control group
        control_group = QGroupBox("Stream Controls")
        control_layout = QVBoxLayout()
        
        # Stream controls row
        stream_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Streaming")
        self.stop_btn = QPushButton("Stop Streaming")
        self.stop_btn.setEnabled(False)
        stream_layout.addWidget(self.start_btn)
        stream_layout.addWidget(self.stop_btn)
        control_layout.addLayout(stream_layout)
        
        # Processing controls row
        process_layout = QHBoxLayout()
        self.process_check = QCheckBox("Enable Signal Processing")
        self.process_check.setStyleSheet("font-size: 12px; padding: 5px;")
        self.capture_button = QPushButton("Capture 5s Window")
        process_layout.addWidget(self.process_check)
        process_layout.addWidget(self.capture_button)
        control_layout.addLayout(process_layout)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        self.setLayout(main_layout)
        
        # LSL inlet and buffer setup
        self.inlet = None
        self.buffer = None
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        
        # Capture variables
        self.capturing = False
        self.capture_buffer = []
        self.sample_count = 0
        self.capture_needed = 0
        
        # Connect signals
        self.start_btn.clicked.connect(self.start_stream)
        self.stop_btn.clicked.connect(self.stop_stream)
        self.timer.timeout.connect(self.update_plot)
        self.capture_button.clicked.connect(self.start_capture)
        
        # BCI system setup
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
                    
                    # Save window plot for analysis with matching style
                    os.makedirs('window_plots', exist_ok=True)
                    plt.style.use('default')  # Reset style
                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111)
                    
                    # Plot all channels with alpha for clarity
                    for ch_idx in range(aug_data.shape[0]):
                        ax.plot(aug_data[ch_idx], alpha=0.5, linewidth=0.5)
                    
                    # Set title and labels
                    pred, conf = self.bci.predict_movement(aug_data)
                    ax.set_title(f"Real-time Window Classification\nPredicted: {pred} (Confidence: {conf:.2%})")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Amplitude")
                    
                    # Set the background color to match
                    ax.set_facecolor('lightgray')
                    fig.patch.set_facecolor('white')
                    
                    # Save and close
                    filename = datetime.now().strftime("window_plots/window_%Y%m%d_%H%M%S_%f.png")
                    plt.tight_layout()
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Update GUI labels
                    self.pred_label.setText(f"Pred: {pred}")
                    self.conf_label.setText(f"Conf: {conf:.2%}")
                    logging.info(f"Model prediction: {pred} (confidence {conf:.2%}) on window of {self.window_size} samples")
