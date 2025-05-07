import sys
import torch
import torch.optim as optim
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
from ML1 import load_local_eeg_data, create_bci_system, EEGAugmentation, MultiSubjectTest
from pylsl import StreamInlet, resolve_streams
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class CalibrationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main container with padding
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Plot group
        plot_group = QGroupBox("EEG Signal Visualization")
        plot_layout = QVBoxLayout()
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        main_layout.addWidget(plot_group)
        
        # Navigation group
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_group.setLayout(nav_layout)
        main_layout.addWidget(nav_group)
        
        # Control group
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        self.load_button = QPushButton("Load EEG Data")
        self.add_button = QPushButton("Add to Calibration")
        self.train_button = QPushButton("Train Model")
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.add_button)
        control_layout.addWidget(self.train_button)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        self.setLayout(main_layout)
        
        # Style the plot
        plt.style.use('default')
        self.figure.patch.set_facecolor('white')
        
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
        
        # Initial button states
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.train_button.setEnabled(False)

    def load_data(self):
        subject_id, ok = QInputDialog.getInt(self, "Subject ID", "Enter Subject ID:", 1, 1, 109)
        if not ok:
            return
        X, y, ch = load_local_eeg_data(subject_id, augment=False)  # Changed to no augmentation
        self.data, self.labels, self.eeg_channel = X, y, ch
        self.idx = 0
        
        # Initialize BCI model with channel count
        self.bci.initialize_model(self.eeg_channel)
        
        # Enable navigation and control buttons
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
        ax.set_title(f"Sample {self.idx+1}/{len(self.data)} - Label: {'Left' if self.labels[self.idx]==0 else 'Right'}")
        ax.set_xlabel("Time")
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
        self.train_button.setEnabled(True)  # Enable train button after adding samples
        QMessageBox.information(self, "Success", "Sample added to calibration set")

    def train_model(self):
        if self.bci and self.eeg_channel:
            try:
                self.bci.train_calibration(num_epochs=10, batch_size=4, learning_rate=1e-3)
                QMessageBox.information(self, "Success", "Model training completed successfully")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Training failed: {str(e)}")

class RealUseWidget(QWidget):
    """Widget for real-use mode: navigate samples, classify, and stream LSL"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main layout with padding
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # EEG Visualization group
        plot_group = QGroupBox("EEG Signal Visualization")
        plot_layout = QVBoxLayout()
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
        # Result group
        result_group = QGroupBox("Classification Results")
        result_layout = QVBoxLayout()
        self.true_label_label = QLabel("True: N/A")
        self.pred_label_label = QLabel("Predicted: N/A")
        self.confidence_label = QLabel("Confidence: N/A")
        for label in [self.true_label_label, self.pred_label_label, self.confidence_label]:
            label.setStyleSheet("font-size: 14px; padding: 5px;")
            result_layout.addWidget(label)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # Control group
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        self.load_button = QPushButton("Load EEG Data")
        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        self.classify_button = QPushButton("Classify Movement")
        nav_layout.addWidget(self.load_button)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.classify_button)
        control_layout.addLayout(nav_layout)
        
        # Streaming controls
        stream_layout = QHBoxLayout()
        self.stream_start_button = QPushButton("Start Stream")
        self.stream_stop_button = QPushButton("Stop Stream")
        stream_layout.addWidget(self.stream_start_button)
        stream_layout.addWidget(self.stream_stop_button)
        control_layout.addLayout(stream_layout)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
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
        
        # Also connect stream buttons
        self.stream_start_button.clicked.connect(self.start_stream)
        self.stream_stop_button.clicked.connect(self.stop_stream)
        
        # Initial button states
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.classify_button.setEnabled(False)
        self.stream_stop_button.setEnabled(False)

    def load_data(self):
        subject_id, ok = QInputDialog.getInt(self, "Subject ID", "Enter Subject ID:", 1, 1, 109)
        if not ok:
            return
        X, y, ch = load_local_eeg_data(subject_id, augment=False)  # Changed to no augmentation
        self.data, self.labels, self.eeg_channel = X, y, ch
        self.idx = 0
        
        # Initialize BCI model with channel count
        self.bci.initialize_model(self.eeg_channel)
        
        # Enable navigation and control buttons
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)
        self.classify_button.setEnabled(self.bci.is_calibrated)  # Only enable if model is calibrated
        
        self.update_plot()

    def update_plot(self):
        if self.data is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        sample = self.data[self.idx]
        for ch in range(sample.shape[0]):
            ax.plot(sample[ch], alpha=0.5, linewidth=0.5)
        ax.set_title(f"Sample {self.idx+1}/{len(self.data)} - Label: {'Left' if self.labels[self.idx]==0 else 'Right'}")
        ax.set_xlabel("Time")
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

    def start_stream(self):
        """Handle start of LSL stream"""
        streams = resolve_streams(wait_time=1.0)
        eeg_streams = [s for s in streams if s.type() == 'EEG']
        if eeg_streams:
            # Enable stop button and disable start button
            self.stream_start_button.setEnabled(False)
            self.stream_stop_button.setEnabled(True)
            QMessageBox.information(self, "Stream Started", "Successfully connected to EEG stream")
        else:
            QMessageBox.warning(self, "No Stream Found", "Could not find any EEG stream")

    def stop_stream(self):
        """Handle stop of LSL stream"""
        # Reset stream controls
        self.stream_start_button.setEnabled(True)
        self.stream_stop_button.setEnabled(False)
        QMessageBox.information(self, "Stream Stopped", "Disconnected from EEG stream")

    def classify_movement(self):
        if self.data is None:
            return
        if not self.bci.is_calibrated:
            QMessageBox.warning(self, "Error", "Model needs to be calibrated first")
            return
            
        try:
            pred, conf = self.bci.predict_movement(self.data[self.idx])
            self.true_label_label.setText(f"True: {'Left' if self.labels[self.idx]==0 else 'Right'}")
            self.pred_label_label.setText(f"Predicted: {pred}")
            self.confidence_label.setText(f"Confidence: {conf:.2%}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Classification failed: {str(e)}")

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

class TestWidget(QWidget):
    """Widget for multi-subject model testing"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main layout with padding
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Plot group for training curves
        plot_group = QGroupBox("Training Metrics")
        plot_layout = QVBoxLayout()
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
        # Status group
        status_group = QGroupBox("Test Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Ready to start testing")
        self.progress_label = QLabel("")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Control group
        control_group = QGroupBox("Test Controls")
        control_layout = QHBoxLayout()
        self.start_test_button = QPushButton("Start Multi-Subject Test")
        self.start_test_button.clicked.connect(self.start_test)
        control_layout.addWidget(self.start_test_button)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        self.setLayout(layout)
        
        # Initialize test system
        self.test_system = None
        self.figure.clear()
        self.canvas.draw()

    def update_plot(self, history):
        """Update the plot with new training metrics"""
        self.figure.clear()
        
        # Create two subplots
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        
        # Plot losses
        epochs = range(1, len(history['train_losses']) + 1)
        ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(epochs, history['train_accs'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_accs'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        ax2.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()

    def start_test(self):
        """Start the multi-subject testing process"""
        self.start_test_button.setEnabled(False)
        self.status_label.setText("Preparing datasets...")
        self.progress_label.setText("This may take a few minutes...")
        
        try:
            # Initialize test system
            self.test_system = MultiSubjectTest(train_samples=20, test_samples=10)
            
            # Run training and evaluation
            history = self.test_system.train_and_evaluate(
                num_epochs=20,
                batch_size=32,
                learning_rate=1e-3
            )
            
            # Update plot with results
            self.update_plot(history)
            
            # Update status
            final_train_acc = history['train_accs'][-1]
            final_val_acc = history['val_accs'][-1]
            self.status_label.setText("Testing completed successfully!")
            self.progress_label.setText(
                f"Final Results:\n"
                f"Training Accuracy: {final_train_acc:.2%}\n"
                f"Validation Accuracy: {final_val_acc:.2%}"
            )
            
        except Exception as e:
            self.status_label.setText("Error during testing")
            self.progress_label.setText(str(e))
        
        finally:
            self.start_test_button.setEnabled(True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI System for Post-Stroke Rehabilitation")
        self.resize(1200, 800)
        
        # Set window style to be more like Java Swing
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QWidget {
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #e0e0e0;
                border: 2px solid #a0a0a0;
                border-radius: 3px;
                min-height: 25px;
                padding: 5px;
                color: #000000;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                border: 2px solid #c0c0c0;
                color: #808080;
            }
            QLabel {
                color: #000000;
                font-size: 12px;
                padding: 2px;
            }
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #505050;
            }
        """)

        # Central widget with stacked layout for modes
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #f0f0f0; padding: 10px;")
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Add a header panel
        header = QWidget()
        header.setStyleSheet("""
            QWidget {
                background-color: #4a6984;
                border-radius: 5px;
                margin: 0px;
                padding: 10px;
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_label = QLabel("BCI System Control Panel")
        header_layout.addWidget(header_label)
        main_layout.addWidget(header)

        # Toolbar for mode selection styled like old Java tabs
        tab_bar = QWidget()
        tab_bar.setStyleSheet("""
            QWidget {
                background-color: #e8e8e8;
                border-bottom: 1px solid #c0c0c0;
            }
            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #c0c0c0;
                border-bottom: none;
                border-radius: 3px 3px 0 0;
                min-width: 100px;
                padding: 5px 15px;
            }
            QPushButton:checked {
                background-color: #ffffff;
                border-bottom: 1px solid #ffffff;
            }
        """)
        tab_layout = QHBoxLayout(tab_bar)
        tab_layout.setSpacing(0)
        tab_layout.setContentsMargins(10, 5, 10, 0)
        
        # Create tab buttons
        self.calib_button = QPushButton("Calibration")
        self.real_button = QPushButton("Real Use")
        self.stream_button = QPushButton("Streaming")
        self.test_button = QPushButton("Testing")
        self.calib_button.setCheckable(True)
        self.real_button.setCheckable(True)
        self.stream_button.setCheckable(True)
        self.test_button.setCheckable(True)
        self.calib_button.setChecked(True)
        
        tab_layout.addWidget(self.calib_button)
        tab_layout.addWidget(self.real_button)
        tab_layout.addWidget(self.stream_button)
        tab_layout.addWidget(self.test_button)
        tab_layout.addStretch()
        main_layout.addWidget(tab_bar)

        # Stacked widget for content
        self.stacked = QStackedWidget()
        self.stacked.setStyleSheet("""
            QStackedWidget {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-top: none;
            }
        """)
        self.calib_widget = CalibrationWidget()
        self.real_widget = RealUseWidget()
        self.stream_widget = StreamingWidget()
        self.test_widget = TestWidget()
        self.stacked.addWidget(self.calib_widget)
        self.stacked.addWidget(self.real_widget)
        self.stacked.addWidget(self.stream_widget)
        self.stacked.addWidget(self.test_widget)
        main_layout.addWidget(self.stacked)

        # Connect tab buttons
        self.calib_button.clicked.connect(lambda: self.switch_mode(0))
        self.real_button.clicked.connect(lambda: self.switch_mode(1))
        self.stream_button.clicked.connect(lambda: self.switch_mode(2))
        self.test_button.clicked.connect(lambda: self.switch_mode(3))

    def switch_mode(self, index: int):
        self.stacked.setCurrentIndex(index)
        # Update tab button states
        self.calib_button.setChecked(index == 0)
        self.real_button.setChecked(index == 1)
        self.stream_button.setChecked(index == 2)
        self.test_button.setChecked(index == 3)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    
    # Remove dark palette settings and use classic light theme
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

class ModelWrapper:
    # ...existing ModelWrapper code...

    def configure_optimizers(self):
        # Added weight_decay for L2 regularization
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01  # L2 regularization factor
        )
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(self.max_epoch * 0.25),
                    int(self.max_epoch * 0.5),
                    int(self.max_epoch * 0.75),
                ],
                gamma=0.1
            ),
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]