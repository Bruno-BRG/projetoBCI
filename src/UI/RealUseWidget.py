# Standard library imports

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QInputDialog,
    QMessageBox, QGroupBox, QComboBox, QFileDialog
)
import os
import numpy as np

# Local imports
from model.BCISystem import create_bci_system
from model.EEGDataLoader import load_local_eeg_data
from model.EEGFilter import EEGFilter
from pylsl import StreamInlet, resolve_streams


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
        self.filter = EEGFilter()  # always use the bandpass filter
        
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
        
        # Model selection dropdown
        model_group = QGroupBox("Select Model")
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        # Populate with .pth files in checkpoints dir
        os.makedirs('checkpoints', exist_ok=True)
        models = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        self.model_combo.addItems(models)
        self.model_combo.currentIndexChanged.connect(self.on_model_change)
        # add a browse button to select model file
        self.browse_btn = QPushButton("Browse Model...")
        self.browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.browse_btn)
        model_group.setLayout(model_layout)
        layout.insertWidget(1, model_group)  # insert after plot_group
        
        # Load initial model
        if models:
            self.on_model_change(0)
        else:
            self.bci = create_bci_system(model_path=None)

    def load_data(self):
        subject_id, ok = QInputDialog.getInt(self, "Subject ID", "Enter Subject ID:", 1, 1, 109)
        if not ok:
            return
        X, y, ch = load_local_eeg_data(subject_id, augment=False)  # Changed to no augmentation
        self.data, self.labels, self.eeg_channel = X, y, ch
        self.filter = EEGFilter()  # reinitialize for loaded offline data
        self.idx = 0
        
        # Only initialize model if we don't already have a calibrated one
        if not self.bci.is_calibrated:
            # Initialize BCI model with channel count
            self.bci.initialize_model(self.eeg_channel)
        
        # Enable navigation buttons
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)
        self.classify_button.setEnabled(self.bci.is_calibrated)
        
        self.update_plot()
        
        # Update classification result labels
        self.true_label_label.setText("True: N/A")
        self.pred_label_label.setText("Predicted: N/A")
        self.confidence_label.setText("Confidence: N/A")

    def update_plot(self):
        if self.data is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        sample = self.data[self.idx]
        # Apply offline filter to the current sample
        filtered = self.filter.filter_offline(sample)
        # Plot each channel of filtered data
        for ch_data in filtered:
            ax.plot(ch_data, alpha=0.5, linewidth=0.5)
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
        # Initialize streaming filter
        self.filter = EEGFilter(sfreq=self.eeg_channel or 250.0)
        streams = resolve_streams(wait_time=1.0)
        eeg_streams = [s for s in streams if s.type() == 'EEG']
        if eeg_streams:
            # Pre-filter streaming data in update_plot loop
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
            # filter the sample before classification
            sample = self.data[self.idx]
            filtered_sample = self.filter.filter_offline(sample)
            pred, conf = self.bci.predict_movement(filtered_sample)
            self.true_label_label.setText(f"True: {'Left' if self.labels[self.idx]==0 else 'Right'}")
            self.pred_label_label.setText(f"Predicted: {pred}")
            self.confidence_label.setText(f"Confidence: {conf:.2%}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Classification failed: {str(e)}")

    def on_model_change(self, index):
        """Handle model selection change"""
        model_name = self.model_combo.currentText()
        if model_name:
            model_path = os.path.join('checkpoints', model_name)
            # create system with selected checkpoint
            self.bci = create_bci_system(model_path=model_path)
            # if data channels already known, initialize model to load weights
            if hasattr(self, 'eeg_channel') and self.eeg_channel:
                self.bci.initialize_model(self.eeg_channel)
                if self.bci.is_calibrated:
                    QMessageBox.information(self, "Model Loaded", f"Loaded model: {model_name}")
                    self.classify_button.setEnabled(True)
                else:
                    QMessageBox.warning(self, "Load Failed", f"Checkpoint incompatible: {model_name}")
                    self.classify_button.setEnabled(False)
            else:
                QMessageBox.information(self, "Model Selected", f"Model selected: {model_name}\nLoad data to apply it.")
        else:
            self.bci = create_bci_system(model_path=None)
            self.classify_button.setEnabled(False)
            QMessageBox.information(self, "No Model", "Using default uncalibrated system.")
    
    def browse_model(self):
        """Open file dialog to select a .pth model file"""
        start_dir = os.path.join(os.getcwd(), 'checkpoints')
        path, _ = QFileDialog.getOpenFileName(self, "Select model file", start_dir, "PyTorch Model (*.pth)")
        if path:
            name = os.path.basename(path)
            # add to combo if not present
            if self.model_combo.findText(name) == -1:
                self.model_combo.addItem(name)
            # select it
            self.model_combo.setCurrentText(name)
