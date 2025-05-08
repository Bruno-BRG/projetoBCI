# Standard library imports
import os

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QInputDialog,
    QMessageBox, QGroupBox
)

# Local imports
from model.BCISystem import create_bci_system
from model.EEGAugmentation import load_local_eeg_data

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
                self.bci.train_calibration(num_epochs=50, batch_size=10, learning_rate=5e-4)
                QMessageBox.information(self, "Success", "Model training completed successfully")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Training failed: {str(e)}")
