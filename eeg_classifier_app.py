"""
EEG Classification GUI Application using PyQt5
Converts the Jupyter notebook EEG classifier into a desktop application
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings("ignore")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QProgressBar, QTextEdit, QGroupBox, QSpinBox,
                            QDoubleSpinBox, QFileDialog, QMessageBox, QTabWidget,
                            QComboBox, QSlider, QCheckBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from mne.io import concatenate_raws
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchmetrics.classification import Accuracy

# Import our custom modules
from eeg_model import EEGClassificationModel, ModelWrapper, EEGDataset, AvgMeter
from data_processor import EEGDataProcessor


class PlotCanvas(FigureCanvas):
    """Custom matplotlib canvas for PyQt5"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

    def plot_eeg_data(self, data, title="EEG Data"):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(data.T)
        ax.set_title(title)
        ax.set_ylabel("Voltage (V)")
        ax.set_xlabel("Sample")
        ax.grid(True, alpha=0.3)
        self.draw()

    def plot_training_curves(self, train_loss, val_loss, train_acc, val_acc):
        self.fig.clear()
        
        # Loss plot
        ax1 = self.fig.add_subplot(221)
        ax1.plot(train_loss, 'r-', label='Train Loss')
        ax1.plot(val_loss, 'b-', label='Val Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2 = self.fig.add_subplot(222)
        ax2.plot(train_acc, 'r-', label='Train Accuracy')
        ax2.plot(val_acc, 'b-', label='Val Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.draw()


class TrainingThread(QThread):
    """Separate thread for model training to prevent GUI freezing"""
    progress_update = pyqtSignal(int)  # Current epoch
    epoch_metrics_update = pyqtSignal(dict)  # Metrics per epoch
    time_update = pyqtSignal(str)  # Time estimates
    log_update = pyqtSignal(str)
    training_complete = pyqtSignal(object)
    
    def __init__(self, arch, dataset, config):
        super().__init__()
        self.arch = arch  # Just the architecture, not the wrapped model
        self.dataset = dataset
        self.config = config
        self.start_time = None
        self.epoch_times = []
        self.should_stop = False
        
    def stop_training(self):
        """Signal to stop training"""
        self.should_stop = True
    def run(self):
        try:
            import time
            from pytorch_lightning import Trainer
            from pytorch_lightning.callbacks import Callback
            from eeg_model import ModelWrapper  # Import here to avoid PyQt5 issues
            
            self.start_time = time.time()
            self.log_update.emit("Starting training...")
            
            # Create the Lightning model wrapper INSIDE the thread
            # This avoids PyQt5 threading issues with class identity
            self.log_update.emit("Creating Lightning model wrapper...")
            model = ModelWrapper(
                arch=self.arch,
                dataset=self.dataset,
                batch_size=self.config['batch_size'],
                lr=self.config['lr'],
                max_epoch=self.config['max_epochs']
            )
            
            self.log_update.emit(f"Model created: {type(model)}")            # Verify it's a proper Lightning module
            import pytorch_lightning as pl
            if not isinstance(model, pl.LightningModule):
                raise TypeError(f"Model is not a LightningModule: {type(model)}")
            
            self.log_update.emit("Model verified as LightningModule")
            
            # Create a custom callback that properly communicates with the GUI thread
            class ProgressCallback(Callback):
                def __init__(self, thread_ref):
                    super().__init__()
                    self.thread_ref = thread_ref
                    self.epoch_start_time = None
                    self.epoch_times = []
                
                def on_train_epoch_start(self, trainer, pl_module):
                    if self.thread_ref.should_stop:
                        trainer.should_stop = True
                        return
                        
                    self.epoch_start_time = time.time()
                    current_epoch = trainer.current_epoch + 1
                    self.thread_ref.progress_update.emit(current_epoch)
                    self.thread_ref.log_update.emit(f"Starting epoch {current_epoch}/{trainer.max_epochs}")
                
                def on_train_epoch_end(self, trainer, pl_module):
                    if self.epoch_start_time:
                        epoch_time = time.time() - self.epoch_start_time
                        self.epoch_times.append(epoch_time)
                        
                        current_epoch = trainer.current_epoch + 1
                        
                        # Calculate ETA
                        if len(self.epoch_times) > 0:
                            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
                            remaining_epochs = trainer.max_epochs - current_epoch
                            eta_seconds = remaining_epochs * avg_epoch_time
                            
                            eta_minutes = eta_seconds / 60
                            if eta_minutes < 1:
                                eta_str = f"{eta_seconds:.0f}s"
                            elif eta_minutes < 60:
                                eta_str = f"{eta_minutes:.1f}m"
                            else:
                                eta_hours = eta_minutes / 60
                                eta_str = f"{eta_hours:.1f}h"
                                
                            self.thread_ref.time_update.emit(f"Epoch: {epoch_time:.1f}s | ETA: {eta_str}")
                        
                        # Get current metrics
                        if hasattr(pl_module, 'train_loss') and hasattr(pl_module, 'val_loss'):
                            metrics = {
                                'epoch': current_epoch,
                                'train_loss': pl_module.train_loss[-1] if pl_module.train_loss else 0,
                                'val_loss': pl_module.val_loss[-1] if pl_module.val_loss else 0,
                                'train_acc': pl_module.train_acc[-1] if pl_module.train_acc else 0,
                                'val_acc': pl_module.val_acc[-1] if pl_module.val_acc else 0,
                                'epoch_time': epoch_time
                            }
                            self.thread_ref.epoch_metrics_update.emit(metrics)
            
            # Setup trainer with our callback
            progress_callback = ProgressCallback(self)
            trainer = Trainer(
                accelerator="auto",
                devices=1,
                max_epochs=self.config['max_epochs'],
                log_every_n_steps=5,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                callbacks=[progress_callback]
            )
            
            # Train model
            self.log_update.emit("Starting trainer.fit...")
            trainer.fit(model)
            
            total_time = time.time() - self.start_time
            self.log_update.emit(f"Training completed successfully in {total_time:.2f} seconds!")
            self.training_complete.emit(model)
            
        except Exception as e:
            self.log_update.emit(f"Training failed: {str(e)}")
            import traceback
            self.log_update.emit(f"Traceback: {traceback.format_exc()}")


class EEGClassifierApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.dataset = None
        self.data_processor = EEGDataProcessor()
        self.training_thread = None
        
        # Training progress tracking
        self.training_start_time = None
        self.best_metrics = {
            'train_loss': float('inf'),
            'val_loss': float('inf'),
            'train_acc': 0.0,
            'val_acc': 0.0
        }
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("EEG Motor Imagery Classification")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        central_widget_layout = QVBoxLayout(central_widget)
        central_widget_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_data_tab()
        self.create_training_tab()
        self.create_prediction_tab()
        
    def create_data_tab(self):
        """Create data loading and preprocessing tab"""
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)
        
        # Data loading section
        data_group = QGroupBox("Data Loading")
        data_layout = QGridLayout(data_group)
        
        self.subjects_spinbox = QSpinBox()
        self.subjects_spinbox.setRange(1, 109)
        self.subjects_spinbox.setValue(79)
        data_layout.addWidget(QLabel("Number of Subjects:"), 0, 0)
        data_layout.addWidget(self.subjects_spinbox, 0, 1)
        
        self.data_path_label = QLabel("Data Path: Not selected")
        self.browse_data_btn = QPushButton("Browse Data Directory")
        self.browse_data_btn.clicked.connect(self.browse_data_directory)
        data_layout.addWidget(self.data_path_label, 1, 0, 1, 2)
        data_layout.addWidget(self.browse_data_btn, 2, 0, 1, 2)
        
        self.load_data_btn = QPushButton("Load and Process Data")
        self.load_data_btn.clicked.connect(self.load_data)
        data_layout.addWidget(self.load_data_btn, 3, 0, 1, 2)
        
        layout.addWidget(data_group)
        
        # Data visualization section
        viz_group = QGroupBox("Data Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        self.data_canvas = PlotCanvas(self, width=8, height=4)
        viz_layout.addWidget(self.data_canvas)
        
        viz_controls = QHBoxLayout()
        self.sample_slider = QSlider(Qt.Horizontal)
        self.sample_slider.setEnabled(False)
        self.sample_slider.valueChanged.connect(self.update_visualization)
        viz_controls.addWidget(QLabel("Sample:"))
        viz_controls.addWidget(self.sample_slider)
        
        viz_layout.addLayout(viz_controls)
        layout.addWidget(viz_group)
        
        # Log section
        log_group = QGroupBox("Data Processing Log")
        log_layout = QVBoxLayout(log_group)
        self.data_log = QTextEdit()
        self.data_log.setMaximumHeight(150)
        log_layout.addWidget(self.data_log)
        layout.addWidget(log_group)
        
        self.tab_widget.addTab(data_tab, "Data")
        
    def create_training_tab(self):
        """Create model training tab"""
        training_tab = QWidget()
        layout = QVBoxLayout(training_tab)
        
        # Training configuration
        config_group = QGroupBox("Training Configuration")
        config_layout = QGridLayout(config_group)
        
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(100)
        config_layout.addWidget(QLabel("Max Epochs:"), 0, 0)
        config_layout.addWidget(self.epochs_spinbox, 0, 1)
        
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 128)
        self.batch_size_spinbox.setValue(10)
        config_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        config_layout.addWidget(self.batch_size_spinbox, 1, 1)
        
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.0001, 0.1)
        self.lr_spinbox.setValue(0.0005)
        self.lr_spinbox.setDecimals(4)
        config_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        config_layout.addWidget(self.lr_spinbox, 2, 1)
        
        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setRange(0.0, 0.9)
        self.dropout_spinbox.setValue(0.125)
        self.dropout_spinbox.setDecimals(3)
        config_layout.addWidget(QLabel("Dropout:"), 3, 0)
        config_layout.addWidget(self.dropout_spinbox, 3, 1)
        
        layout.addWidget(config_group)
        
        # Training controls
        controls_group = QGroupBox("Training Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        
        controls_layout.addWidget(self.train_btn)
        controls_layout.addWidget(self.stop_btn)
        layout.addWidget(controls_group)
          # Progress section
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Main progress bar with epoch info
        progress_info_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Epoch %v/%m")
        self.epoch_label = QLabel("Ready to train")
        self.epoch_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        progress_info_layout.addWidget(self.progress_bar, 3)
        progress_info_layout.addWidget(self.epoch_label, 1)
        progress_layout.addLayout(progress_info_layout)
        
        # Time and ETA information
        time_layout = QHBoxLayout()
        self.time_label = QLabel("Training time: --")
        self.eta_label = QLabel("ETA: --")
        self.time_label.setStyleSheet("color: #FFA500;")
        self.eta_label.setStyleSheet("color: #87CEEB;")
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.eta_label)
        progress_layout.addLayout(time_layout)
        
        # Real-time metrics display
        metrics_layout = QGridLayout()
        
        # Current metrics labels
        self.current_train_loss_label = QLabel("Train Loss: --")
        self.current_val_loss_label = QLabel("Val Loss: --")
        self.current_train_acc_label = QLabel("Train Acc: --")
        self.current_val_acc_label = QLabel("Val Acc: --")
        
        # Best metrics labels
        self.best_train_loss_label = QLabel("Best Train Loss: --")
        self.best_val_loss_label = QLabel("Best Val Loss: --")
        self.best_train_acc_label = QLabel("Best Train Acc: --")
        self.best_val_acc_label = QLabel("Best Val Acc: --")
        
        # Style current metrics
        current_style = "color: #90EE90; font-weight: bold;"
        self.current_train_loss_label.setStyleSheet(current_style)
        self.current_val_loss_label.setStyleSheet(current_style)
        self.current_train_acc_label.setStyleSheet(current_style)
        self.current_val_acc_label.setStyleSheet(current_style)
        
        # Style best metrics
        best_style = "color: #FFD700; font-size: 12px;"
        self.best_train_loss_label.setStyleSheet(best_style)
        self.best_val_loss_label.setStyleSheet(best_style)
        self.best_train_acc_label.setStyleSheet(best_style)
        self.best_val_acc_label.setStyleSheet(best_style)
        
        # Add to grid
        metrics_layout.addWidget(QLabel("Current Metrics:"), 0, 0, 1, 2)
        metrics_layout.addWidget(self.current_train_loss_label, 1, 0)
        metrics_layout.addWidget(self.current_val_loss_label, 1, 1)
        metrics_layout.addWidget(self.current_train_acc_label, 2, 0)
        metrics_layout.addWidget(self.current_val_acc_label, 2, 1)
        
        metrics_layout.addWidget(QLabel("Best Metrics:"), 3, 0, 1, 2)
        metrics_layout.addWidget(self.best_train_loss_label, 4, 0)
        metrics_layout.addWidget(self.best_val_loss_label, 4, 1)
        metrics_layout.addWidget(self.best_train_acc_label, 5, 0)
        metrics_layout.addWidget(self.best_val_acc_label, 5, 1)
        
        progress_layout.addLayout(metrics_layout)
        
        # Training curves plot
        self.training_canvas = PlotCanvas(self, width=10, height=5)
        progress_layout.addWidget(self.training_canvas)
        
        layout.addWidget(progress_group)
        
        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        self.training_log = QTextEdit()
        self.training_log.setMaximumHeight(150)
        log_layout.addWidget(self.training_log)
        layout.addWidget(log_group)
        
        self.tab_widget.addTab(training_tab, "Training")
        
    def create_prediction_tab(self):
        """Create prediction tab"""
        prediction_tab = QWidget()
        layout = QVBoxLayout(prediction_tab)
        
        # Model loading section
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        
        model_controls = QHBoxLayout()
        self.load_model_btn = QPushButton("Load Saved Model")
        self.load_model_btn.clicked.connect(self.load_model)
        self.save_model_btn = QPushButton("Save Current Model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        
        model_controls.addWidget(self.load_model_btn)
        model_controls.addWidget(self.save_model_btn)
        model_layout.addLayout(model_controls)
        
        layout.addWidget(model_group)
        
        # Prediction section
        pred_group = QGroupBox("Real-time Prediction")
        pred_layout = QVBoxLayout(pred_group)
        
        self.predict_btn = QPushButton("Predict Random Sample")
        self.predict_btn.clicked.connect(self.predict_sample)
        self.predict_btn.setEnabled(False)
        
        pred_layout.addWidget(self.predict_btn)
        
        # Prediction results
        self.prediction_canvas = PlotCanvas(self, width=8, height=4)
        pred_layout.addWidget(self.prediction_canvas)
        
        self.prediction_label = QLabel("Prediction: Not available")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
        pred_layout.addWidget(self.prediction_label)
        
        layout.addWidget(pred_group)
        
        self.tab_widget.addTab(prediction_tab, "Prediction")
        
    def browse_data_directory(self):
        """Browse for data directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.data_path_label.setText(f"Data Path: {directory}")
            self.data_processor.set_data_path(directory)
            
    def load_data(self):
        """Load and process EEG data"""
        try:
            self.data_log.append("Starting data loading...")
            n_subjects = self.subjects_spinbox.value()
            
            # Load data using the processor
            X, y, info = self.data_processor.load_data(n_subjects)
            
            # Create dataset
            self.dataset = EEGDataset(x=X, y=y)
            
            self.data_log.append(f"Data loaded successfully!")
            self.data_log.append(f"Shape: {X.shape}")
            self.data_log.append(f"Classes: {len(np.unique(y))} (left/right)")
            
            # Enable visualization
            self.sample_slider.setRange(0, X.shape[0] - 1)
            self.sample_slider.setEnabled(True)
            self.sample_slider.setValue(0)
            
            # Store data for visualization
            self.X = X
            self.y = y
            
            # Update visualization
            self.update_visualization()
            
            # Enable training
            self.train_btn.setEnabled(True)
            
        except Exception as e:
            self.data_log.append(f"Error loading data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            
    def update_visualization(self):
        """Update data visualization"""
        if hasattr(self, 'X') and self.X is not None:
            sample_idx = self.sample_slider.value()
            sample = self.X[sample_idx]
            label = "left" if self.y[sample_idx] == 0 else "right"
            
            self.data_canvas.plot_eeg_data(
                sample, 
                f"EEG Sample {sample_idx} - Label: {label}"
            )
    def start_training(self):
        """Start model training in separate thread"""
        if self.dataset is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return
            
        try:
            import time
            
            # Reset progress tracking
            self.training_start_time = time.time()
            self.best_metrics = {
                'train_loss': float('inf'),
                'val_loss': float('inf'),
                'train_acc': 0.0,
                'val_acc': 0.0
            }
            
            # Create model
            eeg_channel = self.X.shape[1]  # Number of EEG channels
            model = EEGClassificationModel(
                eeg_channel=eeg_channel, 
                dropout=self.dropout_spinbox.value()
            )
            
            # Create training configuration
            config = {
                'max_epochs': self.epochs_spinbox.value(),
                'batch_size': self.batch_size_spinbox.value(),
                'lr': self.lr_spinbox.value()
            }
              # Wrap model
            self.model = ModelWrapper(
                arch=model, 
                dataset=self.dataset, 
                batch_size=config['batch_size'], 
                lr=config['lr'], 
                max_epoch=config['max_epochs']
            )
            
            # Start training thread
            self.training_thread = TrainingThread(self.model, self.dataset, config)
            
            # Connect all signals
            self.training_thread.log_update.connect(self.training_log.append)
            self.training_thread.progress_update.connect(self.update_training_progress)
            self.training_thread.epoch_metrics_update.connect(self.update_epoch_metrics)
            self.training_thread.time_update.connect(self.update_time_info)
            self.training_thread.training_complete.connect(self.training_completed)
            
            self.training_thread.start()
            
            # Update UI
            self.train_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            # Setup progress bar
            self.progress_bar.setRange(0, config['max_epochs'])
            self.progress_bar.setValue(0)
            self.epoch_label.setText("Initializing...")
            
            # Reset time labels
            self.time_label.setText("Training time: 00:00")
            self.eta_label.setText("ETA: Calculating...")
            
            # Reset metrics labels
            self.current_train_loss_label.setText("Train Loss: --")
            self.current_val_loss_label.setText("Val Loss: --")
            self.current_train_acc_label.setText("Train Acc: --")
            self.current_val_acc_label.setText("Val Acc: --")
            
            self.training_log.append("Training started...")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")
    def stop_training(self):
        """Stop training"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop_training()  # Signal the thread to stop gracefully
            self.training_thread.wait(3000)  # Wait up to 3 seconds
            if self.training_thread.isRunning():
                self.training_thread.terminate()  # Force terminate if needed
            self.training_log.append("Training stopped by user.")
            
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.epoch_label.setText("Training stopped")
        self.progress_bar.setValue(0)

    def training_completed(self, model):
        """Handle training completion"""
        import time
        
        self.model = model
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_model_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)
        
        # Update progress display
        max_epochs = self.progress_bar.maximum()
        self.progress_bar.setValue(max_epochs)
        self.epoch_label.setText(f"Training Complete! ({max_epochs} epochs)")
        
        # Calculate final training time
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            if hours > 0:
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                time_str = f"{minutes:02d}:{seconds:02d}"
                
            self.time_label.setText(f"Total time: {time_str}")
            self.eta_label.setText("✓ Complete")
        
        # Plot final training curves
        if hasattr(self.model, 'train_loss') and self.model.train_loss:
            self.training_canvas.plot_training_curves(
                self.model.train_loss,
                self.model.val_loss,
                self.model.train_acc,
                self.model.val_acc
            )
            
            # Log final metrics
            final_train_loss = self.model.train_loss[-1] if self.model.train_loss else "N/A"
            final_val_loss = self.model.val_loss[-1] if self.model.val_loss else "N/A"
            final_train_acc = self.model.train_acc[-1] if self.model.train_acc else "N/A"
            final_val_acc = self.model.val_acc[-1] if self.model.val_acc else "N/A"
            
            self.training_log.append("=" * 50)
            self.training_log.append("TRAINING SUMMARY:")
            self.training_log.append(f"Final Train Loss: {final_train_loss:.4f}" if isinstance(final_train_loss, float) else f"Final Train Loss: {final_train_loss}")
            self.training_log.append(f"Final Val Loss: {final_val_loss:.4f}" if isinstance(final_val_loss, float) else f"Final Val Loss: {final_val_loss}")
            self.training_log.append(f"Final Train Acc: {final_train_acc:.3f}" if isinstance(final_train_acc, float) else f"Final Train Acc: {final_train_acc}")
            self.training_log.append(f"Final Val Acc: {final_val_acc:.3f}" if isinstance(final_val_acc, float) else f"Final Val Acc: {final_val_acc}")
            self.training_log.append(f"Best Val Loss: {self.best_metrics['val_loss']:.4f}")
            self.training_log.append(f"Best Val Acc: {self.best_metrics['val_acc']:.3f}")
            self.training_log.append("=" * 50)
            
    def load_model(self):
        """Load a saved model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "PyTorch Models (*.ckpt *.pth)"
        )
        if file_path:
            try:
                # This would need to be implemented based on your model saving format
                self.training_log.append(f"Model loaded from: {file_path}")
                self.predict_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                
    def save_model(self):
        """Save the current model"""
        if self.model is None:
            QMessageBox.warning(self, "Warning", "No model to save!")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "PyTorch Models (*.ckpt)"
        )
        if file_path:
            try:
                # Save model checkpoint
                torch.save(self.model.state_dict(), file_path)
                self.training_log.append(f"Model saved to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
                
    def predict_sample(self):
        """Predict a random sample"""
        if self.model is None or not hasattr(self, 'X'):
            QMessageBox.warning(self, "Warning", "Please train or load a model first!")
            return
            
        try:
            # Select random sample
            sample_idx = np.random.randint(0, self.X.shape[0])
            sample = self.X[sample_idx]
            actual_label = "left" if self.y[sample_idx] == 0 else "right"
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                sample_tensor = torch.tensor(sample).unsqueeze(0).float()
                output = self.model(sample_tensor)
                prediction = torch.sigmoid(output).item()
                predicted_label = "right" if prediction > 0.5 else "left"
                confidence = prediction if prediction > 0.5 else 1 - prediction
            
            # Update display
            self.prediction_canvas.plot_eeg_data(
                sample,
                f"Sample {sample_idx} - Actual: {actual_label}, Predicted: {predicted_label}"
            )
            
            self.prediction_label.setText(
                f"Prediction: {predicted_label.upper()} (Confidence: {confidence:.2%})\n"
                f"Actual: {actual_label.upper()}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")
    def update_training_progress(self, current_epoch):
        """Update training progress bar and epoch information"""
        import time
        
        self.progress_bar.setValue(current_epoch)
        max_epochs = self.progress_bar.maximum()
        self.epoch_label.setText(f"Epoch {current_epoch}/{max_epochs}")
        
        # Update elapsed time
        if self.training_start_time:
            elapsed = time.time() - self.training_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            
            if hours > 0:
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                time_str = f"{minutes:02d}:{seconds:02d}"
                
            self.time_label.setText(f"Training time: {time_str}")
    
    def update_epoch_metrics(self, metrics):
        """Update current epoch metrics and track best metrics"""
        epoch = metrics.get('epoch', 0)
        train_loss = metrics.get('train_loss', 0)
        val_loss = metrics.get('val_loss', 0)
        train_acc = metrics.get('train_acc', 0)
        val_acc = metrics.get('val_acc', 0)
        
        # Update current metrics display
        self.current_train_loss_label.setText(f"Train Loss: {train_loss:.4f}")
        self.current_val_loss_label.setText(f"Val Loss: {val_loss:.4f}")
        self.current_train_acc_label.setText(f"Train Acc: {train_acc:.3f}")
        self.current_val_acc_label.setText(f"Val Acc: {val_acc:.3f}")
        
        # Update best metrics
        if train_loss < self.best_metrics['train_loss']:
            self.best_metrics['train_loss'] = train_loss
            self.best_train_loss_label.setText(f"Best Train Loss: {train_loss:.4f}")
            
        if val_loss < self.best_metrics['val_loss']:
            self.best_metrics['val_loss'] = val_loss
            self.best_val_loss_label.setText(f"Best Val Loss: {val_loss:.4f}")
            
        if train_acc > self.best_metrics['train_acc']:
            self.best_metrics['train_acc'] = train_acc
            self.best_train_acc_label.setText(f"Best Train Acc: {train_acc:.3f}")
            
        if val_acc > self.best_metrics['val_acc']:
            self.best_metrics['val_acc'] = val_acc
            self.best_val_acc_label.setText(f"Best Val Acc: {val_acc:.3f}")
        
        # Update training curves in real-time
        if hasattr(self.model, 'train_loss') and self.model.train_loss:
            self.training_canvas.plot_training_curves(
                self.model.train_loss,
                self.model.val_loss,
                self.model.train_acc,
                self.model.val_acc
            )
    
    def update_time_info(self, time_info):
        """Update ETA and epoch time information"""
        self.eta_label.setText(f"ETA: {time_info.split('|')[1].strip() if '|' in time_info else time_info}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = EEGClassifierApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
