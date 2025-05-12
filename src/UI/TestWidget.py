# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QGroupBox,
    QInputDialog
)
import os
from datetime import datetime

# Local imports
from model.MultiSubjectTest import MultiSubjectTest

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
        
        # Ask for model name
        name, ok = QInputDialog.getText(self, "Model Name", "Enter name for multi-subject model:", text=datetime.now().strftime("multisubject_%Y%m%d_%H%M%S"))
        if not ok or not name.strip():
            self.status_label.setText("Testing cancelled: no model name provided")
            self.start_test_button.setEnabled(True)
            return
        model_filename = f"{name}.pth"
        os.makedirs('checkpoints', exist_ok=True)
        model_path = os.path.join('checkpoints', model_filename)
        try:
            # Initialize test system with provided save path
            self.test_system = MultiSubjectTest(train_samples=40, test_samples=20, model_path=model_path)
             
            # Run training and evaluation
            history = self.test_system.train_and_evaluate(
                num_epochs=100,
                batch_size=10,
                learning_rate=5e-4
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
                f"Validation Accuracy: {final_val_acc:.2%}\n"
                f"Model saved at: {model_path}"
            )
         
        except Exception as e:
            self.status_label.setText("Error during testing")
            self.progress_label.setText(str(e))
        finally:
            self.start_test_button.setEnabled(True)
