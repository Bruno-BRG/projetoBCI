import sys
from pathlib import Path

# Add the src directory to the Python path
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from UI.CalibrationWidget import CalibrationWidget
from UI.RealUseWidget import RealUseWidget
from UI.StreamingWidget import StreamingWidget
from UI.TestWidget import TestWidget

# PyQt imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QInputDialog,
    QScrollArea, QCheckBox, QMessageBox, QGroupBox, QStyleFactory
)

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
