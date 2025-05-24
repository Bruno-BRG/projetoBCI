#!/usr/bin/env python3
"""
Launcher script for the EEG Classifier GUI Application
"""

import sys
import os

def main():
    """Launch the EEG Classifier application"""
    try:
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # Import and run the application
        from eeg_classifier_app import main as app_main
        app_main()
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
