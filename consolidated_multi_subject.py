"""
Consolidated script for multi-subject EEG classification training.
This script combines functionality from multi_subject_train.py and MultiSubjectTest.py
to provide a streamlined workflow for training models on data from multiple subjects.
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as L
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Import local modules
from src.model.EEGDataset import EEGDataset
from src.model.EEGClassificationModel import EEGClassificationModel
from src.model.ModelWrapper import ModelWrapper
from src.model.EEGDataLoader import load_local_eeg_data
from src.model.ModelTracker import get_device

# Set random seed for reproducibility
SEED = 42
L.seed_everything(SEED, workers=True)

class ConsolidatedMultiSubjectTrainer:
    """
    A consolidated class that handles all aspects of multi-subject EEG training:
    - Data collection from multiple subjects
    - Data preprocessing
    - Model training with PyTorch Lightning
    - Performance evaluation
    - Model saving and inference
    """
    def __init__(
        self,
        test_size=0.2,
        batch_size=10,
        learning_rate=5e-4,
        max_epochs=100,
        dropout=0.2,
        input_length=125,
        model_name="MultiSubjectEEGModel",
        checkpoint_dir="checkpoints"
        
    ):
        """
        Initialize the multi-subject trainer with configurable parameters.
        
        Args:
            test_size: Fraction of data to use as test set (between 0.0 and 1.0)
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            max_epochs: Maximum number of training epochs
            dropout: Dropout rate for the model
            input_length: Standard input time dimension
            model_name: Name prefix for saved model files
            checkpoint_dir: Directory to save model checkpoints
        """
        # Configuration parameters
        self.test_size = test_size
        self.batch_size = batch_size
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.dropout = dropout
        self.input_length = input_length
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        # Data loading handled in main via EEGDataLoader

        # Create timestamp for unique model identification
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Model path for saving
        self.model_path = os.path.join(
            self.checkpoint_dir, 
            f"{self.model_name}_{self.timestamp}.ckpt"
        )
        
        # Device configuration
        self.device = get_device()
        
        # Initialize components
        self.model = None
        self.model_wrapper = None
        self.trainer = None
        self.best_model_path = None
          # Tracking variables
        self.training_history = {
            'train_losses': [], 'val_losses': [],
            'train_accs': [], 'val_accs': []
        }
        
        print(f"\nInitializing ConsolidatedMultiSubjectTrainer:")
        print(f"- Test split ratio: {self.test_size}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Learning rate: {self.lr}")
        print(f"- Max epochs: {self.max_epochs}")
        print(f"- Using device: {self.device}")
        print(f"- Model checkpoint will be saved to: {self.model_path}")
    
# Data loading moved into main()
# Main execution
def main():
    """Main function to run the consolidated multi-subject training"""
    # Load EEG data from all available subjects (1-109)
    subject_ids = list(range(1, 21))
    X, y, eeg_channel = load_local_eeg_data(subject_ids)
    eeg_dataset = EEGDataset(x=X, y=y)
    # Hyperparameters
    MAX_EPOCH = 10
    BATCH_SIZE = 10
    LR = 5e-4
    MODEL_NAME = "EEGClassificationModel"
    CHECKPOINT_DIR = "checkpoints"
    # Build model and wrapper
    model = EEGClassificationModel(eeg_channel=eeg_channel, dropout=0.125)
    model_wrapper = ModelWrapper(model, eeg_dataset, BATCH_SIZE, LR, MAX_EPOCH)

    # Set up logging configuration
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # Configure loggers
    tensorboardlogger = TensorBoardLogger(
        save_dir="logs/",
        name="lightning_logs",
        log_graph=True
    )
    csvlogger = CSVLogger(save_dir="logs/")
    
    # Configure callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint = ModelCheckpoint(
        monitor='val_acc',
        dirpath=CHECKPOINT_DIR,
        mode='max',
        save_top_k=1,
        filename=f'{MODEL_NAME}-{{epoch:02d}}-{{val_acc:.2f}}'
    )
    early_stopping = EarlyStopping(
        monitor="val_acc",
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode="max"
    )

    seed_everything(SEED, workers=True)
    
    # Create trainer with simplified configuration
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCH,
        logger=[tensorboardlogger, csvlogger],
        callbacks=[lr_monitor, checkpoint, early_stopping],
        log_every_n_steps=5,
        enable_progress_bar=True
    )
    
    trainer.fit(model_wrapper)

    # Run test on best checkpoint
    trainer.test(model=model_wrapper, ckpt_path="best")

    # Save and rename best checkpoint
    os.rename(
        checkpoint.best_model_path,
        os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.ckpt")
    )

    # Plot loss curves
    plt.figure(figsize=(8, 4))
    plt.plot(model_wrapper.train_loss, color='r', label='Train Loss')
    plt.plot(model_wrapper.val_loss, color='b', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_plot.png')
    plt.show()

    # Plot accuracy curve
    plt.figure(figsize=(8, 4))
    plt.plot(model_wrapper.train_acc, color='r', label='Train Accuracy')
    plt.plot(model_wrapper.val_acc, color='b', label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('accuracy_plot.png')
    plt.show()

    return checkpoint.best_model_path

if __name__ == "__main__":
    main()

