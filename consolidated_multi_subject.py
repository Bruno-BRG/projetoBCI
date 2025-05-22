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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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
        """        # Configuration parameters
        self.test_size = test_size
        self.batch_size = batch_size
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.dropout = dropout
        self.input_length = input_length
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        
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
    
    def collect_subject_data(self, subject_id):
        """
        Collect all available data from a subject
        
        Args:
            subject_id: The subject ID to collect data from
            
        Returns:
            Tuple of (X, y) with all available subject data, or (None, None) if error occurs
        """
        try:
            X, y, ch = load_local_eeg_data(subject_id, augment=False)
            
            # Check if we have both classes (left and right)
            left_idx = (y == 0).nonzero()[0]
            right_idx = (y == 1).nonzero()[0]
            
            if len(left_idx) == 0 or len(right_idx) == 0:
                print(f"Subject {subject_id:03d}: Missing class data (L:{len(left_idx)}, R:{len(right_idx)})")
                return None, None
                
            print(f"Subject {subject_id:03d}: Successfully collected {len(y)} samples " +
                  f"(L:{len(left_idx)}, R:{len(right_idx)})")
            return X, y
        except Exception as e:
            print(f"Subject {subject_id:03d}: Error loading data - {str(e)}")
            return None, None

    def prepare_datasets(self):
        """
        Prepare training and testing datasets from multiple subjects using train_test_split

        Returns:
            Tuple of (train_X, train_y, test_X, test_y) with prepared data
        """
        print("\nStarting dataset preparation...")
        all_X, all_y = [], []
        
        successful_subjects = 0
        total_subjects = 0
        
        # Try subjects from 1 to 109 (maximum in dataset)
        for subject_id in range(1, 110):
            total_subjects += 1
            try:
                print(f"\nProcessing Subject {subject_id:03d}...")
                
                # Get all available data from subject
                X, y = self.collect_subject_data(subject_id)
                if X is not None and y is not None:
                    all_X.append(X)
                    all_y.append(y)
                    successful_subjects += 1
                
            except Exception as e:
                print(f"Error processing subject {subject_id}: {str(e)}")
                continue
        
        if not all_X or not all_y:
            raise ValueError("Could not collect enough samples from any subject")
            
        # Concatenate all subject data
        X_combined = np.concatenate(all_X)
        y_combined = np.concatenate(all_y)
        
        # Split using scikit-learn's train_test_split with stratification
        final_train_X, final_test_X, final_train_y, final_test_y = train_test_split(
            X_combined, y_combined, 
            test_size=self.test_size,
            random_state=42,
            stratify=y_combined  # Maintain class distribution
        )
        
        print(f"\nDataset preparation completed:")
        print(f"Successfully processed {successful_subjects} out of {total_subjects} subjects")
        print(f"Total training samples: {len(final_train_X)}")
        print(f"Total testing samples: {len(final_test_X)}")
        print(f"Training data shape: {final_train_X.shape}")
        print(f"Testing data shape: {final_test_X.shape}")

        # Standardize time dimension to match expected input dimensions
        if final_train_X.shape[2] != self.input_length:
            print(f"Standardizing time dimension to {self.input_length}...")
            final_train_X = self._standardize_time_dimension(final_train_X)
            final_test_X = self._standardize_time_dimension(final_test_X)
            print(f"After standardization - Train: {final_train_X.shape}, Test: {final_test_X.shape}")
        
        return final_train_X, final_train_y, final_test_X, final_test_y

    def _standardize_time_dimension(self, data):
        """
        Standardize time dimension to self.input_length samples
        
        Args:
            data: Input data array of shape (batch_size, channels, time_points)
            
        Returns:
            Standardized data with consistent time dimension
        """
        batch_size, channels, time_points = data.shape
        
        # If time dimension is longer, truncate
        if time_points > self.input_length:
            return data[:, :, :self.input_length]
        
        # If time dimension is shorter, pad with zeros
        elif time_points < self.input_length:
            padding = np.zeros((batch_size, channels, self.input_length - time_points))
            return np.concatenate([data, padding], axis=2)
        
        # If time dimension is already correct, return as is
        return data
        
    def train(self):
        """
        Full training pipeline: collect data, initialize model, train with Lightning
        
        Returns:
            Path to the best saved model
        """
        # Collect data from multiple subjects
        print("Collecting data from multiple subjects...")
        train_X, train_y, test_X, test_y = self.prepare_datasets()
        
        print(f"\nFinal dataset sizes:")
        print(f"Training data: {train_X.shape}, {train_y.shape}")
        print(f"Test data: {test_X.shape}, {test_y.shape}")
        
        # Create PyTorch datasets
        train_dataset = EEGDataset(x=train_X, y=train_y)
        test_dataset = EEGDataset(x=test_X, y=test_y)
        
        # Create the model with the correct number of channels
        eeg_channel = train_X.shape[1]  # Number of EEG channels
        self.model = EEGClassificationModel(eeg_channel=eeg_channel, dropout=self.dropout)
        
        # Wrap the model for training with Lightning
        self.model_wrapper = ModelWrapper(
            arch=self.model,
            dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=self.batch_size,
            lr=self.lr,
            max_epoch=self.max_epochs
        )
        
        # Setup loggers
        tensorboard_logger = TensorBoardLogger(save_dir="logs/multi_subject/")
        
        # Callbacks
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint = ModelCheckpoint(
            monitor='val_acc',
            dirpath=self.checkpoint_dir,
            filename=f'{self.model_name}_best_{self.timestamp}',
            mode='max',
            save_top_k=1
        )
        early_stopping = EarlyStopping(
            monitor="val_acc", 
            min_delta=0.005, 
            patience=5,
            verbose=True, 
            mode="max"
        )
        
        # Train the model
        print("Starting training...")
        self.trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=self.max_epochs,
            logger=tensorboard_logger,
            callbacks=[lr_monitor, checkpoint, early_stopping],
            log_every_n_steps=10,
        )
        self.trainer.fit(self.model_wrapper)
        
        # Test the model
        print("Testing model...")
        self.trainer.test(self.model_wrapper, ckpt_path=checkpoint.best_model_path)
        
        self.best_model_path = checkpoint.best_model_path
        print(f"Training and evaluation completed. Model saved to {self.best_model_path}")
        return self.best_model_path
        
    def predict_hand_movement(self, sample):
        """
        Predict hand movement from a single EEG sample using the trained model
        
        Args:
            sample: EEG data with shape (channels, time)
            
        Returns:
            Prediction (left or right)
        """
        # Check if model has been trained
        if not self.best_model_path:
            raise ValueError("No trained model available. Run train() first.")
            
        # Ensure sample is correctly shaped
        if sample.ndim == 2:
            # Add batch dimension if not present
            sample = np.expand_dims(sample, axis=0)
        
        # Create inference dataset
        inference_dataset = EEGDataset.inference_dataset(sample)
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=inference_dataset,
            batch_size=1,
            shuffle=False,
        )
        
        # Load model architecture
        eeg_channel = sample.shape[1]
        model = EEGClassificationModel(eeg_channel=eeg_channel)
        
        # Create wrapper
        model_wrapper = ModelWrapper(
            arch=model,
            dataset=inference_dataset,
            batch_size=1,
            lr=self.lr,
            max_epoch=1
        )
        
        # Predict
        trainer = L.Trainer()
        prediction = trainer.predict(
            model=model_wrapper,
            dataloaders=dataloader,
            ckpt_path=self.best_model_path,
        )[0]
        
        # Convert to binary prediction
        pred_class = int(torch.sigmoid(prediction) > 0.5)
        classes = ["left", "right"]
        return classes[pred_class]
    
    @staticmethod
    def load_and_predict(sample, model_path):
        """
        Static method to load a model and make a prediction without training
        
        Args:
            sample: EEG data with shape (channels, time)
            model_path: Path to the trained model checkpoint
            
        Returns:
            Prediction (left or right)
        """
        # Ensure sample is correctly shaped
        if sample.ndim == 2:
            # Add batch dimension if not present
            sample = np.expand_dims(sample, axis=0)
        
        # Create inference dataset
        inference_dataset = EEGDataset.inference_dataset(sample)
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=inference_dataset,
            batch_size=1,
            shuffle=False,
        )
        
        # Load model architecture
        eeg_channel = sample.shape[1]
        model = EEGClassificationModel(eeg_channel=eeg_channel)
        
        # Create wrapper
        model_wrapper = ModelWrapper(
            arch=model,
            dataset=inference_dataset,
            batch_size=1,
            lr=5e-4,
            max_epoch=1
        )
        
        # Predict
        trainer = L.Trainer()
        prediction = trainer.predict(
            model=model_wrapper,
            dataloaders=dataloader,
            ckpt_path=model_path,
        )[0]
        
        # Convert to binary prediction
        pred_class = int(torch.sigmoid(prediction) > 0.5)
        classes = ["left", "right"]
        return classes[pred_class]

# Main execution
def main():
    """Main function to run the consolidated multi-subject training"""
    # Initialize the trainer with default or custom parameters
    trainer = ConsolidatedMultiSubjectTrainer(
        test_size=0.2,  # 20% of data used for testing
        batch_size=10,
        learning_rate=5e-4,
        max_epochs=100,
        dropout=0.2,
        model_name="MultiSubjectEEGModel"
    )
    
    # Run the training process
    best_model_path = trainer.train()
    
    return best_model_path

if __name__ == "__main__":
    main()
