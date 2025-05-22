"""
Script for training the EEG Classification Model on data from multiple subjects.
This leverages the MultiSubjectTest class to collect balanced samples from all available subjects.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as L
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time

# Import local modules
from src.model.MultiSubjectTest import MultiSubjectTest
from src.model.EEGDataset import EEGDataset
from src.model.EEGClassificationModel import EEGClassificationModel
from src.model.ModelWrapper import ModelWrapper

# Set random seed for reproducibility
SEED = 42
L.seed_everything(SEED, workers=True)

def main():
    """Train the model on data from multiple subjects."""
    # Hyperparameters
    MAX_EPOCH = 100
    BATCH_SIZE = 10  # Increased batch size for more subjects
    LR = 5e-4
    TRAIN_SAMPLES_PER_SUBJECT = 36  # Number of training samples to collect per subject
    TEST_SAMPLES_PER_SUBJECT = 9   # Number of test samples to collect per subject
    MODEL_NAME = "MultiSubjectEEGModel"
    CHECKPOINT_DIR = "checkpoints"
    
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Create unique model path with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_{timestamp}.ckpt")
    
    print(f"Training multi-subject EEG classification model...")
    print(f"Checkpoint will be saved to: {model_path}")
    
    # Initialize the MultiSubjectTest class
    multi_subject = MultiSubjectTest(
        train_samples=TRAIN_SAMPLES_PER_SUBJECT,
        test_samples=TEST_SAMPLES_PER_SUBJECT,
        model_path=model_path
    )
    
    # Prepare datasets (this will collect data from all available subjects)
    print("Collecting data from multiple subjects...")
    train_X, train_y, test_X, test_y = multi_subject.prepare_datasets()
    
    print(f"\nFinal dataset sizes:")
    print(f"Training data: {train_X.shape}, {train_y.shape}")
    print(f"Test data: {test_X.shape}, {test_y.shape}")
    
    # Create PyTorch datasets
    train_dataset = EEGDataset(x=train_X, y=train_y)
    test_dataset = EEGDataset(x=test_X, y=test_y)
    
    # Create the model with the correct number of channels
    eeg_channel = train_X.shape[1]  # Number of EEG channels
    model = EEGClassificationModel(eeg_channel=eeg_channel, dropout=0.2)  # Increased dropout for better generalization
    
    # Wrap the model for training with Lightning
    model_wrapper = ModelWrapper(
        arch=model,
        dataset=train_dataset,
        test_dataset=test_dataset,  # Added test dataset explicitly
        batch_size=BATCH_SIZE,
        lr=LR,
        max_epoch=MAX_EPOCH
    )
    
    # Setup loggers
    tensorboard_logger = TensorBoardLogger(save_dir="logs/multi_subject/")
    
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint = ModelCheckpoint(
        monitor='val_acc',
        dirpath=CHECKPOINT_DIR,
        filename=f'{MODEL_NAME}_best_{timestamp}',
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
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCH,
        logger=tensorboard_logger,
        callbacks=[lr_monitor, checkpoint, early_stopping],
        log_every_n_steps=10,
    )
    trainer.fit(model_wrapper)
    
    # Test the model
    print("Testing model...")
    trainer.test(model_wrapper, ckpt_path=checkpoint.best_model_path)
    
    print(f"Training and evaluation completed. Model saved to {checkpoint.best_model_path}")
    return checkpoint.best_model_path

def predict_hand_movement(sample, model_path):
    """
    Predict hand movement from a single EEG sample using the trained model
    
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

if __name__ == "__main__":
    main()
