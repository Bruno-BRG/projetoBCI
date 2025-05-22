"""
Demo script for running the EEG Classification Model adapted from the notebook.
This script shows how to train, evaluate, and use the model for predictions.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.model.EEGDataset import EEGDataset
from src.model.EEGClassificationModel import EEGClassificationModel
from src.model.ModelWrapper import ModelWrapper
from src.model.EEGDataLoader import load_and_process_data

# Set random seed for reproducibility
SEED = 42
L.seed_everything(SEED, workers=True)

def main():
    """Main function to demonstrate model training and evaluation."""
    # Hyperparameters
    MAX_EPOCH = 100
    BATCH_SIZE = 10
    LR = 5e-4
    MODEL_NAME = "EEGClassificationModel"
    CHECKPOINT_DIR = "checkpoints"
    
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Load the data (from CSV files)
    print("Loading data...")
    X, y, eeg_channel = load_and_process_data(subject_id=1)
    print(f"Loaded data: X shape={X.shape}, y shape={y.shape}, channels={eeg_channel}")
    
    # Create the dataset
    eeg_dataset = EEGDataset(x=X, y=y)
    
    # Create the model with the correct number of channels
    model = EEGClassificationModel(eeg_channel=eeg_channel, dropout=0.125)
    
    # Wrap the model for training
    model_wrapper = ModelWrapper(
        arch=model, 
        dataset=eeg_dataset, 
        batch_size=BATCH_SIZE, 
        lr=LR, 
        max_epoch=MAX_EPOCH
    )
    
    # Setup loggers
    tensorboard_logger = TensorBoardLogger(save_dir="logs/")
    
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint = ModelCheckpoint(
        monitor='val_acc',
        dirpath=CHECKPOINT_DIR,
        filename=f'{MODEL_NAME}_best',
        mode='max',
    )
    early_stopping = EarlyStopping(
        monitor="val_acc", 
        min_delta=0.00, 
        patience=3, 
        verbose=False, 
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
        log_every_n_steps=5,
    )
    trainer.fit(model_wrapper)
    
    # Test the model
    print("Testing model...")
    trainer.test(model_wrapper, ckpt_path=checkpoint.best_model_path)
    
    # Example inference
    print("\nRunning example predictions...")
    for _ in range(3):
        # Select a random sample from the test set
        N_SAMPLE = X.shape[0]
        sample_idx = np.random.randint(0, N_SAMPLE)
        sample = X[sample_idx]
        
        # Run prediction
        prediction = predict_hand_movement(
            sample, 
            model_wrapper, 
            os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.ckpt")
        )
        
        # Display results
        classes = ["left", "right"]
        actual = classes[y[sample_idx]]
        print(f"Prediction: {prediction}, Actual: {actual}")
        
        # Plot the EEG data
        plt.figure(figsize=(10, 4))
        plt.plot(sample.T)
        plt.title(f"EEG Sample - Actual: {actual}, Predicted: {prediction}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(f"prediction_example_{_}.png")
        plt.close()

def predict_hand_movement(sample, model_wrapper, checkpoint_path):
    """
    Predict hand movement from a single EEG sample
    
    Args:
        sample: EEG data with shape (channels, time)
        model_wrapper: Trained ModelWrapper instance
        checkpoint_path: Path to the best model checkpoint
        
    Returns:
        Prediction (left or right)
    """
    # Create inference dataset
    inference_dataset = EEGDataset.inference_dataset(sample)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        shuffle=False,
    )
    
    # Predict
    trainer = L.Trainer()
    prediction = trainer.predict(
        model=model_wrapper,
        dataloaders=dataloader,
        ckpt_path=checkpoint_path,
    )[0]
    
    # Convert to binary prediction
    pred_class = int(torch.sigmoid(prediction) > 0.5)
    classes = ["left", "right"]
    return classes[pred_class]

if __name__ == "__main__":
    main()
