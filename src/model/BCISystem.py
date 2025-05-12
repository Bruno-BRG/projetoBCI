# Standard library imports
import os
import logging
from datetime import datetime

# Third-party imports
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Local imports
from .EEGClassificationModel import EEGClassificationModel
from .LightningEEGModel import LightningEEGModel

def get_device():
    """Helper function to get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class BCISystem:
    def __init__(self, model_path=None):
        self.device = get_device()  # Use global device
        self.model = None
        self.eeg_channel = None
        self.calibration_data = {'X': [], 'y': []}
        self.is_calibrated = False
        self.model_path = model_path

    def _get_best_device(self):
        return get_device()  # Use global device function

    def initialize_model(self, eeg_channel):
        # Accept either channel count or list of channel names
        if isinstance(eeg_channel, (list, tuple)):
            ch_count = len(eeg_channel)
        else:
            ch_count = eeg_channel
        self.eeg_channel = ch_count
        
        # Create base model
        base_model = EEGClassificationModel(eeg_channel=ch_count, dropout=0.125)
        
        # Create a dummy input to initialize the classifier
        # This ensures the classifier exists before loading the state dict
        with torch.no_grad():
            dummy_input = torch.zeros((1, ch_count, 125), dtype=torch.float32)
            _ = base_model(dummy_input)  # This will initialize the classifier
        
        # Wrap with Lightning module
        self.model = LightningEEGModel(base_model, learning_rate=5e-4)
        
        # Ensure consistent dtype throughout model - use float32 for better compatibility
        self.model = self.model.float()
        self.model = self.model.to(self.device)
        
        # Try to load checkpoint: first attempt raw model state into base_model
        if self.model_path and os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location=self.device)
                # Attempt loading into base model
                try:
                    base_model.load_state_dict(state)
                    print(f"Loaded raw model state into base model from {self.model_path}")
                    self.is_calibrated = True
                except Exception:
                    # If raw state fails, attempt Lightning checkpoint
                    try:
                        # wrap and load Lightning checkpoint
                        lightning_ckpt = state
                        self.model.load_state_dict(lightning_ckpt, strict=False)
                        self.is_calibrated = True
                        print(f"Loaded Lightning checkpoint from {self.model_path}")
                    except Exception as e:
                        print(f"Could not load checkpoint: {str(e)}")
                        print("Starting with a fresh model - requires calibration")
                        self.is_calibrated = False
            except Exception as e:
                print(f"Error loading checkpoint file: {str(e)}")
                self.is_calibrated = False

    def add_calibration_sample(self, eeg_data, label):
        """Adiciona uma amostra de calibração"""
        self.calibration_data['X'].append(eeg_data)
        self.calibration_data['y'].append(label)

    def train_calibration(self, num_epochs=100, batch_size=10, learning_rate=5e-4):
        """Trains the model with the calibration data using Lightning"""
        if len(self.calibration_data['X']) < 2:
            raise ValueError("Need at least 2 calibration samples")

        X = np.array(self.calibration_data['X'])
        y = np.array(self.calibration_data['y'])
        
        # Create datasets
        train_size = int(0.8 * len(X))
        indices = torch.randperm(len(X))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Use float32 tensors to match model dtype
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                     torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                   torch.tensor(y_val, dtype=torch.float32))
        
        # Create data loaders with workers for better performance
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

        # Configure training
        logger = TensorBoardLogger("lightning_logs", name="bci_model")
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            verbose=True,
            mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best_model",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )

        # Initialize trainer
        trainer = Trainer(
            max_epochs=num_epochs,
            accelerator="auto",
            devices=1,
            logger=logger,
            callbacks=[early_stopping, checkpoint_callback],
            log_every_n_steps=1
        )

        # Store the initial model class and parameters for reloading
        model_class = self.model.__class__
        model_params = {
            "model": self.model.model,
            "learning_rate": learning_rate
        }

        # Train the model
        trainer.fit(
            self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

        # Load best model - fixed to use classmethod on the class, not instance
        self.model = model_class.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            **model_params
        )
        
        if self.model_path:
            torch.save(self.model.state_dict(), self.model_path)
        
        self.is_calibrated = True

    def predict_movement(self, eeg_data):
        """Predicts imagined movement from EEG data with balanced confidence scoring"""
        if not self.is_calibrated:
            raise ValueError("System needs to be calibrated first")

        self.model.eval()
        with torch.no_grad():
            # Convert the input to the same dtype as the model
            input_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Make sure we have the expected time dimension
            if input_tensor.shape[2] != 125:
                if input_tensor.shape[2] > 125:
                    input_tensor = input_tensor[:, :, :125]  # Truncate
                else:
                    pad_size = 125 - input_tensor.shape[2]
                    input_tensor = F.pad(input_tensor, (0, pad_size), "constant", 0)  # Pad
            
            output = self.model(input_tensor)
            
            # Get raw logit and convert to probability with sigmoid
            logit = output.item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
            
            # Calculate confidences for Left and Right
            left_confidence = 1 - prob  # Probability of Left
            right_confidence = prob     # Probability of Right
            
            # Debug output to check prediction distribution
            print(f"DEBUG: Raw probability: {prob:.4f}, Left: {left_confidence:.4f}, Right: {right_confidence:.4f}")
            
            # Define decision boundaries
            LEFT_THRESHOLD = 0.40       # If prob < 0.40, predict Left
            RIGHT_THRESHOLD = 0.60      # If prob > 0.60, predict Right
            
            # Make prediction based on raw probability
            if prob < LEFT_THRESHOLD:
                # Left prediction - directly use the left confidence from model
                # Convert to percentage for display
                return "Left", left_confidence
            elif prob > RIGHT_THRESHOLD:
                # Right prediction - directly use the right confidence from model
                # Convert to percentage for display
                return "Right", right_confidence
            else:
                # Uncertain prediction - calculate uncertainty confidence
                # Higher when closer to 0.5 (maximum uncertainty)
                distance_from_center = abs(prob - 0.5)
                uncertain_confidence = 1.0 - (distance_from_center * 2)  # Scales from 0-1
                return "Uncertain", uncertain_confidence

def create_bci_system(model_path="checkpoints/bci_model.pth"):
    """Cria uma nova instância do sistema BCI"""
    return BCISystem(model_path=model_path)
