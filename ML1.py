import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import pandas as pd
import glob

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention block
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feedforward block
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class EEGClassificationModel(nn.Module):
    def __init__(self, eeg_channel, dropout=0.1):
        super().__init__()
        
        # First conv block - temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(eeg_channel, 25, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm1d(25),
            nn.ReLU(),
        )
        
        # Second conv block - spatial filter
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(25, 25, kernel_size=eeg_channel, groups=25),
            nn.BatchNorm1d(25),
            nn.ReLU(),
        )
        
        # Conv Pool Block 1 (25 -> 50)
        self.conv_pool1 = nn.Sequential(
            nn.Conv1d(25, 50, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        
        # Conv Pool Block 2 (50 -> 100)
        self.conv_pool2 = nn.Sequential(
            nn.Conv1d(50, 100, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        
        # Conv Pool Block 3 (100 -> 200)
        self.conv_pool3 = nn.Sequential(
            nn.Conv1d(100, 200, kernel_size=10, stride=1, padding=5),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        
        # Adaptive pooling to handle variable input lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(200, 1)  # Binary classification (left vs right hand)
        )

    def forward(self, x):
        # Input shape: [batch, channels, time]
        
        # Temporal convolution
        x = self.temporal_conv(x)
        
        # Spatial filter
        x = self.spatial_conv(x)
        
        # Conv-pool blocks
        x = self.conv_pool1(x)
        x = self.conv_pool2(x)
        x = self.conv_pool3(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        return x

class EEGDataset(data.Dataset):
    def __init__(self, x, y=None, inference=False):
        super().__init__()

        N_SAMPLE = x.shape[0]
        val_idx = int(0.9 * N_SAMPLE)
        train_idx = int(0.81 * N_SAMPLE)

        if not inference:
            self.train_ds = {
                'x': x[:train_idx],
                'y': y[:train_idx],
            }
            self.val_ds = {
                'x': x[train_idx:val_idx],
                'y': y[train_idx:val_idx],
            }
            self.test_ds = {
                'x': x[val_idx:],
                'y': y[val_idx:],
            }

class EEGAugmentation:
    augmentation_count = 5  # Number of augmented versions to create per sample

    @staticmethod
    def time_shift(data, max_shift=10):
        """Apply random time shift to the signal"""
        shifted_data = np.roll(data, np.random.randint(-max_shift, max_shift), axis=-1)
        return shifted_data
    
    @staticmethod
    def add_gaussian_noise(data, mean=0, std=0.1):
        """Add random Gaussian noise to the signal"""
        noise = np.random.normal(mean, std * np.std(data), data.shape)
        return data + noise
    
    @staticmethod
    def scale_amplitude(data, scale_range=(0.8, 1.2)):
        """Scale signal amplitude by random factor"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale
    
    @staticmethod
    def augment_data(data):
        """Apply all augmentations to create multiple versions of each sample"""
        augmented_samples = []
        # Keep original data
        augmented_samples.append(data)
        
        # For each original sample
        for i in range(len(data)):
            sample = data[i:i+1]  # Keep dimensions (1, channels, time)
            # Create augmented versions
            for _ in range(EEGAugmentation.augmentation_count):
                aug_sample = sample.copy()
                # Apply augmentations in sequence with different random values
                if np.random.random() > 0.3:
                    aug_sample = EEGAugmentation.time_shift(aug_sample)
                if np.random.random() > 0.3:
                    aug_sample = EEGAugmentation.add_gaussian_noise(aug_sample)
                if np.random.random() > 0.3:
                    aug_sample = EEGAugmentation.scale_amplitude(aug_sample)
                augmented_samples.append(aug_sample)
        
        # Stack all samples together
        return np.concatenate(augmented_samples, axis=0)

# Define the 16 standard electrodes in the desired order
WANTED_CHANNELS = ['C3','C4','Fp1','Fp2','F7','F3','F4','F8',
                   'T7','T8','P7','P3','P4','P8','O1','O2']

def load_and_process_data(subject_id=1, augment=False):  # Changed default to False
    """Load EEG data from CSV files for imagined left/right hand movement and apply epoching"""
    # Find relevant run files (4,8,12 contain imagined left/right hand movement)
    runs = [4, 8, 12]
    subj_dir = os.path.join('eeg_data', 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0', f'S{subject_id:03d}')
    X_list, y_list = [], []

    for run in runs:
        csv_path = os.path.join(subj_dir, f'S{subject_id:03d}R{run:02d}_csv_openbci.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Required CSV file not found: {csv_path}")

        # Read CSV data
        df = pd.read_csv(csv_path, comment='%', engine='python', on_bad_lines='skip')
        eeg_cols = [col for col in df.columns if col.startswith('EXG Channel')]
        data = df[eeg_cols].values.T  # channels x samples

        # Find events from Annotations column
        # T1 marks the start of left hand trials
        # T2 marks the start of right hand trials
        annotations = df['Annotations'].fillna('')
        event_indices = []
        event_types = []

        for idx, annotation in enumerate(annotations):
            if (annotation in ['T1', 'T2']):
                event_indices.append(idx)
                # Convert T1->0 (left), T2->1 (right)
                event_types.append(0 if annotation == 'T1' else 1)

        # Extract epochs: 1s to 4.1s after each event
        sfreq = 125  # OpenBCI sample rate
        samples_per_epoch = int(3.1 * sfreq)  # 3.1s window (4.1s - 1s)
        start_offset = int(sfreq)  # 1s offset

        for evt_idx, evt_type in zip(event_indices, event_types):
            # Extract epoch
            start_idx = evt_idx + start_offset
            end_idx = start_idx + samples_per_epoch
            if end_idx <= data.shape[1]:  # Only use if we have enough samples
                epoch = data[:, start_idx:end_idx]
                X_list.append(epoch)
                y_list.append(evt_type)

    if not X_list:
        raise ValueError(f"No valid epochs found for subject {subject_id}")

    X = np.stack(X_list)
    y = np.array(y_list)
    
    if augment:
        # apply augmentation like previous version
        X = EEGAugmentation.augment_data(X)
        y = np.repeat(y, EEGAugmentation.augmentation_count+1)
    
    return X, y, X.shape[1]  # Return data, labels, and channel count

def load_local_eeg_data(subject_id=1, augment=False):
    """Load EEG data from all CSV files in project root (alias of load_and_process_data)"""
    return load_and_process_data(subject_id, augment)

def load_model(eeg_channel):
    model = EEGClassificationModel(eeg_channel=eeg_channel, dropout=0.125)
    # You can load pretrained weights here if needed
    return model

class ModelTracker:
    def __init__(self, log_dir: str = "runs"):
        self.writer = SummaryWriter(log_dir)
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Create directory for saving plots
        self.plot_dir = "training_plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def log_metrics(self, phase: str, loss: float, predictions: torch.Tensor, labels: torch.Tensor, epoch: int):
        """Log metrics for a training/validation phase"""
        # Ensure predictions and labels have the same shape
        predictions = predictions.view(-1)  # Flatten predictions
        labels = labels.view(-1)  # Flatten labels
        
        # Convert predictions to binary classes
        pred_classes = (predictions > 0.5).float().cpu().numpy()
        true_labels = labels.cpu().numpy()
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, pred_classes)
        
        # Log to tensorboard
        self.writer.add_scalar(f'{phase}/Loss', loss, epoch)
        self.writer.add_scalar(f'{phase}/Accuracy', accuracy, epoch)
        
        # Store metrics
        if phase == 'train':
            self.train_losses.append(loss)
            self.train_accuracies.append(accuracy)
        else:
            self.val_losses.append(loss)
            self.val_accuracies.append(accuracy)
        
        # Every 5 epochs, log confusion matrix
        if epoch % 5 == 0:
            cm = confusion_matrix(true_labels, pred_classes)
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            plt.title(f'{phase} Confusion Matrix - Epoch {epoch}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            self.writer.add_figure(f'{phase}/Confusion_Matrix', fig, epoch)
            plt.close()

        # Save plots after each epoch
        self.save_training_plots()

    def save_training_plots(self):
        """Save training and validation plots"""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle('Training Metrics', fontsize=16, y=0.95)
        
        # Plot losses
        train_epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(train_epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        if len(self.val_losses) > 0:
            val_epochs = range(1, len(self.val_losses) + 1)
            ax1.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(train_epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        if len(self.val_accuracies) > 0:
            val_acc_epochs = range(1, len(self.val_accuracies) + 1)
            ax2.plot(val_acc_epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        ax2.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_metrics.png'))
        plt.close()

    def plot_training_history(self):
        """Plot training history"""
        # Load and return the saved plot
        img_path = os.path.join(self.plot_dir, 'training_metrics.png')
        if (os.path.exists(img_path)):
            return plt.imread(img_path)
        else:
            return self.save_training_plots()

def train_epoch(model: nn.Module, 
                dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                phase: str,
                tracker: ModelTracker) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Train/validate for one epoch"""
    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() if phase == 'train' else None

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            outputs = outputs.squeeze()
            
            # Handle single sample case
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
                
            loss = criterion(outputs, labels.float())

            if phase == 'train':
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        all_preds.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    epoch_loss = running_loss / len(dataloader)
    
    # Stack tensors properly, handling single-item batches
    epoch_preds = torch.cat([p.view(-1) for p in all_preds])
    epoch_labels = torch.cat([l.view(-1) for l in all_labels])

    return epoch_loss, epoch_preds, epoch_labels

def evaluate_model(model: nn.Module, 
                  test_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> Dict[str, Any]:
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_classes = (all_preds > 0.5).astype(int)

    return {
        'accuracy': accuracy_score(all_labels, pred_classes),
        'classification_report': classification_report(all_labels, pred_classes),
        'confusion_matrix': confusion_matrix(all_labels, pred_classes)
    }

class BCISystem:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.eeg_channel = None
        self.calibration_data = {'X': [], 'y': []}
        self.is_calibrated = False
        self.model_path = model_path

    def initialize_model(self, eeg_channel):
        # Accept either channel count or list of channel names
        if isinstance(eeg_channel, (list, tuple)):
            ch_count = len(eeg_channel)
        else:
            ch_count = eeg_channel
        self.eeg_channel = ch_count
        self.model = EEGClassificationModel(eeg_channel=ch_count, dropout=0.125)
        self.model = self.model.double()  # Convert model to double precision
        self.model.to(self.device)
        if self.model_path and os.path.exists(self.model_path):
            # Load checkpoint and filter for matching parameter shapes only
            state = torch.load(self.model_path)
            model_dict = self.model.state_dict()
            filtered_state = {k: v for k, v in state.items()
                              if k in model_dict and v.size() == model_dict[k].size()}
            self.model.load_state_dict(filtered_state, strict=False)
            # Warn about keys not loaded
            missing = set(model_dict.keys()) - set(filtered_state.keys())
            unexpected = set(state.keys()) - set(filtered_state.keys())
            if missing:
                print(f"Warning: missing keys in checkpoint: {missing}")
            if unexpected:
                print(f"Warning: unexpected keys ignored: {unexpected}")
            # Mark calibrated if any weights loaded
            self.is_calibrated = len(filtered_state) > 0

    def add_calibration_sample(self, eeg_data, label):
        """Adiciona uma amostra de calibração"""
        self.calibration_data['X'].append(eeg_data)
        self.calibration_data['y'].append(label)

    def train_calibration(self, num_epochs=20, batch_size=4, learning_rate=1e-3):
        """Trains the model with the calibration data"""
        if len(self.calibration_data['X']) < 2:
            raise ValueError("Need at least 2 calibration samples")

        X = np.array(self.calibration_data['X'])
        y = np.array(self.calibration_data['y'])
        
        # Split data into train and validation sets
        train_size = int(0.8 * len(X))
        indices = torch.randperm(len(X))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create train and validation datasets
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Adjust batch size if we have few samples
        actual_batch_size = min(batch_size, len(X_train))

        # Prepare data loaders
        train_dataset = TensorDataset(torch.DoubleTensor(X_train), torch.DoubleTensor(y_train))
        val_dataset = TensorDataset(torch.DoubleTensor(X_val), torch.DoubleTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=actual_batch_size)

        # Initialize training components
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        tracker = ModelTracker(log_dir="training_runs")

        # Training loop
        best_val_loss = float('inf')
        patience = 7
        no_improve = 0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)  # Shape: [batch_size, 1]
                outputs = outputs.view(-1)    # Flatten to [batch_size]
                labels = labels.float()
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu())
                train_labels.extend(labels.cpu())
            
            train_loss /= len(train_loader)
            train_preds = torch.stack(train_preds)
            train_labels = torch.stack(train_labels)
            tracker.log_metrics('train', train_loss, train_preds, train_labels, epoch)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)  # Shape: [batch_size, 1]
                    outputs = outputs.view(-1)    # Flatten to [batch_size]
                    labels = labels.float()
                    
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_preds.extend(outputs.detach().cpu())
                    val_labels.extend(labels.cpu())
            
            val_loss /= len(val_loader)
            val_preds = torch.stack(val_preds)
            val_labels = torch.stack(val_labels)
            tracker.log_metrics('val', val_loss, val_preds, val_labels, epoch)

            # Rest of the training loop...
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                if self.model_path:
                    torch.save(self.model.state_dict(), self.model_path)
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

        self.is_calibrated = True
        return tracker.plot_training_history()

    def predict_movement(self, eeg_data):
        """Predicts imagined movement from EEG data with confidence scores for all outcomes"""
        if not self.is_calibrated:
            raise ValueError("System needs to be calibrated first")

        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.DoubleTensor(eeg_data).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            
            # Get raw logit and convert to probability with sigmoid
            logit = output.item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
            
            # Calculate confidences for Left and Right
            left_confidence = (1 - prob)  # Scale to percentage
            right_confidence = prob  # Scale to percentage
            
            # Calculate uncertain confidence - higher when both left/right are close to 50%
            confidence_diff = abs(left_confidence - right_confidence)
            uncertain_confidence = max(0, 100 - confidence_diff)
            
            # Define minimum confidence threshold for Left/Right predictions (40%)
            MIN_CONFIDENCE = 40
            
            # If the difference between left/right is small, predict uncertain
            if uncertain_confidence > 40:  # confidence_diff < 60
                return "Uncertain", uncertain_confidence
            
            # Otherwise return the highest confidence prediction if it meets threshold
            if max(left_confidence, right_confidence) >= MIN_CONFIDENCE:
                if left_confidence > right_confidence:
                    return "Left", min(100, left_confidence)  # Cap at 100%
                else:
                    return "Right", min(100, right_confidence)  # Cap at 100%
            
            # If no prediction meets threshold, return uncertain
            return "Uncertain", uncertain_confidence

def create_bci_system(model_path="checkpoints/bci_model.pth"):
    """Cria uma nova instância do sistema BCI"""
    return BCISystem(model_path=model_path)

class ModelWrapper(L.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()  # Remove extra dimension to match target
        y = y.squeeze()  # Ensure target is also squeezed
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.train_accuracy.update(y_hat, y)
        acc = self.train_accuracy.compute()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()  # Remove extra dimension to match target
        y = y.squeeze()  # Ensure target is also squeezed
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.val_accuracy.update(y_hat, y)
        acc = self.val_accuracy.compute()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()  # Remove extra dimension to match target
        y = y.squeeze()  # Ensure target is also squeezed
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.test_accuracy.update(y_hat, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

class MultiSubjectTest:
    def __init__(self, train_samples=20, test_samples=10):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.training_history = {
            'train_losses': [], 'val_losses': [],
            'train_accs': [], 'val_accs': []
        }
        print(f"\nInitializing MultiSubjectTest with {train_samples} training and {test_samples} test samples per subject")
        print(f"Using device: {self.device}")
    
    def collect_balanced_samples(self, subject_id, n_samples):
        """Collect balanced samples (equal left/right) from a subject"""
        try:
            X, y, ch = load_local_eeg_data(subject_id, augment=False)
            left_idx = (y == 0).nonzero()[0]
            right_idx = (y == 1).nonzero()[0]
            
            samples_per_class = n_samples // 2
            if len(left_idx) < samples_per_class or len(right_idx) < samples_per_class:
                print(f"Subject {subject_id:03d}: Insufficient samples (L:{len(left_idx)}, R:{len(right_idx)})")
                return None, None
                
            selected_left = np.random.choice(left_idx, samples_per_class, replace=False)
            selected_right = np.random.choice(right_idx, samples_per_class, replace=False)
            
            selected_idx = np.concatenate([selected_left, selected_right])
            print(f"Subject {subject_id:03d}: Successfully collected {n_samples} balanced samples")
            return X[selected_idx], y[selected_idx]
        except Exception as e:
            print(f"Subject {subject_id:03d}: Error loading data - {str(e)}")
            return None, None

    def prepare_datasets(self):
        """Prepare training and testing datasets from multiple subjects"""
        print("\nStarting dataset preparation...")
        train_X, train_y = [], []
        test_X, test_y = [], []
        
        successful_subjects = 0
        total_subjects = 0
        
        # Try subjects from 1 to 109 (maximum in dataset)
        for subject_id in range(1, 110):
            total_subjects += 1
            try:
                print(f"\nProcessing Subject {subject_id:03d}...")
                
                # Get training samples
                X_train, y_train = self.collect_balanced_samples(subject_id, self.train_samples)
                if X_train is not None:
                    train_X.append(X_train)
                    train_y.append(y_train)
                    
                    # Get testing samples (different from training)
                    X_test, y_test = self.collect_balanced_samples(subject_id, self.test_samples)
                    if X_test is not None:
                        test_X.append(X_test)
                        test_y.append(y_test)
                        successful_subjects += 1
                
            except Exception as e:
                print(f"Error processing subject {subject_id}: {str(e)}")
                continue
        
        if not train_X or not test_X:
            raise ValueError("Could not collect enough samples from any subject")
            
        final_train_X = np.concatenate(train_X)
        final_train_y = np.concatenate(train_y)
        final_test_X = np.concatenate(test_X)
        final_test_y = np.concatenate(test_y)
        
        print(f"\nDataset preparation completed:")
        print(f"Successfully processed {successful_subjects} out of {total_subjects} subjects")
        print(f"Total training samples: {len(final_train_X)}")
        print(f"Total testing samples: {len(final_test_X)}")
        print(f"Training data shape: {final_train_X.shape}")
        print(f"Testing data shape: {final_test_X.shape}")
        
        return final_train_X, final_train_y, final_test_X, final_test_y

    def train_and_evaluate(self, num_epochs=20, batch_size=32, learning_rate=1e-3):
        """Train on collected samples and evaluate performance"""
        try:
            print("\nStarting training and evaluation process...")
            print(f"Parameters: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
            
            # Prepare datasets
            print("\nPreparing datasets...")
            train_X, train_y, test_X, test_y = self.prepare_datasets()
            
            # Initialize model
            print("\nInitializing model...")
            self.model = EEGClassificationModel(eeg_channel=train_X.shape[1])
            self.model = self.model.double().to(self.device)
            
            # Convert to tensors
            train_X = torch.from_numpy(train_X).to(self.device)
            train_y = torch.from_numpy(train_y).to(self.device)
            test_X = torch.from_numpy(test_X).to(self.device)
            test_y = torch.from_numpy(test_y).to(self.device)
            
            # Create data loaders
            print("\nCreating data loaders...")
            train_dataset = TensorDataset(train_X, train_y)
            test_dataset = TensorDataset(test_X, test_y)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Training setup
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            print("\nStarting training loop...")
            
            # Training loop
            for epoch in range(num_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print("Training phase...")
                
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = self.model(inputs).squeeze()
                    loss = criterion(outputs, labels.float())
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    pred = (outputs > 0.5).float()
                    train_correct += (pred == labels).sum().item()
                    train_total += labels.size(0)
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Batch {batch_idx+1}/{len(train_loader)}")
                
                avg_train_loss = train_loss / len(train_loader)
                train_acc = train_correct / train_total
                
                # Validation phase
                print("Validation phase...")
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_idx, (inputs, labels) in enumerate(test_loader):
                        outputs = self.model(inputs).squeeze()
                        loss = criterion(outputs, labels.float())
                        val_loss += loss.item()
                        pred = (outputs > 0.5).float()
                        val_correct += (pred == labels).sum().item()
                        val_total += labels.size(0)
                
                avg_val_loss = val_loss / len(test_loader)
                val_acc = val_correct / val_total
                
                # Store metrics
                self.training_history['train_losses'].append(avg_train_loss)
                self.training_history['val_losses'].append(avg_val_loss)
                self.training_history['train_accs'].append(train_acc)
                self.training_history['val_accs'].append(val_acc)
                
                print(f"Epoch {epoch+1} Results:")
                print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%}")
                print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}")
            
            print("\nTraining completed successfully!")
            return self.training_history
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise

