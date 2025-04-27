import mne
from mne.io import concatenate_raws
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

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

        self.conv = nn.Sequential(
            nn.Conv1d(eeg_channel, eeg_channel, 11, 1, padding=5, bias=False),
            nn.BatchNorm1d(eeg_channel),
            nn.ReLU(True),
            nn.Dropout1d(dropout),
            nn.Conv1d(eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False),
            nn.BatchNorm1d(eeg_channel * 2),
        )

        self.transformer = nn.Sequential(
            PositionalEncoding(eeg_channel * 2, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
        )

        self.mlp = nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel // 2, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = x.mean(dim=-1)
        x = self.mlp(x)
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
    @staticmethod
    def time_shift(data, max_shift=10):
        shifted_data = np.roll(data, np.random.randint(-max_shift, max_shift), axis=-1)
        return shifted_data
    
    @staticmethod
    def add_gaussian_noise(data, mean=0, std=0.1):
        noise = np.random.normal(mean, std, data.shape)
        return data + noise
    
    @staticmethod
    def scale_amplitude(data, scale_range=(0.8, 1.2)):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale
    
    @staticmethod
    def augment_data(data, augmentation_count=5):
        augmented_data = []
        # Create augmented versions of each sample
        for i in range(data.shape[0]):
            sample = data[i:i+1]  # Keep the sample's dimensions
            for _ in range(augmentation_count):
                aug_sample = sample.copy()
                # Apply random augmentations
                if np.random.random() > 0.3:
                    aug_sample = EEGAugmentation.time_shift(aug_sample)
                if np.random.random() > 0.3:
                    aug_sample = EEGAugmentation.add_gaussian_noise(aug_sample)
                if np.random.random() > 0.3:
                    aug_sample = EEGAugmentation.scale_amplitude(aug_sample)
                augmented_data.append(aug_sample)
        # Stack all augmented samples while maintaining original dimensions
        return np.vstack(augmented_data)

def load_and_process_data(subject_id=1, augment=True):
    # Load EEG data for a single subject
    physionet_paths = mne.datasets.eegbci.load_data(subject_id, runs=[4, 8, 12])
    
    parts = [
        mne.io.read_raw_edf(
            path,
            preload=True,
            stim_channel='auto',
            verbose='WARNING',
        )
        for path in physionet_paths
    ]
    raw = concatenate_raws(parts)
    events, _ = mne.events_from_annotations(raw)

    # Get EEG channels
    eeg_channel_inds = mne.pick_types(
        raw.info,
        meg=False,
        eeg=True,
        stim=False,
        eog=False,
        exclude='bads',
    )

    EEG_CHANNEL = int(len(eeg_channel_inds))
    epoched = mne.Epochs(
        raw,
        events,
        dict(left=2, right=3),
        tmin=1,
        tmax=4.1,
        proj=False,
        picks=eeg_channel_inds,
        baseline=None,
        preload=True
    )
    X = (epoched.get_data() * 1e3).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)
    
    if augment:
        # Apply data augmentation
        augmented_X = EEGAugmentation.augment_data(X)
        # Combine original and augmented data
        X = np.concatenate([X, augmented_X], axis=0)
        y = np.concatenate([y, np.repeat(y, augmented_X.shape[0] // y.shape[0])])
    
    return X, y, EEG_CHANNEL

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

    def log_metrics(self, phase: str, loss: float, predictions: torch.Tensor, labels: torch.Tensor, epoch: int):
        """Log metrics for a training/validation phase"""
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

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        return fig

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
            loss = criterion(outputs, labels.float())

            if phase == 'train':
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        all_preds.extend(outputs.detach())
        all_labels.extend(labels.detach())

    epoch_loss = running_loss / len(dataloader)
    epoch_preds = torch.stack(all_preds)
    epoch_labels = torch.stack(all_labels)

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