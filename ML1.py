import os
import mne
from mne.io import concatenate_raws
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
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
    """Load EEG data from all CSV files in project root and apply augmentation"""
    # find csv files for this subject under eeg_data
    subj_dir = os.path.join('eeg_data', 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0', f'S{subject_id:03d}')
    csv_files = glob.glob(os.path.join(subj_dir, '*_csv*.csv'))
    if not csv_files:
        raise FileNotFoundError("No CSV files found for load_and_process_data")
    X_list, y_list = [], []
    for path in csv_files:
        # Read CSV using python engine, skip lines starting with '%', and skip bad lines
        df = pd.read_csv(path, comment='%', engine='python', on_bad_lines='skip')
        # Keep only EEG channels (EXG Channel columns)
        eeg_cols = [col for col in df.columns if col.startswith('EXG Channel')]
        df_eeg = df[eeg_cols]
        arr = df_eeg.values
        # Assume rows=time, cols=channels or vice versa
        if arr.shape[0] < arr.shape[1]:
            data = arr
        else:
            data = arr.T
        X_list.append(data)
        # Simple label from filename
        lower = os.path.basename(path).lower()
        if 'left' in lower:
            y_list.append(0)
        elif 'right' in lower:
            y_list.append(1)
        else:
            y_list.append(0)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    EEG_CHANNEL = X.shape[1]
    # Data augmentation if requested
    if augment:
        X_aug = EEGAugmentation.augment_data(X)
        y_aug = np.repeat(y, X_aug.shape[0] // y.shape[0])
        X = np.concatenate([X, X_aug], axis=0)
        y = np.concatenate([y, y_aug], axis=0)
    return X, y, EEG_CHANNEL

def load_local_eeg_data(subject_id=1, augment=True):
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
        if os.path.exists(img_path):
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
        self.eeg_channel = eeg_channel
        self.model = EEGClassificationModel(eeg_channel=eeg_channel, dropout=0.125)
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

    def train_calibration(self, num_epochs=10, batch_size=32, learning_rate=1e-3):
        """Treina o modelo com os dados de calibração"""
        if len(self.calibration_data['X']) < 10:
            raise ValueError("Necessário pelo menos 10 amostras de calibração")

        X = np.array(self.calibration_data['X'])
        y = np.array(self.calibration_data['y'])

        # Aplica data augmentation nos dados de calibração
        X_aug = EEGAugmentation.augment_data(X)
        y_aug = np.repeat(y, X_aug.shape[0] // y.shape[0])

        # Combina dados originais e aumentados
        X = np.concatenate([X, X_aug], axis=0)
        y = np.concatenate([y, y_aug])

        # Split data into train and validation sets
        train_size = int(0.8 * len(X))
        indices = torch.randperm(len(X))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create train and validation datasets
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Prepare data loaders
        train_dataset = TensorDataset(torch.DoubleTensor(X_train), torch.DoubleTensor(y_train))
        val_dataset = TensorDataset(torch.DoubleTensor(X_val), torch.DoubleTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize training components
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        tracker = ModelTracker(log_dir="training_runs")

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_preds, train_labels = train_epoch(
                self.model, train_loader, criterion, optimizer, 
                self.device, 'train', tracker
            )
            tracker.log_metrics('train', train_loss, train_preds, train_labels, epoch)

            # Validation phase
            with torch.no_grad():
                val_loss, val_preds, val_labels = train_epoch(
                    self.model, val_loader, criterion, None,
                    self.device, 'val', tracker
                )
                tracker.log_metrics('val', val_loss, val_preds, val_labels, epoch)

        self.is_calibrated = True
        if self.model_path:
            # Ensure checkpoint directory exists
            ckpt_dir = os.path.dirname(self.model_path)
            if ckpt_dir and not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
         
        return tracker.plot_training_history()

    def predict_movement(self, eeg_data):
        """Prediz o movimento imaginado a partir dos dados de EEG"""
        if not self.is_calibrated:
            raise ValueError("Sistema precisa ser calibrado primeiro")

        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.DoubleTensor(eeg_data).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            prediction = "Left" if output.item() < 0.5 else "Right"
            confidence = abs(output.item() - 0.5) * 2
            confidence = min(1.0, confidence)  # Limita a confiança a 100%
            return prediction, confidence

def create_bci_system(model_path="checkpoints/bci_model.pth"):
    """Cria uma nova instância do sistema BCI"""
    return BCISystem(model_path=model_path)