import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

class ModelTracker:
    def __init__(self, log_dir="runs"):
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

    def log(self, phase, loss, preds, labels, epoch):
        """Log metrics for a training/validation phase"""
        preds = preds.view(-1)
        labels = labels.view(-1)
        
        # Convert predictions to binary classes
        pred_classes = (preds > 0.5).float().cpu().numpy()
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
        self.save_plots()
    
    def save_plots(self):
        """Save training and validation plots"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle('Training Metrics', fontsize=16, y=0.95)
        
        # Plot losses
        train_epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(train_epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        if self.val_losses:
            val_epochs = range(1, len(self.val_losses) + 1)
            ax1.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(train_epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        if self.val_accuracies:
            val_acc_epochs = range(1, len(self.val_accuracies) + 1)
            ax2.plot(val_acc_epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_metrics.png'))
        plt.close()

    def save_curves(self):
        """Save and return the path to the training curves plot"""
        self.save_plots()
        return os.path.join(self.plot_dir, 'training_metrics.png')

# Global device configuration
def get_device():
    if not hasattr(get_device, 'device'):
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            get_device.device = torch.device('cuda')
            print(f"GPU found: {device_name}")
        else:
            get_device.device = torch.device('cpu')
            print("No GPU found, using CPU")
    return get_device.device



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
