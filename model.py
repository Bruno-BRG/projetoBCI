from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from braindecode.models import EEGInceptionERP
from braindecode.visualization import plot_confusion_matrix
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# --------------------------------------------------------------------- #
# 1. Dataset
# --------------------------------------------------------------------- #
class EEGData(Dataset):
    """Dataset em memória para amostras EEG no formato (N, C, T)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, split: str = "full"):
        assert x.ndim == 3, "Esperado shape (N, C, T)"
        assert len(x) == len(y), "X e y devem ter mesmo comprimento"
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self._set_split_indices(split)

    # --------------- interface Dataset --------------- #
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        return self.x[j], self.y[j]

    # --------------- utilidades --------------- #
    def split(self, which: str) -> "EEGData":
        """Retorna uma nova visão (train/val/test) sem copiar dados."""
        return EEGData(self.x.numpy(), self.y.numpy(), split=which)

    def _set_split_indices(self, split: str):
        n = len(self.x)
        if split == "full":
            self.idx = torch.arange(n)
        else:
            # 70-15-15 % por padrão
            n_train = int(0.7 * n)
            n_val = int(0.15 * n)
            if split == "train":
                self.idx = torch.arange(0, n_train)
            elif split == "val":
                self.idx = torch.arange(n_train, n_train + n_val)
            elif split == "test":
                self.idx = torch.arange(n_train + n_val, n)
            else:
                raise ValueError(f"Split desconhecido: {split}")


# --------------------------------------------------------------------- #
# 2. Modelo de classificação
# --------------------------------------------------------------------- #
class EEGModel(nn.Module):
    """
    Pequeno *wrapper* sobre EEGInceptionERP.

    Args
    ----
    eeg_channel : int  – número de canais (C)
    n_times     : int  – comprimento da janela temporal (T)
    dropout     : float
    """

    def __init__(self,
                 eeg_channel: int,
                 n_times: int = 497,
                 dropout: float = 0.1,
                 sfreq: float = 125.0):
        super().__init__()
        self.net = EEGInceptionERP(
            n_chans=eeg_channel,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=1,
            drop_prob=dropout,
            add_log_softmax=False,   # usaremos BCEWithLogits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T) em microvolts
        return : logits (B, 1)
        """
        assert x.ndim == 3, "Entrada deve ter shape (batch, channels, time)"
        return self.net(x).squeeze(1)   # (B,)


# --------------------------------------------------------------------- #
# 3. Lightning *wrapper* with Real-time Tracking
# --------------------------------------------------------------------- #
class ModelWrapper(pl.LightningModule):
    """Orquestra treino/validação/teste e logging de métricas com tracking em tempo real."""

    def __init__(self,
                 arch: nn.Module,
                 dataset: EEGData,
                 batch_size: int = 32,
                 lr: float = 5e-4,
                 max_epoch: int = 100,
                 enable_tensorboard: bool = True,
                 log_dir: str = "logs"):
        super().__init__()
        
        # Save hyperparameters first
        self.save_hyperparameters(ignore=["arch", "dataset"])
        
        # Initialize core components
        self.arch = arch
        self.criterion = nn.BCEWithLogitsLoss()
        self.dataset = dataset
        
        # Metrics for tracking
        self.train_acc_metric = Accuracy(task="binary")
        self.val_acc_metric = Accuracy(task="binary")
        self.test_acc_metric = Accuracy(task="binary")
          # Real-time tracking variables
        self.batch_losses = {'train': [], 'val': []}
        self.batch_accuracies = {'train': [], 'val': []}
        
        # Validation predictions and labels for confusion matrix
        self.val_predictions = []
        self.val_labels = []
        
        # Epoch-level metrics storage
        self.epoch_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # TensorBoard logging setup
        self.enable_tensorboard = enable_tensorboard
        self.tb_writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                timestamp = int(time.time())
                tb_log_dir = os.path.join(log_dir, f"eeg_model_{timestamp}")
                self.tb_writer = SummaryWriter(tb_log_dir)
                print(f"✓ TensorBoard logging enabled: {tb_log_dir}")
            except ImportError:
                print("⚠ TensorBoard not available, logging disabled")
                self.enable_tensorboard = False
        
        # Training state tracking
        self.epoch_start_time = None
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.training_start_time = None
        
        # Initialize trainer reference to None (will be set by Lightning)
        self.trainer = None    # --------------- forward & step --------------- #
    def forward(self, x):
        return self.arch(x)

    def _common_step(self, batch, stage: str, batch_idx: int = 0):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        
        # Calculate predictions and accuracy
        # For binary classification, convert sigmoid output to binary predictions
        preds = torch.sigmoid(logits)
        binary_preds = (preds > 0.5).float()  # Convert to 0/1 predictions
        
        if stage == "train":
            acc = self.train_acc_metric(binary_preds, y)
        elif stage == "val":
            acc = self.val_acc_metric(binary_preds, y)
        else:  # test
            acc = self.test_acc_metric(binary_preds, y)
        
        # Log step-level metrics
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
          # Store batch-level metrics for real-time tracking
        if stage in ['train', 'val']:
            self.batch_losses[stage].append(loss.item())
            self.batch_accuracies[stage].append(acc.item())
            
            # Collect validation predictions and labels for confusion matrix
            if stage == 'val':
                self.val_predictions.extend(binary_preds.cpu().numpy())
                self.val_labels.extend(y.cpu().numpy())# Log to TensorBoard if enabled
        if self.tb_writer and stage == 'train':
            # Safely get current epoch and dataloader length
            current_epoch = getattr(self.trainer, 'current_epoch', 0) if self.trainer else 0
            train_dataloader = getattr(self.trainer, 'train_dataloader', None) if self.trainer else None
            train_loader_len = len(train_dataloader) if train_dataloader else 1
            global_step = current_epoch * train_loader_len + batch_idx
            self.tb_writer.add_scalar(f'Batch/{stage}_loss', loss.item(), global_step)
            self.tb_writer.add_scalar(f'Batch/{stage}_accuracy', acc.item(), global_step)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test", batch_idx)

    # --------------- epoch hooks for real-time tracking --------------- #    def on_train_epoch_start(self):
        """Called at the start of each training epoch"""
        import time
        self.epoch_start_time = time.time()
          # Clear batch metrics for new epoch
        self.batch_losses = {'train': [], 'val': []}
        self.batch_accuracies = {'train': [], 'val': []}
        
        # Clear validation predictions and labels for new epoch
        self.val_predictions = []
        self.val_labels = []
        
        if self.training_start_time is None:
            self.training_start_time = time.time()
        
        # Safely get current epoch
        current_epoch = getattr(self.trainer, 'current_epoch', 0) if self.trainer else 0
        max_epochs = getattr(self.trainer, 'max_epochs', self.hparams.max_epoch) if self.trainer else self.hparams.max_epoch
        print(f"\n🚀 Starting Epoch {current_epoch + 1}/{max_epochs}")

    def on_train_epoch_end(self):
        """Called at the end of each training epoch - log comprehensive metrics"""
        import time
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
          # Get epoch-level metrics - compute from metric objects for accuracy
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0)
        train_acc = self.train_acc_metric.compute()  # Compute from the metric object
        
        # Store epoch metrics
        self.epoch_metrics['train_loss'].append(float(train_loss))
        self.epoch_metrics['train_acc'].append(float(train_acc))
        self.epoch_metrics['epoch_times'].append(epoch_time)
        
        # Reset the training accuracy metric for next epoch
        self.train_acc_metric.reset()
          # Get current learning rate safely
        current_lr = 0
        if self.trainer and self.trainer.optimizers:
            try:
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            except (IndexError, KeyError):
                current_lr = self.hparams.lr
                
        self.epoch_metrics['learning_rates'].append(current_lr)
        
        # Calculate batch statistics
        if self.batch_losses['train']:
            avg_batch_loss = sum(self.batch_losses['train']) / len(self.batch_losses['train'])
            avg_batch_acc = sum(self.batch_accuracies['train']) / len(self.batch_accuracies['train'])
        else:
            avg_batch_loss = avg_batch_acc = 0        # Log to TensorBoard
        if self.tb_writer:
            current_epoch = getattr(self.trainer, 'current_epoch', 0) if self.trainer else 0
            self.tb_writer.add_scalar('Epoch/train_loss', train_loss, current_epoch)
            self.tb_writer.add_scalar('Epoch/train_accuracy', train_acc, current_epoch)
            self.tb_writer.add_scalar('Epoch/learning_rate', current_lr, current_epoch)
            self.tb_writer.add_scalar('Epoch/epoch_time', epoch_time, current_epoch)
            self.tb_writer.add_scalar('Epoch/avg_batch_loss', avg_batch_loss, current_epoch)
            self.tb_writer.add_scalar('Epoch/avg_batch_accuracy', avg_batch_acc, current_epoch)
        
        current_epoch = getattr(self.trainer, 'current_epoch', 0) if self.trainer else 0
        print(f"📊 Epoch {current_epoch + 1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
        print(f"   Learning Rate: {current_lr:.2e} | Time: {epoch_time:.1f}s")

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch"""
        # Get validation metrics - compute them from the metric objects
        val_loss = self.trainer.callback_metrics.get('val_loss_epoch', 0)
        val_acc = self.val_acc_metric.compute()  # Compute from the metric object
        
        # Store validation metrics
        self.epoch_metrics['val_loss'].append(float(val_loss))
        self.epoch_metrics['val_acc'].append(float(val_acc))
        
        # Update best metrics
        if val_loss < self.best_val_loss:
            self.best_val_loss = float(val_loss)
        if val_acc > self.best_val_acc:
            self.best_val_acc = float(val_acc)        # Reset the validation accuracy metric for next epoch
        self.val_acc_metric.reset()
        
        # Generate and save confusion matrix using braindecode
        self._generate_confusion_matrix(val_acc)
        
        # Log to TensorBoard
        if self.tb_writer:
            current_epoch = getattr(self.trainer, 'current_epoch', 0) if self.trainer else 0
            self.tb_writer.add_scalar('Epoch/val_loss', val_loss, current_epoch)
            self.tb_writer.add_scalar('Epoch/val_accuracy', val_acc, current_epoch)
            self.tb_writer.add_scalar('Epoch/best_val_loss', self.best_val_loss, current_epoch)
            self.tb_writer.add_scalar('Epoch/best_val_accuracy', self.best_val_acc, current_epoch)
        
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
        print(f"   Best Val Loss: {self.best_val_loss:.4f} | Best Val Acc: {self.best_val_acc:.3f}")

    def on_train_end(self):
        """Called when training completes"""
        import time
        total_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        print(f"\n🎉 Training Complete!")
        print(f"   Total Time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"   Total Epochs: {len(self.epoch_metrics['train_loss'])}")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print(f"   Best Val Acc: {self.best_val_acc:.3f}")
        
        # Final TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar('Training/total_time', total_time, 0)
            self.tb_writer.add_scalar('Training/final_best_val_loss', self.best_val_loss, 0)
            self.tb_writer.add_scalar('Training/final_best_val_acc', self.best_val_acc, 0)
            
            # Log hyperparameters and final metrics
            hparam_dict = {
                'lr': self.hparams.lr,
                'batch_size': self.hparams.batch_size,
                'max_epochs': self.hparams.max_epoch
            }
            metric_dict = {
                'best_val_acc': self.best_val_acc,
                'best_val_loss': self.best_val_loss,
                'final_train_acc': self.epoch_metrics['train_acc'][-1] if self.epoch_metrics['train_acc'] else 0            }
            self.tb_writer.add_hparams(hparam_dict, metric_dict)
            self.tb_writer.close()

    # --------------- utility methods --------------- #
    def get_training_summary(self) -> dict:
        """Get comprehensive training summary"""
        import time
        current_epoch = getattr(self.trainer, 'current_epoch', 0) if self.trainer else 0
        current_lr = 0
        if self.trainer and self.trainer.optimizers:
            try:
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            except (IndexError, KeyError):
                current_lr = self.hparams.lr
        
        return {
            'current_epoch': current_epoch,
            'total_epochs': len(self.epoch_metrics['train_loss']),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'epoch_metrics': self.epoch_metrics,
            'current_lr': current_lr,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }

    def plot_training_curves(self, save_path: str = None):
        """Generate and optionally save training curves"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.epoch_metrics['train_loss']:
                print("No training data to plot yet.")
                return None
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('EEG Model Training Progress (Real-time)', fontsize=16, fontweight='bold')
            
            epochs = range(1, len(self.epoch_metrics['train_loss']) + 1)
            
            # Loss curves
            ax1.plot(epochs, self.epoch_metrics['train_loss'], 'b-', label='Training Loss', linewidth=2)
            if self.epoch_metrics['val_loss']:
                val_epochs = range(1, len(self.epoch_metrics['val_loss']) + 1)
                ax1.plot(val_epochs, self.epoch_metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title('Loss Curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy curves
            ax2.plot(epochs, self.epoch_metrics['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
            if self.epoch_metrics['val_acc']:
                val_epochs = range(1, len(self.epoch_metrics['val_acc']) + 1)
                ax2.plot(val_epochs, self.epoch_metrics['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            ax2.set_title('Accuracy Curves')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Learning rate schedule
            if self.epoch_metrics['learning_rates']:
                ax3.plot(epochs, self.epoch_metrics['learning_rates'], 'g-', linewidth=2)
                ax3.set_title('Learning Rate Schedule')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Learning Rate')
                ax3.set_yscale('log')
                ax3.grid(True, alpha=0.3)
              # Epoch times
            if self.epoch_metrics['epoch_times']:
                ax4.plot(epochs, self.epoch_metrics['epoch_times'], 'm-', linewidth=2)
                ax4.set_title('Epoch Duration')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Time (seconds)')
                ax4.grid(True, alpha=0.3)
                plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Training curves saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"Error plotting training curves: {e}")
            return None

    def _generate_confusion_matrix(self, val_acc: float):
        """Generate and save confusion matrix using braindecode visualization"""
        if len(self.val_predictions) == 0 or len(self.val_labels) == 0:
            return
            
        try:
            # Convert predictions and labels to numpy arrays
            predictions = np.array(self.val_predictions)
            labels = np.array(self.val_labels)
            
            # Generate confusion matrix using sklearn
            cm = confusion_matrix(labels, predictions)
            
            # Define class names for binary classification
            class_names = ['Non-Target', 'Target']
            
            # Generate confusion matrix plot using braindecode
            fig = plot_confusion_matrix(
                confusion_mat=cm,
                class_names=class_names,
                figsize=(8, 6),
                with_f1_score=True,
                class_names_fontsize=12
            )
            
            # Get current epoch for logging
            current_epoch = getattr(self.trainer, 'current_epoch', 0) if self.trainer else 0
            
            # Add title with validation accuracy
            plt.suptitle(f'Validation Confusion Matrix - Epoch {current_epoch + 1}\nAccuracy: {val_acc:.3f}', 
                        fontsize=14, y=0.98)
            
            # Log to TensorBoard if available
            if self.tb_writer:
                self.tb_writer.add_figure('Validation/Confusion_Matrix', fig, current_epoch)
            
            # Save to file if possible
            try:
                # Create results directory if it doesn't exist
                results_dir = "results"
                os.makedirs(results_dir, exist_ok=True)
                
                # Save confusion matrix
                cm_filename = os.path.join(results_dir, f'confusion_matrix_epoch_{current_epoch + 1}.png')
                plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
                print(f"💾 Confusion matrix saved: {cm_filename}")
                
            except Exception as e:
                print(f"⚠ Could not save confusion matrix to file: {e}")
            
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠ Error generating confusion matrix: {e}")

    # --------------- loaders --------------- #
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        
        # Add learning rate scheduler for better training
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            min_lr=1e-7
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_epoch"
            }
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset.split("train"),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset.split("val"),
            batch_size=self.hparams.batch_size,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True
        )


# --------------------------------------------------------------------- #
# 4. Utilitário simples
# --------------------------------------------------------------------- #
class AvgMeter:
    """Média móvel exponencial para acompanhamento de métricas."""

    def __init__(self, momentum: float = 0.95):
        self.momentum = momentum
        self.val = 0.0
        self.first = True

    def update(self, new_val: float):
        if self.first:
            self.val = new_val
            self.first = False
        else:
            self.val = self.val * self.momentum + new_val * (1 - self.momentum)
