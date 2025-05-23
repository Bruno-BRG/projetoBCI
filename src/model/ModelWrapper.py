import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as L
import matplotlib.pyplot as plt
import torchmetrics



try:
    from .AvgMeter import AvgMeter
except ImportError:
    from AvgMeter import AvgMeter

class ModelWrapper(L.LightningModule):
    def __init__(self, arch, dataset, batch_size, lr, max_epoch, test_dataset=None, cpu_optimize=True, fast_mode=False):
        """
        Lightning Module wrapper for EEG Classification Model
        
        Args:
            arch: The model architecture
            dataset: The main dataset for training and validation
            batch_size: Batch size for training
            lr: Learning rate
            max_epoch: Maximum number of epochs
            test_dataset: Optional separate test dataset
            cpu_optimize: Whether to apply CPU-specific optimizations
            fast_mode: Enable fast mode with less validation and simpler processing
        """
        super().__init__()
        self.save_hyperparameters(ignore=['arch', 'dataset', 'test_dataset'])
        
        self.arch = arch
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.fast_mode = fast_mode
        
        # Apply CPU optimizations if requested
        if cpu_optimize and not torch.cuda.is_available():
            # Set number of threads for better CPU performance
            torch.set_num_threads(max(4, torch.get_num_threads()))
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(max(4, torch.get_num_interop_threads()))
            print(f"CPU optimization: Using {torch.get_num_threads()} computation threads")

        # Binary classification metrics for each phase
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.test_accuracy = torchmetrics.Accuracy(task="binary")

        # Automatic optimization disabled for custom training loop
        self.automatic_optimization = False

        # History tracking
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        # Moving average meters for smoother metrics
        self.train_loss_recorder = AvgMeter()
        self.val_loss_recorder = AvgMeter()
        self.train_acc_recorder = AvgMeter()
        self.val_acc_recorder = AvgMeter()
        
        # Validation frequency (validate less often in fast mode)
        self.val_check_interval = 5 if fast_mode else 1

    def forward(self, x):
        return self.arch(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.train_accuracy.update(y_hat, y)
        acc = self.train_accuracy.compute().detach().cpu()

        # Manual optimization
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # Update metrics
        self.train_loss_recorder.update(loss.detach())
        self.train_acc_recorder.update(acc)        # Log metrics - only essential metrics in progress bar
        # Log detailed metrics to TensorBoard
        self.log_dict({
            "train/loss": loss,
            "train/accuracy": acc,
        }, prog_bar=False, logger=True)
        
        # Log minimal metrics to progress bar
        self.log_dict({
            "loss": loss,
            "acc": acc,
        }, prog_bar=True, logger=False)

        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.val_accuracy.update(y_hat, y)
        acc = self.val_accuracy.compute().detach().cpu()

        # Update metrics
        self.val_loss_recorder.update(loss.detach())
        self.val_acc_recorder.update(acc)        # Log metrics
        metrics = {
            "val_loss": loss,
            "val_acc": acc,
            "val/loss": loss,  # For TensorBoard
            "val/accuracy": acc,
        }
        self.log_dict(metrics, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        # Step the learning rate scheduler
        sch = self.lr_schedulers()
        sch.step()

        # Record metrics for this epoch
        self.train_loss.append(self.train_loss_recorder.show().detach().cpu().numpy())
        self.train_acc.append(self.train_acc_recorder.show().detach().cpu().numpy())

        # Reset meters for next epoch
        self.train_loss_recorder = AvgMeter()
        self.train_acc_recorder = AvgMeter()

    def on_validation_epoch_end(self):
        # Record metrics for this epoch
        self.val_loss.append(self.val_loss_recorder.show().detach().cpu().numpy())
        self.val_acc.append(self.val_acc_recorder.show().detach().cpu().numpy())

        # Reset meters for next epoch
        self.val_loss_recorder = AvgMeter()
        self.val_acc_recorder = AvgMeter()

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.test_accuracy.update(y_hat, y)

        # Log metrics
        metrics = {
            "test_loss": loss,
            "test_acc": self.test_accuracy.compute(),
            "test/loss": loss,
            "test/accuracy": self.test_accuracy.compute(),
        }
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01  # L2 regularization
        )
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(self.max_epoch * 0.25),
                    int(self.max_epoch * 0.5),
                    int(self.max_epoch * 0.75),
                ],
                gamma=0.1
            ),
            "name": "learning_rate",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def on_train_end(self):
        # Plot final training curves
        self._plot_metrics()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset.split("train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Best for CPU is often 0 workers
            pin_memory=False,  # Pin memory is more beneficial for GPU
            persistent_workers=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset.split("val"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Best for CPU is often 0 workers
            pin_memory=False,
        )
        
    def test_dataloader(self):
        if self.test_dataset is not None:
            # Use the separate test dataset if provided
            return torch.utils.data.DataLoader(
                dataset=self.test_dataset.split("test"),
                batch_size=self.batch_size,  # Use the same batch size for testing
                shuffle=False,
                num_workers=0,
            )
        else:
            # Fall back to the test split from the main dataset
            return torch.utils.data.DataLoader(
                dataset=self.dataset.split("test"),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

    def _plot_metrics(self):
        """Plot training metrics at the end of training."""
        # Loss curves
        plt.figure(figsize=(10, 4))
        plt.plot(self.train_loss, color='r', label='Training Loss')
        plt.plot(self.val_loss, color='b', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_plot.png')
        plt.close()

        # Accuracy curves
        plt.figure(figsize=(10, 4))
        plt.plot(self.train_acc, color='r', label='Training Accuracy')
        plt.plot(self.val_acc, color='b', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig('accuracy_plot.png')
        plt.close()
