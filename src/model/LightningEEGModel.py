import torch
import pytorch_lightning as L
import torch.nn.functional as F
import torchmetrics

class LightningEEGModel(L.LightningModule):
    def __init__(self, model, learning_rate=5e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # Ensure input data has consistent time dimension
        if x.shape[2] != 125:  # If time dimension is not 125
            # Resize to expected input length by either truncating or padding
            if x.shape[2] > 125:
                x = x[:, :, :125]  # Truncate
            else:
                pad_size = 125 - x.shape[2]
                x = F.pad(x, (0, pad_size), "constant", 0)  # Pad with zeros
                
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.view(-1), y.float())
        acc = self.train_accuracy(y_hat.view(-1), y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Apply same input normalization as in training step
        if x.shape[2] != 125:  # If time dimension is not 125
            if x.shape[2] > 125:
                x = x[:, :, :125]  # Truncate
            else:
                pad_size = 125 - x.shape[2]
                x = F.pad(x, (0, pad_size), "constant", 0)  # Pad with zeros
                
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.view(-1), y.float())
        acc = self.val_accuracy(y_hat.view(-1), y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(self.trainer.max_epochs * 0.25),
                int(self.trainer.max_epochs * 0.5),
                int(self.trainer.max_epochs * 0.75)
            ],
            gamma=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

