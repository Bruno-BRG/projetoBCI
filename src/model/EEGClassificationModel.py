import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .PositionalEncoding import PositionalEncoding
except ImportError:
    from PositionalEncoding import PositionalEncoding
try:
    from .TransformerBlock import TransformerBlock
except ImportError:
    from TransformerBlock import TransformerBlock

class EEGClassificationModel(nn.Module):
    """
    Transformer-based EEG classification model with convolutional front-end.
    This implementation directly matches the structure from the EEG_mne_cnn.ipynb notebook.
    """
    def __init__(self, eeg_channel, dropout=0.1, fast_model=False):
        super().__init__()
        
        # Use a simpler, faster model architecture if requested
        self.fast_model = fast_model
        
        if fast_model:
            # Simpler convolutional network for faster training on CPU
            self.conv = nn.Sequential(
                nn.Conv1d(eeg_channel, eeg_channel, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm1d(eeg_channel),
                nn.ReLU(True),
                nn.Dropout1d(dropout),
                nn.Conv1d(eeg_channel, eeg_channel * 2, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm1d(eeg_channel * 2),
                nn.ReLU(True),
                nn.AvgPool1d(2),  # Downsample to reduce computation
            )
            
            # Simple classifier head - no transformer for faster processing
            self.classifier = nn.Sequential(
                nn.Flatten(),  # Flatten the feature maps
                nn.Linear(eeg_channel * 2 * (250 // 2), eeg_channel * 4),  # Adjust size based on your data
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(eeg_channel * 4, eeg_channel // 2),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(eeg_channel // 2, 1),
            )
        else:
            # Original convolutional front-end
            self.conv = nn.Sequential(
                nn.Conv1d(
                    eeg_channel, eeg_channel, 11, 1, padding=5, bias=False
                ),
                nn.BatchNorm1d(eeg_channel),
                nn.ReLU(True),
                nn.Dropout1d(dropout),
                nn.Conv1d(
                    eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False
                ),
                nn.BatchNorm1d(eeg_channel * 2),
            )
            
            # Transformer encoder
            self.transformer = nn.Sequential(
                PositionalEncoding(eeg_channel * 2, dropout),
                TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
                TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            )
            
            # Output MLP
            self.mlp = nn.Sequential(
                nn.Linear(eeg_channel * 2, eeg_channel // 2),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(eeg_channel // 2, 1),
            )
        
        # Log model initialization
        print(f"EEGClassificationModel initialized with channels={eeg_channel}, fast_model={fast_model}")
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, channels, time)
        
        Returns:
            Output tensor with predictions
        """
        if self.fast_model:
            # Fast model path for CPU
            x = self.conv(x)
            x = self.classifier(x)
            return x
            
        # Original model path
        # Pass through convolutional layers
        x = self.conv(x)
        
        # Reshape for transformer: (batch, time, channels) 
        x = x.permute(0, 2, 1)
        
        # Pass through transformer blocks
        x = self.transformer(x)
        
        # Reshape back for output: (batch, channels, time)
        x = x.permute(0, 2, 1)
        
        # Global average pooling over time dimension
        x = x.mean(dim=-1)
        
        # MLP classifier
        x = self.mlp(x)
        
        return x
        
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.view(-1), y.float())
        acc = self.train_accuracy(y_hat.view(-1), y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
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
        
    def predict_proba(self, x):
        """Get probability predictions with sigmoid activation"""
        self.eval()
        with torch.no_grad():
            output = self(x)
            return torch.sigmoid(output).item()
