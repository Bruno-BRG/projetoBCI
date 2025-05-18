import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics

class EEGClassificationModel(L.LightningModule):
    """
    Unified EEGNet model with integrated training capabilities.
    Based on "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
    """
    def __init__(self, eeg_channel, dropout=0.25, temporal_filters=8, depth_multiplier=2, 
                 input_length=125, learning_rate=5e-4):
        super().__init__()
        # Store hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Model architecture parameters
        self.F1 = temporal_filters
        self.D = depth_multiplier
        self.F2 = self.D * self.F1
        self.eeg_channel = eeg_channel
        self.input_length = input_length
        
        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        
        # Bloco 1: Filtragem Temporal + Restrição de Norma Máxima + Dropout
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),  # Convolução temporal
            nn.BatchNorm2d(self.F1),
            # Convolução depthwise com multiplicador de canal D
            nn.Conv2d(self.F1, self.D * self.F1, (eeg_channel, 1), groups=self.F1, bias=False),  
            nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(inplace=True),  # inplace=True economiza memória
            nn.AvgPool2d((1, 4)),  # Pooling médio para reduzir a dimensão temporal
            nn.Dropout(dropout)
        )
        
        # Bloco 2: Convolução separável + Pooling médio + Dropout
        self.block2 = nn.Sequential(
            # Convolução separável: Depthwise + Pointwise
            nn.Conv2d(self.D * self.F1, self.D * self.F1, (1, 16), padding=(0, 8), groups=self.D * self.F1, bias=False),  # Depthwise
            nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False),  # Pointwise
            nn.BatchNorm2d(self.F2),
            nn.ELU(inplace=True),  # inplace=True economiza memória
            nn.AvgPool2d((1, 8)),  # Pooling médio para redução adicional de dimensionalidade
            nn.Dropout(dropout)
        )
        
        # Log expected feature dimensions for debugging
        print(f"EEGNet initialized with: channels={eeg_channel}, F1={self.F1}, D={self.D}, F2={self.F2}")
        
        # Pre-calculate the feature size based on input dimensions
        # This is to ensure consistent classifier dimensions across model instances
        self._feature_size = self._calculate_feature_size()
        self.classifier = nn.Linear(self._feature_size, 1)
        
        # Apply max norm constraint to all convolutional layers
        self._apply_max_norm_constraint()
        
        # Initialize weights
        self._initialize_weights()
        
    def _calculate_feature_size(self):
        """Pre-calculate the feature size based on standard input dimensions"""
        # Create a dummy input with the standard shape
        dummy_input = torch.zeros(1, 1, self.eeg_channel, self.input_length)
        
        # Calculate the output size after passing through convolutional blocks
        with torch.no_grad():
            x = self.block1(dummy_input)
            x = self.block2(x)
            feature_size = x.view(1, -1).size(1)
            print(f"Calculated feature size: {feature_size}")
            return feature_size
        
    def _apply_max_norm_constraint(self, max_val=1.0):
        # Apply weight constraint to conv layers (L2 norm < 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = torch.renorm(
                    m.weight.data, p=2, dim=0, maxnorm=max_val
                )
    
    def _initialize_weights(self):
        """Initialize weights for faster convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def _normalize_input(self, x):
        """Ensure input has correct time dimension"""
        if x.shape[2] != self.input_length:
            if x.shape[2] > self.input_length:
                x = x[:, :, :self.input_length]
            else:
                pad_size = self.input_length - x.shape[2]
                x = F.pad(x, (0, pad_size), "constant", 0)
        return x
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self._normalize_input(x)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(batch_size, -1)
        return self.classifier(x)
    
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
