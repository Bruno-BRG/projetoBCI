import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGClassificationModel(nn.Module):
    """
    Implementação otimizada da arquitetura EEGNet baseada em:
    "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
    """
    def __init__(self, eeg_channel, dropout=0.25, temporal_filters=8, depth_multiplier=2, input_length=125):
        super().__init__()
        
        # Parâmetros
        self.F1 = temporal_filters       # número de filtros temporais (8 no artigo)
        self.D = depth_multiplier        # multiplicador de profundidade (2 no artigo)
        self.F2 = self.D * self.F1       # número de filtros pontuais = F1 * D (16 no artigo)
        self.eeg_channel = eeg_channel   # Armazena contagem de canais para cálculos de forma
        self.input_length = input_length # Dimensão temporal
        
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
    
    def forward(self, x):
        # Input shape: [batch, channels, time]
        # Need to reshape to [batch, 1, channels, time] for 2D convolution
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # Add channel dimension -> [batch, 1, channels, time]
        
        # Forward through blocks
        x = self.block1(x)
        x = self.block2(x)
        
        # Flatten and classify
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x
    
    def l2_regularization(self):
        # Apply L2 regularization during training if needed
        l2_reg = torch.tensor(0., device=next(self.parameters()).device)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, 2)
        return l2_reg
