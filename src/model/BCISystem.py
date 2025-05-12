# Importações da biblioteca padrão
import os
import logging
from datetime import datetime

# Importações de terceiros
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Importações locais
from .EEGClassificationModel import EEGClassificationModel
from .LightningEEGModel import LightningEEGModel

def get_device():
    """Função auxiliar para obter o melhor dispositivo disponível"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class BCISystem:
    def __init__(self, model_path=None):
        self.device = get_device()  # Usa dispositivo global
        self.model = None
        self.eeg_channel = None
        self.calibration_data = {'X': [], 'y': []}
        self.is_calibrated = False
        self.model_path = model_path

    def _get_best_device(self):
        return get_device()  # Usa função global de dispositivo

    def initialize_model(self, eeg_channel):
        # Aceita tanto a contagem de canais quanto a lista de nomes de canais
        if isinstance(eeg_channel, (list, tuple)):
            ch_count = len(eeg_channel)
        else:
            ch_count = eeg_channel
        self.eeg_channel = ch_count
        
        # Cria o modelo base
        base_model = EEGClassificationModel(eeg_channel=ch_count, dropout=0.125)
        
        # Cria uma entrada fictícia para inicializar o classificador
        # Isso garante que o classificador exista antes de carregar o state dict
        with torch.no_grad():
            dummy_input = torch.zeros((1, ch_count, 125), dtype=torch.float32)
            _ = base_model(dummy_input)  # Isso inicializará o classificador
        
        # Envolve com o módulo Lightning
        self.model = LightningEEGModel(base_model, learning_rate=5e-4)
        
        # Garante consistência de dtype em todo o modelo - usa float32 para melhor compatibilidade
        self.model = self.model.float()
        self.model = self.model.to(self.device)
        
        # Tenta carregar o checkpoint: primeiro tenta o estado bruto do modelo no base_model
        if self.model_path and os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location=self.device)
                # Tenta carregar no modelo base
                try:
                    base_model.load_state_dict(state)
                    print(f"Carregado estado bruto do modelo no modelo base de {self.model_path}")
                    self.is_calibrated = True
                except Exception:
                    # Se o estado bruto falhar, tenta o checkpoint do Lightning
                    try:
                        # Envolve e carrega o checkpoint do Lightning
                        lightning_ckpt = state
                        self.model.load_state_dict(lightning_ckpt, strict=False)
                        self.is_calibrated = True
                        print(f"Carregado checkpoint do Lightning de {self.model_path}")
                    except Exception as e:
                        print(f"Não foi possível carregar o checkpoint: {str(e)}")
                        print("Iniciando com um modelo novo - requer calibração")
                        self.is_calibrated = False
            except Exception as e:
                print(f"Erro ao carregar o arquivo de checkpoint: {str(e)}")
                self.is_calibrated = False

    def add_calibration_sample(self, eeg_data, label):
        """Adiciona uma amostra de calibração"""
        self.calibration_data['X'].append(eeg_data)
        self.calibration_data['y'].append(label)

    def train_calibration(self, num_epochs=100, batch_size=10, learning_rate=5e-4):
        """Treina o modelo com os dados de calibração usando Lightning"""
        if len(self.calibration_data['X']) < 2:
            raise ValueError("É necessário pelo menos 2 amostras de calibração")

        X = np.array(self.calibration_data['X'])
        y = np.array(self.calibration_data['y'])
        
        # Cria datasets
        train_size = int(0.8 * len(X))
        indices = torch.randperm(len(X))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Usa tensores float32 para corresponder ao dtype do modelo
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                     torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                   torch.tensor(y_val, dtype=torch.float32))
        
        # Cria data loaders com workers para melhor desempenho
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

        # Configura o treinamento
        logger = TensorBoardLogger("lightning_logs", name="bci_model")
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            verbose=True,
            mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best_model",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )

        # Inicializa o treinador
        trainer = Trainer(
            max_epochs=num_epochs,
            accelerator="auto",
            devices=1,
            logger=logger,
            callbacks=[early_stopping, checkpoint_callback],
            log_every_n_steps=1
        )

        # Armazena a classe e os parâmetros iniciais do modelo para recarregar
        model_class = self.model.__class__
        model_params = {
            "model": self.model.model,
            "learning_rate": learning_rate
        }

        # Treina o modelo
        trainer.fit(
            self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

        # Carrega o melhor modelo - corrigido para usar o método de classe na classe, não na instância
        self.model = model_class.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            **model_params
        )
        
        if self.model_path:
            torch.save(self.model.state_dict(), self.model_path)
        
        self.is_calibrated = True

    def predict_movement(self, eeg_data):
        """Prevê movimento imaginado a partir dos dados de EEG com pontuação de confiança balanceada"""
        if not self.is_calibrated:
            raise ValueError("O sistema precisa ser calibrado primeiro")

        self.model.eval()
        with torch.no_grad():
            # Converte a entrada para o mesmo dtype do modelo
            input_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Certifica-se de que temos a dimensão de tempo esperada
            if input_tensor.shape[2] != 125:
                if input_tensor.shape[2] > 125:
                    input_tensor = input_tensor[:, :, :125]  # Trunca
                else:
                    pad_size = 125 - input_tensor.shape[2]
                    input_tensor = F.pad(input_tensor, (0, pad_size), "constant", 0)  # Preenche
            
            output = self.model(input_tensor)
            
            # Obtém o logit bruto e converte para probabilidade com sigmoid
            logit = output.item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
            
            # Calcula as confianças para Esquerda e Direita
            left_confidence = 1 - prob  # Probabilidade de Esquerda
            right_confidence = prob     # Probabilidade de Direita
            
            # Saída de depuração para verificar a distribuição da previsão
            print(f"DEBUG: Probabilidade bruta: {prob:.4f}, Esquerda: {left_confidence:.4f}, Direita: {right_confidence:.4f}")
            
            # Define os limites de decisão
            LEFT_THRESHOLD = 0.40       # Se prob < 0.40, prevê Esquerda
            RIGHT_THRESHOLD = 0.60      # Se prob > 0.60, prevê Direita
            
            # Faz a previsão com base na probabilidade bruta
            if prob < LEFT_THRESHOLD:
                # Previsão de Esquerda - usa diretamente a confiança de esquerda do modelo
                # Converte para porcentagem para exibição
                return "Left", left_confidence
            elif prob > RIGHT_THRESHOLD:
                # Previsão de Direita - usa diretamente a confiança de direita do modelo
                # Converte para porcentagem para exibição
                return "Right", right_confidence
            else:
                # Previsão incerta - calcula a confiança de incerteza
                # Maior quando mais próximo de 0.5 (máxima incerteza)
                distance_from_center = abs(prob - 0.5)
                uncertain_confidence = 1.0 - (distance_from_center * 2)  # Escala de 0-1
                return "Uncertain", uncertain_confidence

def create_bci_system(model_path="checkpoints/bci_model.pth"):
    """Cria uma nova instância do sistema BCI"""
    return BCISystem(model_path=model_path)
