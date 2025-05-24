
from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from braindecode.models import EEGInceptionERP


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
# 3. Lightning *wrapper*
# --------------------------------------------------------------------- #
class ModelWrapper(pl.LightningModule):
    """Orquestra treino/validação/teste e logging de métricas."""

    def __init__(self,
                 arch: nn.Module,
                 dataset: EEGData,
                 batch_size: int = 32,
                 lr: float = 5e-4,
                 max_epoch: int = 100):
        super().__init__()
        self.save_hyperparameters(ignore=["arch", "dataset"])
        self.arch = arch
        self.criterion = nn.BCEWithLogitsLoss()
        self.acc = Accuracy(task="binary")
        self.dataset = dataset

        # buffers para curvas
        self.train_loss, self.val_loss = [], []
        self.train_acc, self.val_acc = [], []

    # --------------- forward & step --------------- #
    def forward(self, x):
        return self.arch(x)

    def _common_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        acc = self.acc(torch.sigmoid(logits), y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        if stage == "train":
            self.train_loss.append(loss.item())
            self.train_acc.append(acc.item())
        elif stage == "val":
            self.val_loss.append(loss.item())
            self.val_acc.append(acc.item())
        return loss

    def training_step(self, batch, _):
        return self._common_step(batch, "train")

    def validation_step(self, batch, _):
        self._common_step(batch, "val")

    # --------------- loaders --------------- #
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset.split("train"),
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset.split("val"),
            batch_size=self.hparams.batch_size,
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
