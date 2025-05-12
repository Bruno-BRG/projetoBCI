# Importação do módulo de dados do PyTorch
import torch.utils.data as data

class EEGDataset(data.Dataset):
    def __init__(self, x, y=None, inference=False):
        super().__init__()

        N_SAMPLE = x.shape[0]
        val_idx = int(0.9 * N_SAMPLE)    # Índice para divisão do conjunto de validação
        train_idx = int(0.81 * N_SAMPLE) # Índice para divisão do conjunto de treinamento

        if not inference:
            # Conjunto de treinamento
            self.train_ds = {
                'x': x[:train_idx],
                'y': y[:train_idx],
            }
            # Conjunto de validação
            self.val_ds = {
                'x': x[train_idx:val_idx],
                'y': y[train_idx:val_idx],
            }
            # Conjunto de teste
            self.test_ds = {
                'x': x[val_idx:],
                'y': y[val_idx:],
            }
