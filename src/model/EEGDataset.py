import torch
import torch.utils.data as data

class EEGDataset(data.Dataset):
    def __init__(self, x, y=None, inference=False):
        super().__init__()
        self.__split = None

        N_SAMPLE = x.shape[0]
        val_idx = int(0.9 * N_SAMPLE)    # Validation split index
        train_idx = int(0.81 * N_SAMPLE) # Training split index

        if not inference:
            # Training dataset
            self.train_ds = {
                'x': x[:train_idx],
                'y': y[:train_idx],
            }
            # Validation dataset
            self.val_ds = {
                'x': x[train_idx:val_idx],
                'y': y[train_idx:val_idx],
            }
            # Test dataset
            self.test_ds = {
                'x': x[val_idx:],
                'y': y[val_idx:],
            }
        else:
            self.__split = "inference"
            self.inference_ds = {
                'x': [x],
            }

    def __len__(self):
        return len(self.dataset['x'])

    def __getitem__(self, idx):
        x = self.dataset['x'][idx]
        if self.__split != "inference":
            y = self.dataset['y'][idx]
            x = torch.tensor(x).float()
            y = torch.tensor(y).unsqueeze(-1).float()
            return x, y
        else:
            x = torch.tensor(x).float()
            return x

    def split(self, __split):
        self.__split = __split
        return self

    @classmethod
    def inference_dataset(cls, x):
        return cls(x, inference=True)

    @property
    def dataset(self):
        assert self.__split is not None, "Please specify the split of dataset!"

        if self.__split == "train":
            return self.train_ds
        elif self.__split == "val":
            return self.val_ds
        elif self.__split == "test":
            return self.test_ds
        elif self.__split == "inference":
            return self.inference_ds
        else:
            raise TypeError("Unknown type of split!")
