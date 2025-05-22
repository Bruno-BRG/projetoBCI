import torch
import numpy as np

class AvgMeter(object):
    """
    Average meter for tracking losses or metrics during training.
    Maintains a moving average over a specified window.
    This implementation matches the notebook's AvgMeter.
    """
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.losses = []

    def update(self, val):
        self.losses.append(val)

    def show(self):
        out = torch.mean(
            torch.stack(
                self.losses[np.maximum(len(self.losses)-self.num, 0):]
            )
        )
        return out
