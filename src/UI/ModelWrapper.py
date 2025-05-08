import torch
import torch.optim as optim
import pytorch_lightning as L
import torchmetrics

class ModelWrapper(L.LightningModule):
    # ...existing ModelWrapper code...

    def configure_optimizers(self):
        # Added weight_decay for L2 regularization
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01  # L2 regularization factor
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
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]