# Model Adaptation Guide

This guide explains how the models from the `EEG_mne_cnn.ipynb` notebook were adapted to work with your system.

## Components Implemented

1. **EEGDataset**: A PyTorch Dataset implementation for EEG data that handles train/val/test splits.
2. **PositionalEncoding**: A module for adding positional information to transformer inputs.
3. **TransformerBlock**: A standard transformer encoder block with multi-head attention.
4. **EEGClassificationModel**: The main model combining convolutional layers with transformer blocks.
5. **ModelWrapper**: A PyTorch Lightning wrapper for training the model.
6. **AvgMeter**: A utility for tracking and averaging metrics during training.

## Adaptation Notes

### Data Format

The notebook uses `.edf` files loaded with MNE, while your system uses `.csv` files loaded with pandas:

- **Notebook**: Uses `mne.io.read_raw_edf()` to load EEG data
- **Your System**: Uses pandas to load CSV data from `csv_openbci.csv` files

Your `load_and_process_data()` function in `EEGDataLoader.py` already handles loading from CSV files, so you can continue using it.

### Usage Example

Here's how to use the adapted models in your system:

```python
from src.model.EEGDataset import EEGDataset
from src.model.EEGClassificationModel import EEGClassificationModel
from src.model.ModelWrapper import ModelWrapper
from src.model.EEGDataLoader import load_and_process_data
import pytorch_lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Load the data (from your CSV files instead of edf)
X, y, eeg_channel = load_and_process_data(subject_id=1)

# Create the dataset
eeg_dataset = EEGDataset(x=X, y=y)

# Create the model with the correct number of channels
model = EEGClassificationModel(eeg_channel=eeg_channel, dropout=0.125)

# Hyperparameters
MAX_EPOCH = 100
BATCH_SIZE = 10
LR = 5e-4
MODEL_NAME = "EEGClassificationModel"

# Wrap the model for training
model_wrapper = ModelWrapper(
    arch=model, 
    dataset=eeg_dataset, 
    batch_size=BATCH_SIZE, 
    lr=LR, 
    max_epoch=MAX_EPOCH
)

# Callbacks
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint = ModelCheckpoint(
    monitor='val_acc',
    dirpath='checkpoints',
    filename=f'{MODEL_NAME}_best',
    mode='max',
)
early_stopping = EarlyStopping(
    monitor="val_acc", 
    min_delta=0.00, 
    patience=3, 
    verbose=False, 
    mode="max"
)

# Train the model
trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=MAX_EPOCH,
    callbacks=[lr_monitor, checkpoint, early_stopping],
    log_every_n_steps=5,
)
trainer.fit(model_wrapper)

# Test the model
trainer.test(model_wrapper)

# For inference
def predict_hand_movement(sample):
    """
    Predict hand movement from a single EEG sample
    
    Args:
        sample: EEG data with shape (channels, time)
        
    Returns:
        Prediction (0=left, 1=right)
    """
    # Create inference dataset
    inference_dataset = EEGDataset.inference_dataset(sample)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        shuffle=False,
    )
    
    # Predict
    trainer = L.Trainer()
    prediction = trainer.predict(
        model=model_wrapper,
        dataloaders=dataloader,
        ckpt_path='checkpoints/EEGClassificationModel_best.ckpt',
    )[0]
    
    # Convert to binary prediction
    pred_class = int(torch.sigmoid(prediction) > 0.5)
    classes = ["left", "right"]
    return classes[pred_class]
```

## Main Differences

1. **Model Architecture**: The model architecture is identical to the notebook, with the same convolutional layers, transformer blocks, and MLP head.

2. **Training Process**: The ModelWrapper now uses manual optimization as in the notebook, with the automatic_optimization flag set to False.

3. **Data Handling**: We've implemented the complete EEGDataset class from the notebook, which handles splitting data into train/val/test sets.

4. **Visualization**: The ModelWrapper includes functions to plot loss and accuracy curves at the end of training.
