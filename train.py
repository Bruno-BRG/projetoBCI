import os
import argparse
from src.data.data_loader import load_and_process_data
from src.models.eeg_model import EEGClassificationModel
from src.training.trainer import BCITrainer

def main(args):
    # Create checkpoint directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load and preprocess data
    X, y, n_time = load_and_process_data(
        subject_id=args.subject_id,
        augment=args.augment
    )
    
    # Initialize model
    model = EEGClassificationModel(
        eeg_channel=X.shape[1],  # Number of EEG channels
        dropout=args.dropout
    )
    
    # Initialize trainer
    trainer = BCITrainer(
        model=model,
        device=args.device,
        log_dir=args.log_dir
    )
    
    # Train model
    best_val_loss = trainer.train(
        X=X,
        y=y,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        learning_rate=args.learning_rate
    )
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Save model
    trainer.save_model(os.path.join('checkpoints', 'bci_model.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BCI Model')
    parser.add_argument('--subject-id', type=int, default=1, help='Subject ID to use for training')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--log-dir', type=str, default='training_runs', help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    main(args)