"""
Demo script to test the EEG classification components
This script demonstrates the core functionality without the GUI
"""

import numpy as np
import torch
from eeg_model import EEGClassificationModel, EEGDataset
from data_processor import EEGDataProcessor

def test_components():
    """Test the individual components"""
    print("Testing EEG Classification Components...")
    print("=" * 50)
    
    # Test data processor
    print("1. Testing Data Processor...")
    processor = EEGDataProcessor()
    print("   ✓ Data processor initialized")
    
    # Create dummy data for testing (if real data not available)
    print("\n2. Creating test data...")
    n_samples, n_channels, n_timepoints = 100, 64, 497  # Typical EEG dimensions
    X_dummy = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    y_dummy = np.random.randint(0, 2, n_samples).astype(np.int64)
    print(f"   ✓ Dummy data created: {X_dummy.shape}")
    
    # Test dataset
    print("\n3. Testing EEG Dataset...")
    dataset = EEGDataset(X_dummy, y_dummy)
    train_loader = torch.utils.data.DataLoader(
        dataset.split("train"), batch_size=10, shuffle=True
    )
    print(f"   ✓ Dataset created with {len(dataset.split('train'))} training samples")
    
    # Test model
    print("\n4. Testing EEG Model...")
    model = EEGClassificationModel(eeg_channel=n_channels, dropout=0.1)
    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    print("\n5. Testing Forward Pass...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        x_batch, y_batch = sample_batch
        output = model(x_batch)
        print(f"   ✓ Forward pass successful: {x_batch.shape} -> {output.shape}")
        
        # Test prediction
        prediction = torch.sigmoid(output)
        predicted_classes = (prediction > 0.5).float()
        print(f"   ✓ Predictions: {predicted_classes.squeeze().tolist()}")
    
    print("\n" + "=" * 50)
    print("✅ All components working correctly!")
    print("\nTo run the full GUI application:")
    print("python eeg_classifier_app.py")
    print("\nOr use the launcher:")
    print("python launcher.py")

if __name__ == "__main__":
    test_components()
