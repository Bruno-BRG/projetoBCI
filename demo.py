"""
Demo script for EEG Classification using EEGInceptionERP from braindecode
This script demonstrates the core functionality without the GUI
Following NASA coding standards with proper error handling and assertions
"""

import sys
import numpy as np
import torch
import torch.utils.data
from typing import Tuple, Optional
import traceback
from model import EEGModel, EEGData

def create_synthetic_eeg_data(n_samples: int = 100, n_channels: int = 64, n_timepoints: int = 497) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic EEG data for testing purposes
    
    Args:
        n_samples: Number of samples to generate
        n_channels: Number of EEG channels (default 64)
        n_timepoints: Number of time points per sample (default 497)
        
    Returns:
        Tuple of (X_data, y_labels) where X_data has shape (n_samples, n_channels, n_timepoints)
    """
    assert n_samples > 0, "Number of samples must be positive"
    assert n_channels > 0, "Number of channels must be positive"
    assert n_timepoints > 0, "Number of timepoints must be positive"
    
    # Generate realistic EEG-like data with some structure
    X_data = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    
    # Add some realistic amplitude scaling (EEG is typically in microvolts)
    X_data *= 50e-6  # Scale to ~50 microvolts
    
    # Generate binary labels
    y_labels = np.random.randint(0, 2, n_samples).astype(np.int64)
    
    print(f"   ✓ Generated synthetic EEG data: {X_data.shape}")
    print(f"   ✓ Data range: [{X_data.min():.2e}, {X_data.max():.2e}]")
    print(f"   ✓ Labels distribution: {np.bincount(y_labels)}")
    
    return X_data, y_labels


def test_data_processor() -> bool:
    """Test the EEG data processor component"""
    try:
        print("1. Testing Data Processor...")
        from data_processor import EEGDataProcessor
        
        processor = EEGDataProcessor()
        assert processor is not None, "Data processor should be initialized"
        print("   ✓ Data processor initialized successfully")
        return True
        
    except ImportError as e:
        print(f"   ✗ Failed to import EEGDataProcessor: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Error testing data processor: {e}")
        return False


def test_dataset(X_data: np.ndarray, y_labels: np.ndarray) -> Optional[object]:
    """Test the EEG dataset component"""
    try:
        print("\n2. Testing EEG Dataset...")
        from model import EEGData
        
        dataset = EEGData(X_data, y_labels)
        assert dataset is not None, "Dataset should be initialized"
        
        # Test train split
        train_dataset = dataset.split("train")
        train_length = len(train_dataset)
        assert train_length > 0, "Training dataset should not be empty"
        
        print(f"   ✓ Dataset created with {train_length} training samples")
        
        # Test data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=10, shuffle=True
        )
        assert train_loader is not None, "Data loader should be created"
        print("   ✓ Data loader created successfully")
        
        return dataset
        
    except ImportError as e:
        print(f"   ✗ Failed to import EEGData: {e}")
        return None
    except Exception as e:
        print(f"   ✗ Error testing dataset: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return None


def test_model(n_channels: int, n_timepoints: int) -> Optional[object]:
    """Test the EEG classification model"""
    try:
        print("\n3. Testing EEGInceptionERP Model...")
        from model import EEGModel

        model = EEGModel(
            eeg_channel=n_channels,
            dropout=0.1,
            n_times=n_timepoints
        )
        assert model is not None, "Model should be initialized"
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0, "Model should have parameters"
        assert trainable_params > 0, "Model should have trainable parameters"
        
        print(f"   ✓ EEGInceptionERP model created")
        print(f"   ✓ Total parameters: {total_params:,}")
        print(f"   ✓ Trainable parameters: {trainable_params:,}")
        
        return model
        
    except ImportError as e:
        print(f"   ✗ Failed to import EEGModel: {e}")
        return None
    except Exception as e:
        print(f"   ✗ Error testing model: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return None


def test_forward_pass(model: object, dataset: object) -> bool:
    """Test model forward pass with sample data"""
    try:
        print("\n4. Testing Forward Pass...")
        
        # Create data loader
        train_dataset = dataset.split("train")
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=5, shuffle=False
        )
        
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            # Get a sample batch
            sample_batch = next(iter(train_loader))
            x_batch, y_batch = sample_batch
            
            assert x_batch.shape[0] > 0, "Batch should not be empty"
            assert len(x_batch.shape) == 3, "Input should be 3D: (batch, channels, time)"
            
            # Forward pass
            output = model(x_batch)
            
            assert output is not None, "Model output should not be None"
            assert output.shape[0] == x_batch.shape[0], "Output batch size should match input"
            
            print(f"   ✓ Forward pass successful: {x_batch.shape} -> {output.shape}")
            
            # Test predictions
            prediction = torch.sigmoid(output)
            predicted_classes = (prediction > 0.5).float()
            
            assert prediction.min() >= 0.0, "Sigmoid output should be >= 0"
            assert prediction.max() <= 1.0, "Sigmoid output should be <= 1"
            
            print(f"   ✓ Predictions generated: {predicted_classes.squeeze().tolist()}")
            print(f"   ✓ Prediction probabilities range: [{prediction.min():.3f}, {prediction.max():.3f}]")
            
        return True
        
    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False


def main() -> None:
    """Main demo function following NASA coding standards"""
    print("EEG Classification Demo using EEGInceptionERP from braindecode")
    print("=" * 70)
    
    # Test parameters
    n_samples = 3555
    n_channels = 16
    n_timepoints = 497
    
    success_count = 0
    total_tests = 4
    
    try:
        # Test 1: Data processor
        if test_data_processor():
            success_count += 1
        
        # Generate synthetic data
        print(f"\nGenerating synthetic EEG data...")
        X_data, y_labels = create_synthetic_eeg_data(n_samples, n_channels, n_timepoints)
        
        # Test 2: Dataset
        dataset = test_dataset(X_data, y_labels)
        if dataset is not None:
            success_count += 1
        
        # Test 3: Model
        model = test_model(n_channels, n_timepoints)
        if model is not None:
            success_count += 1
        
        # Test 4: Forward pass (only if both model and dataset work)
        if model is not None and dataset is not None:
            if test_forward_pass(model, dataset):
                success_count += 1
        
    except Exception as e:
        print(f"\n✗ Unexpected error in main: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    
    # Final results
    print("\n" + "=" * 70)
    print(f"Demo Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✓ All tests passed! EEGInceptionERP integration successful.")
        print("✓ Model uses braindecode's EEGInceptionERP architecture")
        print("✓ Ready for training with your EEG data!")
        print("\nNext steps:")
        print("  - Run the full GUI application: python eeg_classifier_app.py")
        print("  - Or use the launcher: python launcher.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        print("✗ Ensure all dependencies are installed: pip install -r requirements.txt")
        
    print("=" * 70)


if __name__ == "__main__":
    main()
