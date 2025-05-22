# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np

# Local imports
from .EEGClassificationModel import EEGClassificationModel
from .EEGDataLoader import load_local_eeg_data
from .ModelTracker import get_device

class MultiSubjectTest:
    def __init__(self, train_samples=40, test_samples=20, model_path=None):
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.model_path = model_path  # path to save trained model
        self.device = get_device()  # Use global device
        self.model = None
        self.training_history = {
            'train_losses': [], 'val_losses': [],
            'train_accs': [], 'val_accs': []
        }
        self.input_length = 125  # Standard input time dimension
        print(f"\nInitializing MultiSubjectTest with {train_samples} training and {test_samples} test samples per subject")
        print(f"Using device: {self.device}")
    
    def collect_balanced_samples(self, subject_id, n_samples):
        """Collect balanced samples (equal left/right) from a subject"""
        try:
            X, y, ch = load_local_eeg_data(subject_id, augment=False)
            left_idx = (y == 0).nonzero()[0]
            right_idx = (y == 1).nonzero()[0]
            
            samples_per_class = n_samples // 2
            if len(left_idx) < samples_per_class or len(right_idx) < samples_per_class:
                print(f"Subject {subject_id:03d}: Insufficient samples (L:{len(left_idx)}, R:{len(right_idx)})")
                return None, None
                
            selected_left = np.random.choice(left_idx, samples_per_class, replace=False)
            selected_right = np.random.choice(right_idx, samples_per_class, replace=False)
            
            selected_idx = np.concatenate([selected_left, selected_right])
            print(f"Subject {subject_id:03d}: Successfully collected {n_samples} balanced samples")
            return X[selected_idx], y[selected_idx]
        except Exception as e:
            print(f"Subject {subject_id:03d}: Error loading data - {str(e)}")
            return None, None

    def prepare_datasets(self):
        """Prepare training and testing datasets from multiple subjects"""
        print("\nStarting dataset preparation...")
        train_X, train_y = [], []
        test_X, test_y = [], []
        
        successful_subjects = 0
        total_subjects = 0
        
        # Try subjects from 1 to 109 (maximum in dataset)
        for subject_id in range(1, 110):
            total_subjects += 1
            try:
                print(f"\nProcessing Subject {subject_id:03d}...")
                
                # Get training samples
                X_train, y_train = self.collect_balanced_samples(subject_id, self.train_samples)
                if X_train is not None:
                    train_X.append(X_train)
                    train_y.append(y_train)
                    
                    # Get testing samples (different from training)
                    X_test, y_test = self.collect_balanced_samples(subject_id, self.test_samples)
                    if X_test is not None:
                        test_X.append(X_test)
                        test_y.append(y_test)
                        successful_subjects += 1
                
            except Exception as e:
                print(f"Error processing subject {subject_id}: {str(e)}")
                continue
        
        if not train_X or not test_X:
            raise ValueError("Could not collect enough samples from any subject")
            
        final_train_X = np.concatenate(train_X)
        final_train_y = np.concatenate(train_y)
        final_test_X = np.concatenate(test_X)
        final_test_y = np.concatenate(test_y)
        
        print(f"\nDataset preparation completed:")
        print(f"Successfully processed {successful_subjects} out of {total_subjects} subjects")
        print(f"Total training samples: {len(final_train_X)}")
        print(f"Total testing samples: {len(final_test_X)}")
        print(f"Training data shape: {final_train_X.shape}")
        print(f"Testing data shape: {final_test_X.shape}")

        # Standardize time dimension to match expected input dimensions
        if final_train_X.shape[2] != self.input_length:
            print(f"Standardizing time dimension to {self.input_length}...")
            final_train_X = self._standardize_time_dimension(final_train_X)
            final_test_X = self._standardize_time_dimension(final_test_X)
            print(f"After standardization - Train: {final_train_X.shape}, Test: {final_test_X.shape}")
        
        return final_train_X, final_train_y, final_test_X, final_test_y

    def _standardize_time_dimension(self, data):
        """Standardize time dimension to self.input_length samples"""
        batch_size, channels, time_points = data.shape
        
        # If time dimension is longer, truncate
        if time_points > self.input_length:
            return data[:, :, :self.input_length]
        
        # If time dimension is shorter, pad with zeros
        elif time_points < self.input_length:
            padding = np.zeros((batch_size, channels, self.input_length - time_points))
            return np.concatenate([data, padding], axis=2)
        
        # If time dimension is already correct, return as is
        return data

    def train_and_evaluate(self, num_epochs=100, batch_size=10, learning_rate=5e-4):
        """Train on collected samples and evaluate performance"""
        try:
            print("\nStarting training and evaluation process...")
            print(f"Parameters: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
            
            # Prepare datasets
            print("\nPreparing datasets...")
            train_X, train_y, test_X, test_y = self.prepare_datasets()
            
            # Initialize model
            print("\nInitializing model...")
            self.model = EEGClassificationModel(eeg_channel=train_X.shape[1], input_length=self.input_length)
            self.model = self.model.float().to(self.device)  # Use float instead of double
            
            # Convert to tensors
            train_X = torch.tensor(train_X, dtype=torch.float32).to(self.device)
            train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
            test_X = torch.tensor(test_X, dtype=torch.float32).to(self.device)
            test_y = torch.tensor(test_y, dtype=torch.float32).to(self.device)
            
            # Create data loaders
            print("\nCreating data loaders...")
            train_dataset = TensorDataset(train_X, train_y)
            test_dataset = TensorDataset(test_X, test_y)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Training setup
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            print("\nStarting training loop...")
            
            # Training loop
            for epoch in range(num_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print("Training phase...")
                
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = self.model(inputs).squeeze()
                    loss = criterion(outputs, labels.float())
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    pred = (outputs > 0.5).float()
                    train_correct += (pred == labels).sum().item()
                    train_total += labels.size(0)
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Batch {batch_idx+1}/{len(train_loader)}")
                
                avg_train_loss = train_loss / len(train_loader)
                train_acc = train_correct / train_total
                
                # Validation phase
                print("Validation phase...")
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_idx, (inputs, labels) in enumerate(test_loader):
                        outputs = self.model(inputs).squeeze()
                        loss = criterion(outputs, labels.float())
                        val_loss += loss.item()
                        pred = (outputs > 0.5).float()
                        val_correct += (pred == labels).sum().item()
                        val_total += labels.size(0)
                
                avg_val_loss = val_loss / len(test_loader)
                val_acc = val_correct / val_total
                
                # Store metrics
                self.training_history['train_losses'].append(avg_train_loss)
                self.training_history['val_losses'].append(avg_val_loss)
                self.training_history['train_accs'].append(train_acc)
                self.training_history['val_accs'].append(val_acc)
                
                print(f"Epoch {epoch+1} Results:")
                print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%}")
                print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}")
            
            print("\nTraining completed successfully!")
            # Save model if a save path was provided
            if self.model_path:
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Saved multi-subject model to {self.model_path}")
            return self.training_history
         
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise
