import matplotlib.pyplot as plt
import numpy as np
from ML1 import load_and_process_data, EEGAugmentation
import math

def test_data_loading():
    # Try loading data for subject 1
    try:
        # Load raw data first (without augmentation)
        X_raw, y_raw, ch = load_and_process_data(1, augment=False)
        print(f"Raw samples: {len(X_raw)}")
        
        # Calculate grid dimensions - make it roughly square
        n_samples = len(X_raw)
        grid_size = math.ceil(math.sqrt(n_samples))
        n_rows = grid_size
        n_cols = grid_size
        
        # Create a large figure
        fig = plt.figure(figsize=(20, 20))
        fig.suptitle("All EEG Samples - Subject 1", fontsize=16)
        
        # Plot each sample in its own subplot
        for i in range(n_samples):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            sample = X_raw[i]
            # Plot all channels
            for ch_idx in range(sample.shape[0]):
                ax.plot(sample[ch_idx], alpha=0.5, linewidth=0.5)
            ax.set_title(f"Sample {i+1}\nLabel: {'Left' if y_raw[i]==0 else 'Right'}", fontsize=8)
            # Remove axis labels to save space
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig("all_samples_grid.png", dpi=300, bbox_inches='tight')
        print("Saved grid plot of all samples to all_samples_grid.png")
        
        # Now show augmentation effect on first sample
        augmented = EEGAugmentation.augment_data(X_raw[0:1])
        n_aug = len(augmented)
        
        # Create figure for augmented versions
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle("Original + Augmented Versions of First Sample", fontsize=16)
        
        # Plot original and each augmented version
        for i in range(n_aug):
            ax = fig.add_subplot(2, math.ceil(n_aug/2), i + 1)
            sample = augmented[i]
            for ch_idx in range(sample.shape[0]):
                ax.plot(sample[ch_idx], alpha=0.5, linewidth=0.5)
            ax.set_title(f"{'Original' if i==0 else f'Augmented {i}'}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig("augmented_versions.png", dpi=300, bbox_inches='tight')
        print("Saved augmented versions plot to augmented_versions.png")
        
        # Print statistics
        print(f"\nAugmentation results:")
        print(f"Original samples: {len(X_raw)}")
        X_aug, y_aug, _ = load_and_process_data(1, augment=True)
        print(f"After augmentation: {len(X_aug)}")
        print(f"Augmentation factor: {len(X_aug)/len(X_raw)}x")
        
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    test_data_loading()