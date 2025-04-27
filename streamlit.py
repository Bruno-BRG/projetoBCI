import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Import asyncio first and set policy
import asyncio
import platform
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ML1 import load_and_process_data, load_model, EEGDataset, ModelTracker, evaluate_model
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Force torch to use CPU for stability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

st.title("BCI EEG Classification Dashboard")

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'eeg_channel' not in st.session_state:
    st.session_state.eeg_channel = None
if 'tracker' not in st.session_state:
    st.session_state.tracker = ModelTracker(log_dir="streamlit_runs")

# Sidebar for configuration
st.sidebar.header("Configuration")
st.sidebar.markdown("---")

# Training parameters in sidebar
learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-2, 1e-3, format="%.5f")
num_epochs = st.sidebar.slider("Number of Epochs", 1, 50, 10)
batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)

# Data loading section
st.header("Data Management")

# Add patient selection
subject_id = st.number_input("Select patient ID (1-109)", min_value=1, max_value=109, value=1)

# Add data augmentation toggle
use_augmentation = st.checkbox("Use data augmentation", value=True)

@st.cache_data
def load_eeg_data():
    return load_and_process_data(subject_id=subject_id, augment=use_augmentation)

if st.button("Load EEG Data"):
    try:
        with st.spinner("Loading and processing EEG data..."):
            X, y, eeg_channel = load_eeg_data()
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.eeg_channel = eeg_channel
            st.session_state.data_loaded = True
            st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

# Display data info if loaded
if st.session_state.data_loaded:
    st.subheader("Dataset Information")
    st.write(f"Patient ID: {subject_id}")
    st.write(f"Number of samples: {st.session_state.X.shape[0]}")
    st.write(f"Number of EEG channels: {st.session_state.eeg_channel}")
    st.write(f"Time points per sample: {st.session_state.X.shape[2]}")
    if use_augmentation:
        st.write("Data augmentation: Enabled")
    else:
        st.write("Data augmentation: Disabled")

    # Plot sample EEG data
    if st.button("View Sample EEG Signal"):
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            sample_idx = np.random.randint(0, st.session_state.X.shape[0])
            channel_idx = np.random.randint(0, st.session_state.X.shape[1])
            ax.plot(st.session_state.X[sample_idx, channel_idx])
            ax.set_title(f"Sample EEG Signal (Sample {sample_idx}, Channel {channel_idx})")
            ax.set_xlabel("Time Points")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error plotting data: {str(e)}")

# Model section
st.header("Model Management")
if st.session_state.data_loaded:
    # Model initialization and training
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Initialize Model"):
            try:
                with st.spinner("Initializing model..."):
                    st.session_state.model = load_model(st.session_state.eeg_channel)
                    st.session_state.model.to(device)
                    st.success("Model initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing model: {str(e)}")

    with col2:
        if st.session_state.model is not None and st.button("Train Model"):
            try:
                # Prepare data and ensure model is in double precision
                X_tensor = torch.DoubleTensor(st.session_state.X)
                y_tensor = torch.DoubleTensor(st.session_state.y)
                
                # Split data into train and validation sets (90-10 split)
                train_size = int(0.9 * len(X_tensor))
                indices = torch.randperm(len(X_tensor))
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                # Create train and validation datasets
                train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
                val_dataset = TensorDataset(X_tensor[val_indices], y_tensor[val_indices])
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                # Ensure model is in double precision
                st.session_state.model = st.session_state.model.double()

                # Training setup
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(st.session_state.model.parameters(), lr=learning_rate)

                # Training loop with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                for epoch in range(num_epochs):
                    # Training phase
                    st.session_state.model.train()
                    train_running_loss = 0.0
                    train_all_preds = []
                    train_all_labels = []

                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        optimizer.zero_grad()
                        outputs = st.session_state.model(inputs)
                        loss = criterion(outputs.squeeze(), labels)
                        loss.backward()
                        optimizer.step()

                        train_running_loss += loss.item()
                        train_all_preds.extend(outputs.detach())
                        train_all_labels.extend(labels.detach())

                    # Validation phase
                    st.session_state.model.eval()
                    val_running_loss = 0.0
                    val_all_preds = []
                    val_all_labels = []

                    with torch.no_grad():
                        for inputs, labels in val_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = st.session_state.model(inputs)
                            loss = criterion(outputs.squeeze(), labels)
                            
                            val_running_loss += loss.item()
                            val_all_preds.extend(outputs.detach())
                            val_all_labels.extend(labels.detach())

                    # Calculate epoch metrics
                    train_epoch_loss = train_running_loss / len(train_loader)
                    val_epoch_loss = val_running_loss / len(val_loader)
                    
                    # Log metrics for both training and validation
                    train_preds = torch.stack(train_all_preds)
                    train_labels = torch.stack(train_all_labels)
                    val_preds = torch.stack(val_all_preds)
                    val_labels = torch.stack(val_all_labels)
                    
                    st.session_state.tracker.log_metrics('train', train_epoch_loss, train_preds, train_labels, epoch)
                    st.session_state.tracker.log_metrics('val', val_epoch_loss, val_preds, val_labels, epoch)

                    # Update progress
                    progress = (epoch + 1) / num_epochs
                    progress_bar.progress(progress)
                    status_text.text(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')

                st.success("Training completed!")

            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                raise e  # Re-raise to see full traceback

    # Visualization section
    if st.session_state.model is not None:
        st.header("Model Visualization")

        # Training history
        if len(st.session_state.tracker.train_losses) > 0:
            st.subheader("Training History")
            fig = st.session_state.tracker.plot_training_history()
            st.pyplot(fig)
            plt.close()

        # Model evaluation
        st.subheader("Model Evaluation")
        if st.button("Evaluate Model"):
            try:
                # Prepare test data and ensure model is in the correct precision
                X_test = torch.DoubleTensor(st.session_state.X[-100:])  # Changed to DoubleTensor
                y_test = torch.DoubleTensor(st.session_state.y[-100:])  # Changed to DoubleTensor
                test_dataset = TensorDataset(X_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=32)

                # Ensure model is in double precision
                st.session_state.model = st.session_state.model.double()
                
                # Get evaluation metrics
                eval_results = evaluate_model(st.session_state.model, test_loader, device)

                # Display results
                st.write(f"Test Accuracy: {eval_results['accuracy']:.4f}")
                st.text("Classification Report:")
                st.text(eval_results['classification_report'])

                # Plot confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(eval_results['confusion_matrix'], annot=True, fmt='d', ax=ax)
                plt.title("Confusion Matrix")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")

        # Sample predictions
        st.subheader("Make Predictions")
        sample_idx = st.number_input("Select sample index for prediction", 
                                   min_value=0, 
                                   max_value=st.session_state.X.shape[0]-1, 
                                   value=0)

        if st.button("Predict"):
            try:
                # Prepare input and ensure double precision
                sample = torch.DoubleTensor(st.session_state.X[sample_idx:sample_idx+1]).to(device)

                # Make prediction
                with torch.no_grad():
                    st.session_state.model.double()
                    st.session_state.model.eval()
                    prediction = st.session_state.model(sample)
                    
                    predicted_class = "Left" if prediction.item() < 0.5 else "Right"
                    actual_class = "Left" if st.session_state.y[sample_idx] == 0 else "Right"
                    
                    # Display prediction results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Actual Movement")
                        st.markdown(f"**{actual_class}**")
                    with col2:
                        st.markdown("### Predicted Movement")
                        st.markdown(f"**{predicted_class}**")
                    
                    # Add prediction confidence with proper clamping
                    raw_confidence = abs(prediction.item() - 0.5) * 2  # Scale to 0-2 range
                    confidence = min(1.0, raw_confidence)  # Clamp to max 1.0
                    st.markdown("### Prediction Confidence")
                    st.progress(confidence)
                    st.write(f"Raw confidence: {raw_confidence:.2%}")
                    st.write(f"Displayed confidence: {confidence:.2%}")
                    
                    # Indicate if prediction was correct
                    is_correct = predicted_class == actual_class
                    st.markdown("### Prediction Result")
                    if is_correct:
                        st.success("✓ Correct Prediction!")
                    else:
                        st.error("✗ Incorrect Prediction")

                    # Plot the EEG signals for this sample
                    fig, ax = plt.subplots(figsize=(12, 6))
                    for channel in range(st.session_state.X.shape[1]):
                        ax.plot(st.session_state.X[sample_idx, channel], alpha=0.5, 
                               label=f'Channel {channel}' if channel == 0 else None)
                    ax.set_title(f"EEG Signals for Sample {sample_idx}")
                    ax.set_xlabel("Time Points")
                    ax.set_ylabel("Amplitude")
                    if channel == 0:
                        ax.legend()
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
else:
    st.warning("Please load the data first before working with the model.")