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
from ML1 import load_and_process_data, load_model, EEGDataset
import torch

# Force torch to use CPU for stability
device = torch.device('cpu')
torch.set_num_threads(1)

st.title("EEG Classification Interface")

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

# Data loading section
st.header("Data Management")

@st.cache_data
def load_eeg_data():
    return load_and_process_data()

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
    st.write(f"Number of samples: {st.session_state.X.shape[0]}")
    st.write(f"Number of EEG channels: {st.session_state.eeg_channel}")
    st.write(f"Time points per sample: {st.session_state.X.shape[2]}")

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
    if st.button("Initialize Model"):
        try:
            with st.spinner("Initializing model..."):
                st.session_state.model = load_model(st.session_state.eeg_channel)
                st.session_state.model.to(device)
                st.success("Model initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")

    if st.session_state.model is not None:
        st.subheader("Make Predictions")
        sample_idx = st.number_input("Select sample index for prediction", 
                                   min_value=0, 
                                   max_value=st.session_state.X.shape[0]-1, 
                                   value=0)

        if st.button("Predict"):
            try:
                # Prepare input
                sample = torch.from_numpy(st.session_state.X[sample_idx:sample_idx+1]).to(device)

                # Make prediction
                with torch.no_grad():
                    st.session_state.model.eval()
                    prediction = st.session_state.model(sample)
                    predicted_class = "Left" if prediction.item() < 0.5 else "Right"
                    actual_class = "Left" if st.session_state.y[sample_idx] == 0 else "Right"

                    # Display results
                    st.write(f"Predicted movement: {predicted_class}")
                    st.write(f"Actual movement: {actual_class}")

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