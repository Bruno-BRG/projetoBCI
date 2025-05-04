# First import asyncio and set policy before any other imports
import asyncio
import platform
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Import required modules before session state initialization
import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Required for streamlit
import matplotlib.pyplot as plt
plt.style.use('dark_background')  # Use dark theme for plots

from collections import deque
import queue
import time
import threading
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from pylsl import StreamInlet, resolve_streams
from ML1 import create_bci_system, load_local_eeg_data

# Configure device and visualization settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

# EEG visualization settings
CHANNEL_COLORS = [
    '#4B0082',  # Purple
    '#0000FF',  # Blue
    '#00FF00',  # Green
    '#FFFF00',  # Yellow
    '#FFA500',  # Orange
    '#FF0000',  # Red
    '#FF69B4',  # Pink
    '#808080',  # Gray
    '#800080',  # Deep Purple
    '#000080',  # Navy Blue
    '#008000',  # Dark Green
    '#FFD700',  # Gold
    '#FF4500',  # Orange Red
    '#DC143C',  # Crimson
    '#FF1493',  # Deep Pink
    '#A9A9A9',  # Dark Gray
]

# Initialize all session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    # System state
    st.session_state.bci_system = create_bci_system()
    st.session_state.mode = 'calibration'
    st.session_state.calibration_count = 0
    
    # Data state
    st.session_state.current_data = None
    st.session_state.current_labels = None
    st.session_state.current_sample_idx = 0
    st.session_state.eeg_channel = None
    
    # Streaming state
    st.session_state.lsl_stream = None
    st.session_state.streaming = False
    st.session_state.stop_streaming = False
    st.session_state.eeg_buffer = deque(maxlen=2000)  # Increased buffer size
    st.session_state.prediction_buffer = deque(maxlen=100)
    st.session_state.stream_data_queue = queue.Queue()
    st.session_state.auto_scale = True
    st.session_state.scale_factor = 1.0
    st.session_state.display_channels = list(range(16))  # Show all channels by default

# Rest of the application code
st.title("Sistema BCI para Reabilitação Pós-AVC")

# Seleção de modo
st.sidebar.header("Configuração")
mode = st.sidebar.radio("Modo de Operação", ['Calibração', 'Uso Real', 'Streaming OpenBCI'])
st.session_state.mode = 'streaming' if mode == 'Streaming OpenBCI' else ('calibration' if mode == 'Calibração' else 'real_use')

def init_lsl_stream():
    """Initialize LSL stream connection"""
    try:
        print("Looking for an EEG stream...")
        # Get all available streams first
        streams = resolve_streams()
        
        st.markdown("### Available LSL Streams:")
        if not streams:
            st.error("No LSL streams found at all! Is your OpenBCI GUI running?")
            return None
            
        # Show all available streams
        for idx, stream in enumerate(streams):
            st.markdown(f"{idx + 1}. Name: {stream.name()}, Type: {stream.type()}, Channel Count: {stream.channel_count()}")
        
        # Filter for EEG streams
        eeg_streams = [stream for stream in streams if stream.type() == 'EEG']
        
        if not eeg_streams:
            st.error("No EEG streams found! Make sure OpenBCI GUI is streaming via LSL and the stream type is set to 'EEG'")
            return None
            
        # Connect to the first available EEG stream
        selected_stream = eeg_streams[0]
        inlet = StreamInlet(selected_stream)
        
        # Show detailed stream information
        st.success(f"""Connected to LSL stream:
        - Name: {selected_stream.name()}
        - Type: {selected_stream.type()}
        - Channel count: {selected_stream.channel_count()}
        - Sampling rate: {selected_stream.nominal_srate()} Hz
        - Source ID: {selected_stream.source_id()}
        """)
        
        return inlet
    except Exception as e:
        st.error(f"Error connecting to LSL stream: {str(e)}")
        st.exception(e)  # Show full error traceback
        return None

def process_eeg_chunk(chunk, timestamps):
    """Process incoming EEG data chunk"""
    if st.session_state.bci_system and st.session_state.bci_system.is_calibrated:
        # Prepare data for model
        data = np.array(chunk).T  # Transpose to match expected shape
        prediction, confidence = st.session_state.bci_system.predict_movement(data)
        st.session_state.prediction_buffer.append((prediction, confidence))
    
    # Store raw data
    for i, sample in enumerate(chunk):
        st.session_state.eeg_buffer.append((timestamps[i], sample))

def streaming_worker():
    """Background worker for LSL streaming"""
    while st.session_state.streaming:
        if st.session_state.lsl_stream:
            chunk, timestamps = st.session_state.lsl_stream.pull_chunk()
            if chunk:
                process_eeg_chunk(chunk, timestamps)
        time.sleep(0.1)  # Small delay to prevent CPU overload

def safe_streaming_worker():
    """Thread-safe background worker for LSL streaming"""
    try:
        print("Starting streaming worker...")
        while not st.session_state.stop_streaming:
            if st.session_state.lsl_stream:
                # Use a smaller timeout to be more responsive
                chunk, timestamps = st.session_state.lsl_stream.pull_chunk(timeout=0.1)
                if chunk:
                    # Log receiving data
                    print(f"Received chunk of size: {len(chunk)} samples x {len(chunk[0])} channels")
                    # Put data in queue instead of directly processing
                    st.session_state.stream_data_queue.put((chunk, timestamps))
                else:
                    print("No data received in this iteration")
            time.sleep(0.01)  # Smaller sleep time for better responsiveness
    except Exception as e:
        print(f"Streaming worker error: {str(e)}")
        st.error(f"Streaming worker error: {str(e)}")
    finally:
        print("Streaming worker stopped")

def create_eeg_figure(data, num_samples=500, auto_scale=True, scale_factor=1.0):
    """Create a matplotlib figure for EEG visualization"""
    plt.close('all')  # Close any existing plots
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
    ax.set_facecolor('black')
    
    # Calculate vertical spacing for channels
    num_channels = len(st.session_state.display_channels)
    channel_spacing = 2.0
    offsets = np.arange(num_channels) * channel_spacing
    
    # Auto-scaling calculation
    if auto_scale and len(data) > 0:
        max_amp = np.max(np.abs(data))
        if max_amp > 0:
            scale_factor = (channel_spacing * 0.4) / max_amp
    
    # Plot each channel
    for idx, ch in enumerate(st.session_state.display_channels):
        if ch < data.shape[1]:  # Make sure channel exists in data
            signal = data[-num_samples:, ch] * scale_factor
            ax.plot(signal + offsets[idx], 
                   color=CHANNEL_COLORS[ch], 
                   label=f'Ch{ch+1}',
                   linewidth=1,
                   alpha=0.8)
    
    # Customize plot appearance
    ax.set_ylim(-channel_spacing, num_channels * channel_spacing)
    ax.set_xlim(0, num_samples)
    ax.grid(True, color='#333333', linestyle='-', alpha=0.3)
    
    # Add channel labels
    for idx, ch in enumerate(st.session_state.display_channels):
        ax.text(-50, offsets[idx], f'Ch{ch+1}', 
                color=CHANNEL_COLORS[ch], 
                va='center',
                fontsize=8)
    
    # Customize ticks and labels
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333333')
    
    ax.set_title("Real-time EEG", color='white', pad=20, fontsize=12)
    ax.set_xlabel("Time (s)", color='white', fontsize=10)
    ax.set_ylabel("Channels", color='white', fontsize=10)
    
    # Add time markers
    time_points = np.linspace(0, num_samples, 6)
    time_labels = [f"-{i}s" for i in range(5, -1, -1)]
    ax.set_xticks(time_points)
    ax.set_xticklabels(time_labels, color='white')
    
    # Adjust layout
    plt.tight_layout()
    return fig

# Streaming Interface
if st.session_state.mode == 'streaming':
    st.header("OpenBCI Streaming Mode")
    
    # Add OpenBCI LSL Configuration Instructions
    with st.expander("OpenBCI LSL Configuration Instructions"):
        st.markdown("""
        ### How to configure OpenBCI GUI for LSL streaming:
        1. Open OpenBCI GUI
        2. Connect to your device
        3. Click on Networking Widget
        4. Select "LSL" from the dropdown
        5. Set Stream Type to "EEG"
        6. Click "Start" in the LSL widget
        7. Then click "Start Streaming" below
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.streaming:
            if st.button("Start Streaming"):
                st.session_state.lsl_stream = init_lsl_stream()
                if st.session_state.lsl_stream:
                    st.session_state.streaming = True
                    st.session_state.stop_streaming = False
                    # Clear buffers when starting new stream
                    st.session_state.eeg_buffer.clear()
                    st.session_state.prediction_buffer.clear()
                    st.session_state.stream_data_queue = queue.Queue()
                    threading.Thread(target=safe_streaming_worker, daemon=True).start()
        else:
            if st.button("Stop Streaming"):
                st.session_state.stop_streaming = True
                st.session_state.streaming = False
                if st.session_state.lsl_stream:
                    st.session_state.lsl_stream = None
                st.warning("Streaming stopped")
    
    with col2:
        update_interval = st.slider("Update Interval (ms)", 100, 1000, 500)
    
    # Add debug metrics
    debug_metrics = st.empty()
    
    # Create two columns for EEG and predictions
    col1, col2 = st.columns(2)
    
    # Real-time EEG plot placeholder
    with col1:
        st.subheader("Raw EEG Signal")
        eeg_plot = st.empty()
        
    # Real-time predictions placeholder
    with col2:
        st.subheader("Movement Predictions")
        prediction_plot = st.empty()
        confidence_plot = st.empty()
    
    # Process data from queue and update plots
    if st.session_state.streaming:
        placeholder = st.empty()
        with placeholder.container():
            try:
                # Update debug metrics more frequently
                with debug_metrics.container():
                    st.markdown("### Debug Information")
                    st.markdown(f"""
                    - Buffer size: {len(st.session_state.eeg_buffer)} samples
                    - Queue size: {st.session_state.stream_data_queue.qsize()} chunks
                    - Last update: {time.strftime('%H:%M:%S')}
                    - Streaming active: {st.session_state.streaming}
                    - Stop flag: {st.session_state.stop_streaming}
                    """)
                
                # Process all available data in queue
                data_received = False
                while not st.session_state.stream_data_queue.empty():
                    chunk, timestamps = st.session_state.stream_data_queue.get_nowait()
                    if chunk:  # Only process if we actually got data
                        process_eeg_chunk(chunk, timestamps)
                        data_received = True
                
                # Plot EEG data if we have any
                if len(st.session_state.eeg_buffer) > 0:
                    # Convert buffer to numpy array, handling tuple unpacking
                    buffer_data = list(st.session_state.eeg_buffer)
                    timestamps, samples = zip(*buffer_data)
                    data = np.array(samples)
                    
                    # Create and display the EEG plot
                    fig = create_eeg_figure(
                        data,
                        num_samples=500,
                        auto_scale=st.session_state.auto_scale,
                        scale_factor=st.session_state.scale_factor
                    )
                    eeg_plot.pyplot(fig)
                    plt.close(fig)  # Clean up
                
                # Update predictions if available
                if len(st.session_state.prediction_buffer) > 0:
                    predictions, confidences = zip(*list(st.session_state.prediction_buffer))
                    
                    # Show latest prediction
                    latest_pred = predictions[-1]
                    latest_conf = confidences[-1]
                    
                    # Format prediction display
                    prediction_color = "#00ff00" if latest_conf > 0.7 else "#ffff00"
                    prediction_plot.markdown(
                        f"<div style='text-align: center; padding: 10px; background-color: {prediction_color}; border-radius: 5px;'>"
                        f"<h3>Current Prediction:</h3>"
                        f"<h2>{latest_pred}</h2>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Show confidence bar
                    confidence_plot.progress(latest_conf)
                    confidence_plot.markdown(f"Confidence: {latest_conf:.2%}")
                
                # Add debug information
                st.markdown("---")
                st.markdown("### Debug Information")
                st.markdown(f"Buffer size: {len(st.session_state.eeg_buffer)} samples")
                st.markdown(f"Last update: {time.strftime('%H:%M:%S')}")
                
                time.sleep(update_interval / 1000)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing stream data: {str(e)}")
                st.exception(e)  # This will show the full traceback
            
    # Visualization controls in sidebar for streaming mode
    if st.session_state.mode == 'streaming':
        with st.sidebar:
            st.markdown("### Visualization Settings")
            st.session_state.auto_scale = st.checkbox("Auto Scale", value=True)
            if not st.session_state.auto_scale:
                st.session_state.scale_factor = st.slider("Manual Scale", 0.1, 10.0, 1.0, 0.1)
            
            # Channel display settings
            st.markdown("### Channel Display")
            all_channels = st.checkbox("Show All Channels", value=True)
            if not all_channels:
                st.session_state.display_channels = st.multiselect(
                    "Select Channels",
                    options=list(range(16)),
                    default=list(range(8)),
                    format_func=lambda x: f"Channel {x+1}"
                )
            else:
                st.session_state.display_channels = list(range(16))
                
            # Update rate control
            st.markdown("### Display Settings")
            st.session_state.update_rate = st.slider(
                "Update Rate (Hz)", 
                min_value=1, 
                max_value=60, 
                value=20
            )
            st.session_state.time_window = st.slider(
                "Time Window (s)", 
                min_value=1, 
                max_value=10, 
                value=5
            )

# Original calibration and real use modes continue below
elif st.session_state.mode == 'calibration':
    st.header("Modo de Calibração")
    st.markdown("""
    Neste modo, você irá selecionar amostras dos dados gravados para treinar o sistema.
    1. Carregue os dados do paciente usando o botão no menu lateral
    2. Selecione as amostras para treinar o modelo
    3. Treine o modelo com as amostras selecionadas
    """)

    if st.session_state.current_data is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Amostra Anterior"):
                st.session_state.current_sample_idx = max(0, st.session_state.current_sample_idx - 1)
                
        with col2:
            st.write(f"Amostra atual: {st.session_state.current_sample_idx + 1}/{len(st.session_state.current_data)}")
            
        with col3:
            if st.button("Próxima Amostra"):
                st.session_state.current_sample_idx = min(
                    len(st.session_state.current_data) - 1,
                    st.session_state.current_sample_idx + 1
                )

        # Mostrar o sinal EEG atual
        fig, ax = plt.subplots(figsize=(12, 4))
        current_sample = st.session_state.current_data[st.session_state.current_sample_idx]
        current_label = st.session_state.current_labels[st.session_state.current_sample_idx]
        for channel in range(current_sample.shape[0]):
            ax.plot(current_sample[channel], alpha=0.5)
        ax.set_title(f"Sinal EEG - Movimento {'Esquerdo' if current_label == 0 else 'Direito'}")
        ax.set_xlabel("Tempo")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Usar para Calibração"):
                st.session_state.bci_system.add_calibration_sample(
                    current_sample,
                    current_label
                )
                st.session_state.calibration_count += 1
                st.success(f"Amostra adicionada! Total: {st.session_state.calibration_count}")

        # Inicializar/Treinar modelo
        if st.session_state.calibration_count >= min_samples:
            if st.button("Treinar Modelo"):
                try:
                    with st.spinner("Treinando o modelo..."):
                        if not st.session_state.bci_system.model:
                            st.session_state.bci_system.initialize_model(st.session_state.eeg_channel)
                        
                        st.session_state.bci_system.train_calibration(
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate
                        )
                    st.success("Modelo treinado com sucesso!")
                except Exception as e:
                    st.error(f"Erro durante o treinamento: {str(e)}")
        else:
            st.warning(f"Necessário mais {min_samples - st.session_state.calibration_count} amostras para treinar")

else:
    st.header("Modo de Uso Real")
    if not st.session_state.bci_system.is_calibrated:
        st.error("Sistema precisa ser calibrado primeiro! Mude para o modo de calibração.")
    else:
        st.markdown("""
        Neste modo, o sistema irá classificar os movimentos imaginados dos dados gravados.
        1. Use os controles para navegar entre as amostras
        2. Clique em 'Classificar Movimento' para ver a predição
        """)

        if st.session_state.current_data is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Amostra Anterior "):
                    st.session_state.current_sample_idx = max(0, st.session_state.current_sample_idx - 1)
                    
            with col2:
                st.write(f"Amostra atual: {st.session_state.current_sample_idx + 1}/{len(st.session_state.current_data)}")
                
            with col3:
                if st.button("Próxima Amostra "):
                    st.session_state.current_sample_idx = min(
                        len(st.session_state.current_data) - 1,
                        st.session_state.current_sample_idx + 1
                    )

            if st.button("Classificar Movimento"):
                try:
                    current_sample = st.session_state.current_data[st.session_state.current_sample_idx]
                    current_label = st.session_state.current_labels[st.session_state.current_sample_idx]
                    
                    prediction, confidence = st.session_state.bci_system.predict_movement(current_sample)
                    
                    # Exibir resultado
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Movimento Real")
                        st.markdown(f"**{'Esquerdo' if current_label == 0 else 'Direito'}**")
                    with col2:
                        st.markdown("### Movimento Detectado")
                        st.markdown(f"**{prediction}**")
                    
                    # Barra de confiança
                    st.markdown("### Confiança da Predição")
                    st.progress(confidence)
                    st.write(f"Confiança: {confidence:.2%}")
                    
                    # Plotar o sinal EEG
                    fig, ax = plt.subplots(figsize=(12, 4))
                    for channel in range(current_sample.shape[0]):
                        ax.plot(current_sample[channel], alpha=0.5)
                    ax.set_title("Sinal EEG Atual")
                    ax.set_xlabel("Tempo")
                    ax.set_ylabel("Amplitude")
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Erro durante a classificação: {str(e)}")
        else:
            st.warning("Carregue os dados do paciente primeiro usando o menu lateral")

# Mostrar métricas e estatísticas
if st.session_state.bci_system.is_calibrated:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Status do Sistema")
    st.sidebar.success("✓ Sistema Calibrado")
    if os.path.exists(st.session_state.bci_system.model_path):
        st.sidebar.success("✓ Modelo Salvo")