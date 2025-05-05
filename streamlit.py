# Handle OpenMP thread issues
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Import asyncio first and set policy
import asyncio
import platform
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import torch
import numpy as np
from ML1 import create_bci_system, load_local_eeg_data
from pylsl import StreamInlet, resolve_streams
import plotly.graph_objects as go
from collections import deque
import time
import matplotlib.pyplot as plt

# Force torch to use CPU for stability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

# Initialize session state variables
if 'bci_system' not in st.session_state:
    st.session_state.bci_system = create_bci_system()
if 'mode' not in st.session_state:
    st.session_state.mode = 'calibration'
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False
if 'inlet' not in st.session_state:
    st.session_state.inlet = None
if 'channel_data' not in st.session_state:
    st.session_state.channel_data = None
if 'figure' not in st.session_state:
    st.session_state.figure = None
if 'plot_placeholder' not in st.session_state:
    st.session_state.plot_placeholder = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'current_labels' not in st.session_state:
    st.session_state.current_labels = None
if 'current_sample_idx' not in st.session_state:
    st.session_state.current_sample_idx = 0
if 'eeg_channel' not in st.session_state:
    st.session_state.eeg_channel = None
if 'calibration_count' not in st.session_state:
    st.session_state.calibration_count = 0
if 'time_vector' not in st.session_state:
    st.session_state.time_vector = None
if 'plot_container' not in st.session_state:
    st.session_state.plot_container = st.empty()
if 'fig_dict' not in st.session_state:
    st.session_state.fig_dict = None

st.title("Sistema BCI para Reabilitação Pós-AVC")

# Seleção de modo
st.sidebar.header("Configuração")
mode = st.sidebar.radio("Modo de Operação", ['Calibração', 'Uso Real', 'Streaming'])
if mode == 'Calibração':
    st.session_state.mode = 'calibration'
elif mode == 'Uso Real':
    st.session_state.mode = 'real_use'
else:
    st.session_state.mode = 'streaming'

# Parâmetros de calibração no sidebar
if st.session_state.mode == 'calibration':
    st.sidebar.markdown("---")
    st.sidebar.header("Parâmetros de Calibração")
    subject_id = st.sidebar.number_input("ID do Paciente", 1, 109, 1)
    num_epochs = st.sidebar.slider("Épocas de Treinamento", 5, 30, 10)
    learning_rate = st.sidebar.slider("Taxa de Aprendizado", 1e-5, 1e-2, 1e-3, format="%.5f")
    batch_size = st.sidebar.slider("Tamanho do Lote", 1, 32, 4)  # Changed batch size range and default
    min_samples = st.sidebar.number_input("Amostras Mínimas para Calibração", 10, 100, 20)

    # Botão para carregar dados
    if st.sidebar.button("Carregar Dados do Paciente"):
        try:
            with st.spinner("Carregando dados do EEG..."):
                X, y, eeg_channel = load_local_eeg_data(subject_id=subject_id, augment=False)
                st.session_state.current_data = X
                st.session_state.current_labels = y
                st.session_state.current_sample_idx = 0
                st.session_state.eeg_channel = eeg_channel
                st.success(f"Dados carregados com sucesso! Total de amostras: {len(X)}")
        except Exception as e:
            st.error(f"Erro ao carregar dados: {str(e)}")

# LSL Stream Settings in sidebar
if st.session_state.mode == 'streaming':
    st.sidebar.markdown("---")
    st.sidebar.header("LSL Stream Settings")
    window_size = st.sidebar.slider("Window Size (seconds)", 1, 10, 5)
    update_interval = st.sidebar.slider("Update Interval (ms)", 50, 500, 100)

def init_lsl_stream():
    """Initialize LSL stream connection"""
    try:
        streams = resolve_streams(wait_time=1.0)
        eeg_streams = [stream for stream in streams if stream.type() == 'EEG']
        if not eeg_streams:
            return None
        inlet = StreamInlet(eeg_streams[0])
        return inlet
    except Exception as e:
        st.error(f"Error connecting to LSL stream: {str(e)}")
        return None

def create_empty_figure(n_channels, window_size, sample_rate=250):
    """Create an empty Plotly figure for streaming"""
    fig = go.Figure()
    times = np.linspace(-window_size, 0, int(window_size * sample_rate))
    st.session_state.time_vector = times
    
    for i in range(n_channels):
        fig.add_trace(go.Scatter(
            x=times,
            y=np.zeros_like(times),
            name=f'Channel {i+1}',
            mode='lines'
        ))
    
    fig.update_layout(
        title='Real-time EEG Data',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (µV)',
        showlegend=True,
        uirevision='constant',  # Prevents zooming reset on updates
        height=600,
        xaxis=dict(
            range=[-window_size, 0],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey',
        ),
        yaxis=dict(
            range=[-100, 100],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey',
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 0, 'redraw': False},
                                'fromcurrent': True,
                                'mode': 'immediate',
                                'transition': {'duration': 0}}],
                 'label': 'Play',
                 'method': 'animate'}
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'y': 0,
            'xanchor': 'right'
        }],
    )
    return fig

def update_streaming_plot():
    """Update the streaming plot with new data"""
    if not st.session_state.is_streaming or not st.session_state.inlet:
        return

    # Get chunk of data
    try:
        chunk, timestamps = st.session_state.inlet.pull_chunk(timeout=0.0, max_samples=32)
        
        if chunk:
            # Update channel data
            for sample in chunk:
                for i, value in enumerate(sample):
                    st.session_state.channel_data[i].append(value)
            
            # Create new figure with updated data
            fig = go.Figure()
            # Number of samples in buffer
            buf_len = st.session_state.channel_data[0].maxlen
            # Add traces for each channel using sample index on x-axis
            for i, channel_data in enumerate(st.session_state.channel_data):
                fig.add_trace(go.Scatter(
                    x=list(range(len(channel_data))),
                    y=list(channel_data),
                    name=f'Channel {i+1}',
                    mode='lines'
                ))
            
            # Update layout
            fig.update_layout(
                title='Real-time EEG Data',
                xaxis_title='Samples',
                yaxis_title='Amplitude (µV)',
                showlegend=True,
                uirevision=True,
                height=600,
                xaxis=dict(
                    title='Samples',
                    range=[0, buf_len],  # fixed sample window
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey',
                ),
                yaxis=dict(
                    title='Amplitude (µV)',
                    range=[-100, 100],  # fixed amplitude window
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGrey',
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
            )
            
            # Update the plot in the designated placeholder
            return fig
            
    except Exception as e:
        st.error(f"Error updating plot: {str(e)}")
        st.session_state.is_streaming = False
        return None

def initialize_figure_dict(n_channels, window_size, sample_rate=250):
    """Initialize the figure dictionary for efficient updates"""
    # Create time vector
    times = np.linspace(-window_size, 0, int(window_size * sample_rate))
    
    # Initialize traces for each channel
    traces = []
    for i in range(n_channels):
        trace = {
            'type': 'scatter',
            'x': times,
            'y': np.zeros_like(times),
            'name': f'Channel {i+1}',
            'mode': 'lines',
        }
        traces.append(trace)
    
    # Create the layout
    layout = {
        'title': 'Real-time EEG Data',
        'xaxis': {
            'title': 'Time (s)',
            'range': [-window_size, 0],
            'showgrid': True,
            'gridwidth': 1,
            'gridcolor': 'LightGrey'
        },
        'yaxis': {
            'title': 'Amplitude (µV)',
            'range': [-100, 100],
            'showgrid': True,
            'gridwidth': 1,
            'gridcolor': 'LightGrey'
        },
        'showlegend': True,
        'uirevision': True,
        'height': 600,
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'margin': {'l': 60, 'r': 20, 't': 40, 'b': 40}
    }
    
    return {'data': traces, 'layout': layout}

def update_figure_dict(fig_dict, channel_data):
    """Update figure dictionary with new data"""
    times = np.linspace(-window_size, 0, len(next(iter(channel_data))))
    for i, data in enumerate(channel_data):
        fig_dict['data'][i]['x'] = times
        fig_dict['data'][i]['y'] = list(data)
    return fig_dict

# Main content area
if st.session_state.mode == 'calibration':
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
            if st.button("Amostra Anterior", key="prev_sample_calibration"):
                st.session_state.current_sample_idx = max(0, st.session_state.current_sample_idx - 1)
                
        with col2:
            st.write(f"Amostra atual: {st.session_state.current_sample_idx + 1}/{len(st.session_state.current_data)}")
            
        with col3:
            if st.button("Próxima Amostra", key="next_sample_calibration"):
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
            if st.button("Usar para Calibração", key="use_for_calibration"):
                st.session_state.bci_system.add_calibration_sample(
                    current_sample,
                    current_label
                )
                st.session_state.calibration_count += 1
                st.success(f"Amostra adicionada! Total: {st.session_state.calibration_count}")

        # Inicializar/Treinar modelo
        if st.session_state.calibration_count >= min_samples:
            if st.button("Treinar Modelo", key="train_model"):
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

elif st.session_state.mode == 'real_use':
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
                if st.button("Amostra Anterior ", key="prev_sample_real_use"):
                    st.session_state.current_sample_idx = max(0, st.session_state.current_sample_idx - 1)
                    
            with col2:
                st.write(f"Amostra atual: {st.session_state.current_sample_idx + 1}/{len(st.session_state.current_data)}")
                
            with col3:
                if st.button("Próxima Amostra ", key="next_sample_real_use"):
                    st.session_state.current_sample_idx = min(
                        len(st.session_state.current_data) - 1,
                        st.session_state.current_sample_idx + 1
                    )

            if st.button("Classificar Movimento", key="classify_movement"):
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

elif st.session_state.mode == 'streaming':
    st.header("OpenBCI LSL Stream")
    
    # Create placeholders for dynamic content
    plot_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Stream control buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        if not st.session_state.is_streaming:
            if st.button("Start Stream", key="start_stream"):
                inlet = init_lsl_stream()
                if inlet:
                    info = inlet.info()
                    n_channels = info.channel_count()
                    sample_rate = int(info.nominal_srate())
                    buffer_size = int(window_size * sample_rate)
                    
                    # Initialize streaming state
                    st.session_state.channel_data = [deque(maxlen=buffer_size) for _ in range(n_channels)]
                    st.session_state.inlet = inlet
                    st.session_state.is_streaming = True
                    st.session_state.last_update = time.time()
                    
                    status_placeholder.success(f"Connected to {info.name()} stream with {n_channels} channels")
                else:
                    status_placeholder.error("No LSL streams found! Please ensure OpenBCI is streaming data.")
        else:
            if st.button("Stop Stream", key="stop_stream"):
                st.session_state.is_streaming = False
                st.session_state.inlet = None
                st.session_state.channel_data = None
                status_placeholder.info("Stream stopped")
    
    # Update visualization
    if st.session_state.is_streaming:
        fig = update_streaming_plot()
        if fig is not None:
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(update_interval / 1000.0)

# Mostrar métricas e estatísticas
if st.session_state.bci_system.is_calibrated:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Status do Sistema")
    st.sidebar.success("✓ Sistema Calibrado")
    if os.path.exists(st.session_state.bci_system.model_path):
        st.sidebar.success("✓ Modelo Salvo")