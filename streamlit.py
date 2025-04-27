import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Import asyncio first and set policy
import asyncio
import platform
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ML1 import create_bci_system, load_local_eeg_data
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Force torch to use CPU for stability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

# Initialize session state variables
if 'bci_system' not in st.session_state:
    st.session_state.bci_system = create_bci_system()
if 'mode' not in st.session_state:
    st.session_state.mode = 'calibration'
if 'calibration_count' not in st.session_state:
    st.session_state.calibration_count = 0
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'current_labels' not in st.session_state:
    st.session_state.current_labels = None
if 'current_sample_idx' not in st.session_state:
    st.session_state.current_sample_idx = 0

st.title("Sistema BCI para Reabilitação Pós-AVC")

# Seleção de modo
st.sidebar.header("Configuração")
mode = st.sidebar.radio("Modo de Operação", ['Calibração', 'Uso Real'])
st.session_state.mode = 'calibration' if mode == 'Calibração' else 'real_use'

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

# Seção principal
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