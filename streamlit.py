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
from ML1 import load_and_process_data, load_model, EEGDataset, ModelTracker, evaluate_model, create_bci_system
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Force torch to use CPU for stability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

# Configuração inicial
if 'bci_system' not in st.session_state:
    st.session_state.bci_system = create_bci_system()
if 'mode' not in st.session_state:
    st.session_state.mode = 'calibration'
if 'calibration_count' not in st.session_state:
    st.session_state.calibration_count = 0

st.title("Sistema BCI para Reabilitação Pós-AVC")

# Seleção de modo
st.sidebar.header("Configuração")
mode = st.sidebar.radio("Modo de Operação", ['Calibração', 'Uso Real'])
st.session_state.mode = 'calibration' if mode == 'Calibração' else 'real_use'

# Parâmetros de calibração no sidebar
if st.session_state.mode == 'calibration':
    st.sidebar.markdown("---")
    st.sidebar.header("Parâmetros de Calibração")
    num_epochs = st.sidebar.slider("Épocas de Treinamento", 5, 30, 10)
    learning_rate = st.sidebar.slider("Taxa de Aprendizado", 1e-5, 1e-2, 1e-3, format="%.5f")
    batch_size = st.sidebar.slider("Tamanho do Lote", 8, 64, 32)
    min_samples = st.sidebar.number_input("Amostras Mínimas para Calibração", 10, 100, 20)

# Seção principal
if st.session_state.mode == 'calibration':
    st.header("Modo de Calibração")
    st.markdown("""
    Neste modo, você irá coletar dados de treinamento para calibrar o sistema.
    1. Peça ao paciente para imaginar o movimento da mão especificada
    2. Colete os dados do EEG
    3. Indique qual movimento foi imaginado
    4. Repita o processo até ter amostras suficientes
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Coletar Amostra Mão Esquerda"):
            # Simulando a coleta de dados EEG
            X, _, eeg_channel = load_and_process_data(subject_id=1, augment=False)
            sample = X[np.random.randint(0, len(X))]
            st.session_state.bci_system.add_calibration_sample(sample, 0)  # 0 para esquerda
            st.session_state.calibration_count += 1
            st.success(f"Amostra coletada! Total: {st.session_state.calibration_count}")

    with col2:
        if st.button("Coletar Amostra Mão Direita"):
            # Simulando a coleta de dados EEG
            X, _, eeg_channel = load_and_process_data(subject_id=1, augment=False)
            sample = X[np.random.randint(0, len(X))]
            st.session_state.bci_system.add_calibration_sample(sample, 1)  # 1 para direita
            st.session_state.calibration_count += 1
            st.success(f"Amostra coletada! Total: {st.session_state.calibration_count}")

    # Inicializar/Treinar modelo
    if st.session_state.calibration_count >= min_samples:
        if st.button("Treinar Modelo"):
            try:
                with st.spinner("Treinando o modelo..."):
                    if not st.session_state.bci_system.model:
                        X, _, eeg_channel = load_and_process_data(subject_id=1, augment=False)
                        st.session_state.bci_system.initialize_model(eeg_channel)
                    
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
        Neste modo, o sistema irá classificar os movimentos imaginados em tempo real.
        1. Peça ao paciente para imaginar o movimento de uma das mãos
        2. Clique em 'Classificar Movimento' para ver a predição
        """)

        if st.button("Classificar Movimento"):
            try:
                # Simulando a coleta de dados EEG em tempo real
                X, _, _ = load_and_process_data(subject_id=1, augment=False)
                sample = X[np.random.randint(0, len(X))]
                
                prediction, confidence = st.session_state.bci_system.predict_movement(sample)
                
                # Exibir resultado
                st.markdown("### Resultado da Classificação")
                st.markdown(f"**Movimento Detectado: {prediction}**")
                
                # Barra de confiança
                st.markdown("### Confiança da Predição")
                st.progress(confidence)
                st.write(f"Confiança: {confidence:.2%}")
                
                # Plotar o sinal EEG
                fig, ax = plt.subplots(figsize=(12, 4))
                for channel in range(sample.shape[0]):
                    ax.plot(sample[channel], alpha=0.5)
                ax.set_title("Sinal EEG Atual")
                ax.set_xlabel("Tempo")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Erro durante a classificação: {str(e)}")

# Mostrar métricas e estatísticas
if st.session_state.bci_system.is_calibrated:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Status do Sistema")
    st.sidebar.success("✓ Sistema Calibrado")
    if os.path.exists(st.session_state.bci_system.model_path):
        st.sidebar.success("✓ Modelo Salvo")