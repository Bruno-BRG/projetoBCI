"""
Script para baixar dados do PhysioNet EEG BCI dataset.
Os arquivos serão salvos em ./data/
"""

import warnings
import numpy as np
import mne
from mne.io import concatenate_raws
import os

warnings.filterwarnings("ignore")

# Configurações
N_SUBJECT = 109
BASELINE_EYE_OPEN = [1]
BASELINE_EYE_CLOSED = [2]
OPEN_CLOSE_LEFT_RIGHT_FIST = [3, 7, 11]
IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]
OPEN_CLOSE_BOTH_FIST = [5, 9, 13]
IMAGINE_OPEN_CLOSE_BOTH_FIST = [6, 10, 14]

# Caminho onde os dados serão salvos
data_path = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_path, exist_ok=True)

print(f"📥 Iniciando download dos dados do PhysioNet...")
print(f"📁 Caminho de destino: {data_path}")
print(f"👥 Número de sujeitos: {N_SUBJECT - 30} (1 a 79)")
print(f"🎯 Tarefas: Imaginação de movimento (esquerda/direita)")
print()

try:
    # Baixar dados dos sujeitos 1 a 79 (N_SUBJECT - 30)
    print("⏳ Baixando arquivos EDF do PhysioNet (isso pode levar alguns minutos)...")
    physionet_paths = []
    for subject_id in range(1, (N_SUBJECT - 60) + 1):
        for run in OPEN_CLOSE_LEFT_RIGHT_FIST:
            path = mne.datasets.eegbci.load_data(
                subject_id,
                runs=[run],
                path=data_path,
            )
            physionet_paths.extend(path)
    physionet_paths = np.array(physionet_paths)
    print(f"✅ Download concluído! {len(physionet_paths)} arquivos baixados.")
    print()

    # Ler os arquivos
    print("📖 Lendo os arquivos EDF...")
    parts = [
        mne.io.read_raw_edf(
            path,
            preload=True,
            stim_channel='auto',
            verbose='WARNING',
        )
        for path in physionet_paths
    ]
    print(f"✅ {len(parts)} arquivos lidos com sucesso.")
    print()

    # Concatenar dados
    print("🔗 Concatenando dados de todos os sujeitos...")
    raw = concatenate_raws(parts)
    print(f"✅ Dados concatenados. Shape: {raw.get_data().shape}")
    print()

    # Informações sobre o sinal
    sample_raw_data = raw.get_data()[0, :500]
    events, _ = mne.events_from_annotations(raw)
    print(f"📊 Total de eventos: {len(events)}")
    print()

    # Extrair informações dos canais EEG
    eeg_channel_inds = mne.pick_types(
        raw.info,
        meg=False,
        eeg=True,
        stim=False,
        eog=False,
        exclude='bads',
    )
    EEG_CHANNEL = int(len(eeg_channel_inds))
    print(f"🧠 Canais EEG detectados: {EEG_CHANNEL}")
    print()

    # Criar épocas
    print("⏱️ Criando épocas (1s a 4.1s após o estímulo)...")
    epoched = mne.Epochs(
        raw,
        events,
        dict(left=2, right=3),
        tmin=1,
        tmax=4.1,
        proj=False,
        picks=eeg_channel_inds,
        baseline=None,
        preload=True
    )
    print(f"✅ Épocas criadas. Total: {len(epoched)}")
    print()

    # Processar dados
    print("🔄 Processando dados (convertendo para float32)...")
    X = (epoched.get_data() * 1e3).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)
    CLASSES = ["left", "right"]
    
    print(f"✅ Dados processados!")
    print(f"   - Shape de X: {X.shape}")
    print(f"   - Shape de y: {y.shape}")
    print(f"   - Classes: {CLASSES}")
    print()

    # Salvar informações resumidas
    info_file = os.path.join(data_path, "dataset_info.txt")
    with open(info_file, 'w') as f:
        f.write("PhysioNet EEG BCI Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Sujeitos: 1 a 79 (total {N_SUBJECT - 30})\n")
        f.write(f"Tarefa: Imaginação de movimento (esquerda/direita)\n")
        f.write(f"Canais EEG: {EEG_CHANNEL}\n")
        f.write(f"Total de épocas: {len(epoched)}\n")
        f.write(f"Classes: {', '.join(CLASSES)}\n")
        f.write(f"Frequência de amostragem: {raw.info['sfreq']} Hz\n")
        f.write(f"Duração de cada época: 1s a 4.1s (3.1s total)\n")

    print(f"💾 Informações salvas em: {info_file}")
    print()
    print("✨ Download e processamento concluído com sucesso!")

except Exception as e:
    print(f"❌ Erro durante o download: {e}")
    import traceback
    traceback.print_exc()
