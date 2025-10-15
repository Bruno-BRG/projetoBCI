"""
Script de teste para verificar o transformador.py
Este script simula o processamento de um arquivo EDF com anotações T0, T1, T2
"""

import numpy as np
import mne
import os
import tempfile

def criar_edf_teste():
    """
    Cria um arquivo EDF de teste com anotações T0, T1, T2
    """
    # Parâmetros
    sfreq = 125  # Hz (mesma taxa do OpenBCI)
    n_channels = 16
    duracao = 60  # segundos
    n_samples = int(sfreq * duracao)
    
    # Nomes dos canais (padrão 10-20)
    ch_names = ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T7', 'T8', 
                'C3', 'C4', 'P7', 'P8', 'P3', 'P4', 'O1', 'O2']
    
    # Criar dados sintéticos (ruído + alguma atividade simulada)
    # Valores em Volts para MNE (será convertido internamente)
    np.random.seed(42)
    data = np.random.randn(n_channels, n_samples) * 50e-6  # 50 microvolts em volts
    
    # Criar info do MNE
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Criar objeto Raw
    raw = mne.io.RawArray(data, info)
    
    # Adicionar anotações (eventos)
    # T1 = mão esquerda em 10s
    # T0 = repouso em 20s
    # T2 = mão direita em 30s
    # T0 = repouso em 40s
    # T1 = mão esquerda em 50s
    
    onset = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # tempos em segundos
    duration = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # duração dos eventos
    description = ['T1', 'T0', 'T2', 'T0', 'T1']  # descrições
    
    annotations = mne.Annotations(onset=onset, duration=duration, description=description)
    raw.set_annotations(annotations)
    
    # Salvar arquivo EDF temporário
    temp_dir = tempfile.gettempdir()
    edf_path = os.path.join(temp_dir, 'teste_eeg.edf')
    
    # Exportar para EDF
    raw.export(edf_path, overwrite=True)
    
    print(f"✓ Arquivo EDF de teste criado: {edf_path}")
    print(f"  - Canais: {n_channels}")
    print(f"  - Taxa de amostragem: {sfreq} Hz")
    print(f"  - Duração: {duracao} segundos")
    print(f"  - Anotações: {len(annotations)} eventos")
    
    return edf_path


def testar_transformador(arquivo_edf):
    """
    Testa o transformador com o arquivo EDF criado
    """
    print("\n" + "="*60)
    print("TESTANDO O TRANSFORMADOR")
    print("="*60)
    
    # Importar a função
    import sys
    sys.path.insert(0, '/home/apolo/dev/BrainBridge/tools')
    from transformador import processar_edf_para_openbci
    
    # Configurações de teste
    canais_desejados = []  # Vazio = todos os canais
    filtragem = (0.5, 60)
    algarismos_significativos = 5
    
    # Processar
    try:
        caminho_saida = processar_edf_para_openbci(
            arquivo_edf, 
            filtragem, 
            canais_desejados, 
            algarismos_significativos
        )
        
        print("\n" + "="*60)
        print("✓ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*60)
        
        # Verificar o arquivo gerado
        import pandas as pd
        
        # Ler o CSV pulando as linhas de cabeçalho
        df = pd.read_csv(caminho_saida, skiprows=4)
        
        print(f"\n📊 Informações do arquivo gerado:")
        print(f"  - Arquivo: {caminho_saida}")
        print(f"  - Total de amostras: {len(df)}")
        print(f"  - Colunas: {len(df.columns)}")
        print(f"  - Coluna Annotations presente: {'Annotations' in df.columns}")
        
        # Verificar anotações
        if 'Annotations' in df.columns:
            anotacoes = df[df['Annotations'].notna() & (df['Annotations'] != '')]
            print(f"\n🎯 Anotações encontradas no CSV: {len(anotacoes)}")
            
            if len(anotacoes) > 0:
                print("\n📍 Detalhes das anotações:")
                for idx, row in anotacoes.iterrows():
                    print(f"  Sample {idx}: {row['Annotations']}")
        
        # Mostrar primeiras linhas
        print("\n📄 Primeiras 3 linhas do CSV (colunas principais):")
        print(df[['Sample Index', 'EXG Channel 0', 'EXG Channel 1', 'Annotations']].head(3))
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO durante o processamento: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 INICIANDO TESTE DO TRANSFORMADOR EDF → CSV OpenBCI")
    print("="*60)
    
    # Criar arquivo EDF de teste
    arquivo_edf = criar_edf_teste()
    
    # Testar o transformador
    sucesso = testar_transformador(arquivo_edf)
    
    if sucesso:
        print("\n✅ TODOS OS TESTES PASSARAM!")
    else:
        print("\n❌ TESTE FALHOU!")
