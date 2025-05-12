# 🧠 Sistema de Interface Cérebro-Computador (BCI) para Reabilitação Pós-AVC

Este projeto implementa um sistema BCI completo para auxiliar na reabilitação de pacientes pós-AVC, utilizando sinais EEG do OpenBCI.

## ✨ Características Principais

- Interface gráfica PyQt5 para visualização e controle em tempo real
- Suporte para 16 canais EEG específicos do OpenBCI
- Pipeline de treinamento otimizado com PyTorch Lightning
- Interoperabilidade entre formatos CSV do OpenBCI e EDF
- Sistema de calibração personalizada por paciente
- Visualização em tempo real dos sinais EEG
- Classificação de movimento imaginado (esquerda/direita)

## 🔧 Configuração dos Canais EEG

O sistema utiliza os seguintes 16 canais EEG:
```python
canais = ['C3','C4','Fp1','Fp2','F7','F3','F4','F8',
          'T7','T8','P7','P3','P4','P8','O1','O2']
```

## 📊 Estrutura do Projeto

```
src/
├── model/              # Implementações dos modelos e processamento
│   ├── BCISystem.py    # Sistema BCI principal
│   ├── EEGAugmentation.py # Aumentação de dados EEG
│   └── ...
└── UI/                 # Interface gráfica
    ├── MainWindow.py   # Janela principal
    ├── CalibrationWidget.py # Widget de calibração
    └── ...
```

## 🚀 Como Usar

1. **Calibração**
   - Colete dados de calibração do paciente
   - Treine o modelo personalizado
   - Salve o modelo calibrado

2. **Uso em Tempo Real**
   - Carregue um modelo treinado
   - Conecte o dispositivo OpenBCI
   - Inicie a classificação em tempo real

3. **Testes Multi-Paciente**
   - Execute testes em múltiplos conjuntos de dados
   - Visualize métricas de desempenho
   - Compare resultados entre pacientes

## 📝 Notas Técnicas

### Interoperabilidade OpenBCI-CSV ↔ EDF

O sistema suporta:
- Conversão de CSV do OpenBCI para formato MNE Raw
- Transferência de anotações entre EDF e CSV
- Marcadores LSL para gravações ao vivo
- Coluna de TRIGGER opcional para exportação

### Pipeline de Processamento

1. Carregamento de dados brutos do OpenBCI
2. Pré-processamento e filtragem
3. Extração de características
4. Classificação usando redes neurais
5. Feedback em tempo real

## 🛠 Requisitos

- Python 3.x
- PyTorch
- PyQt5
- MNE-Python
- OpenBCI Python SDK
- pylsl (Lab Streaming Layer)
````

# projetoBCI
