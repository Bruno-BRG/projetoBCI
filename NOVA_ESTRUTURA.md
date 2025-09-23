# BrainBridge - Nova Estrutura Proposta

## Visão Geral
Reorganização do sistema BrainBridge com estrutura mais limpa e separação clara de responsabilidades.

## Nova Estrutura de Diretórios

```
BrainBridge/
├── main.py                    # Ponto de entrada principal único
├── config/
│   ├── __init__.py
│   ├── settings.py           # Configurações centralizadas
│   └── constants.py          # Constantes do sistema
│
├── core/                     # Núcleo do sistema
│   ├── __init__.py
│   ├── eeg_data.py          # Classes para dados EEG
│   ├── patient.py           # Gestão de pacientes
│   └── session.py           # Sessões de gravação/streaming
│
├── acquisition/             # Aquisição de dados
│   ├── __init__.py
│   ├── udp_receiver.py      # Recebimento UDP
│   ├── data_logger.py       # Gravação de dados
│   └── simulators.py        # Simuladores para testes
│
├── processing/              # Processamento de sinais
│   ├── __init__.py
│   ├── filters.py           # Filtros digitais
│   ├── features.py          # Extração de características
│   └── preprocessing.py     # Pré-processamento
│
├── ml/                      # Machine Learning
│   ├── __init__.py
│   ├── models.py            # Definições de modelos
│   ├── trainer.py           # Treinamento
│   ├── predictor.py         # Predição em tempo real
│   └── evaluation.py        # Avaliação de modelos
│
├── gui/                     # Interface gráfica
│   ├── __init__.py
│   ├── main_window.py       # Janela principal
│   ├── widgets/             # Widgets customizados
│   │   ├── __init__.py
│   │   ├── eeg_plot.py      # Gráfico EEG
│   │   ├── patient_form.py  # Formulário de paciente
│   │   ├── streaming.py     # Widget de streaming
│   │   └── training.py      # Widget de treinamento
│   └── dialogs/             # Diálogos
│       ├── __init__.py
│       ├── patient_dialog.py
│       └── training_dialog.py
│
├── database/                # Banco de dados
│   ├── __init__.py
│   ├── manager.py           # Gerenciador de DB
│   └── models.py            # Modelos de dados
│
├── communication/           # Comunicação externa
│   ├── __init__.py
│   ├── esp32.py            # Comunicação ESP32
│   ├── unity.py            # Comunicação Unity
│   └── protocols.py        # Protocolos de comunicação
│
├── utils/                   # Utilitários
│   ├── __init__.py
│   ├── logging.py          # Sistema de logs
│   ├── validation.py       # Validações
│   └── helpers.py          # Funções auxiliares
│
├── data/                    # Dados do sistema
│   ├── recordings/          # Gravações EEG
│   ├── models/             # Modelos treinados
│   ├── database/           # Arquivo SQLite
│   └── logs/               # Logs do sistema
│
└── tests/                   # Testes
    ├── __init__.py
    ├── test_core.py
    ├── test_acquisition.py
    ├── test_processing.py
    └── test_ml.py
```

## Vantagens da Nova Estrutura

1. **Separação clara**: Cada módulo tem responsabilidade bem definida
2. **Escalabilidade**: Fácil adicionar novas funcionalidades
3. **Manutenibilidade**: Código organizado por funcionalidade
4. **Testabilidade**: Estrutura facilita criação de testes
5. **Ponto de entrada único**: `main.py` centraliza a inicialização

## Principais Mudanças

- **Unificação**: HardThinking integrado como módulo `ml/`
- **Simplificação**: Redução de níveis desnecessários de diretórios
- **Clareza**: Nomes mais descritivos e organização lógica
- **Flexibilidade**: Estrutura permite diferentes modos de execução

## Migração

A migração será feita gradualmente:
1. Criar nova estrutura de esqueleto
2. Mover funcionalidades uma por vez
3. Atualizar imports e dependências
4. Testar cada módulo migrado
5. Remover estrutura antiga