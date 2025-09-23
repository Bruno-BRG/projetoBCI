# BrainBridge v2

Sistema BCI (Brain-Computer Interface) para classificação de Motor Imagery utilizando sinais EEG.

## Nova Estrutura

Esta é a versão refatorada do BrainBridge, seguindo uma arquitetura mais limpa e modular conforme especificado no documento `NOVA_ESTRUTURA.md`.

## Instalação

```bash
cd brainbridge_v2
pip install -r requirements.txt
```

## Uso

### Interface Gráfica (Padrão)
```bash
python main.py
```

### Interface de Linha de Comando
```bash
python main.py --cli
```

### Modo Simulação
```bash
python main.py --simulate
```

### Treinamento de Modelos
```bash
python main.py --train
```

### Verificar Ambiente
```bash
python main.py --check-env
```

## Estrutura de Diretórios

```
brainbridge_v2/
├── main.py                    # Ponto de entrada único
├── config/                    # Configurações
├── core/                      # Classes fundamentais
├── acquisition/               # Aquisição de dados
├── processing/                # Processamento de sinais
├── ml/                        # Machine Learning
├── gui/                       # Interface gráfica
├── database/                  # Banco de dados
├── communication/             # Comunicação externa
├── utils/                     # Utilitários
├── data/                      # Dados do sistema
└── tests/                     # Testes
```

## Status de Desenvolvimento

- ✅ Estrutura de diretórios criada
- ✅ Configurações básicas implementadas
- ✅ Classes core (EEG, Patient, Session)
- ✅ Módulo de aquisição (UDP, Logger, Simuladores)
- ✅ Processamento básico (Filtros, Pré-processamento)
- ✅ Ponto de entrada principal
- 🔄 Interface gráfica (em desenvolvimento)
- 🔄 Machine Learning (em desenvolvimento)
- 🔄 Database (em desenvolvimento)
- 🔄 Comunicação externa (em desenvolvimento)

## Próximos Passos

1. Implementar interface gráfica completa
2. Migrar funcionalidade de ML do HardThinking
3. Implementar gerenciador de banco de dados
4. Adicionar comunicação ESP32/Unity
5. Criar testes automatizados