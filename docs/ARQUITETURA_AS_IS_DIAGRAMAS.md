# BrainBridge v2 - Arquitetura AS-IS

Este documento descreve a arquitetura **como esta implementada hoje** no codigo.
Fonte principal: `brainbridge_v2/`.

## 1) Diagrama de Contexto (C4 - Nivel 1)

```mermaid
flowchart LR
    Therapist["Terapeuta / Operador"]
    Patient["Paciente"]
    BrainBridge["BrainBridge v2 (Desktop App)"]
    OpenBCI["OpenBCI / Stream EEG via UDP"]
    UnityVR["Unity VR (TCP + UDP Broadcast + ZMQ)"]
    ESP32["ESP32 (Serial)"]
    SQLite["SQLite (pacientes + gravacoes)"]
    ModelFiles["Modelos .keras/.h5 (filesystem)"]
    CSVFiles["CSV OpenBCI (filesystem)"]

    Therapist --> BrainBridge
    Patient --> BrainBridge
    OpenBCI -->|Amostras EEG UDP| BrainBridge
    BrainBridge <-->|Comandos e mensagens| UnityVR
    BrainBridge <-->|Triggers LEFT/RIGHT| ESP32
    BrainBridge --> SQLite
    BrainBridge --> CSVFiles
    BrainBridge --> ModelFiles
```

## 2) Diagrama de Containers (C4 - Nivel 2)

```mermaid
flowchart TB
    subgraph App["BrainBridge v2"]
        Entry["main.py\nBootstrap + env checks"]
        GUI["PyQt5 GUI\nmain_window + widgets"]
        Acquisition["Acquisition\nstreaming_thread + udp_receiver"]
        Processing["Processing\nbutter_filter"]
        ML["ML\ntrainer + models + tensorflow_adapter + predictor"]
        Comm["Communication\nunity + esp32"]
        DB["Database\nmanager (sqlite3)"]
        Core["Core Domain\npatient + eeg_data + session"]
    end

    EEG["OpenBCI UDP Source"] --> Acquisition
    Acquisition --> Processing
    Processing --> GUI
    GUI --> ML
    GUI --> Comm
    GUI --> DB
    GUI --> Core
    GUI --> CSV["CSV Files"]
    ML --> ModelStore[".keras/.h5"]
    DB --> SQLite["SQLite DB"]
    Comm <--> Unity["Unity VR"]
    Comm <--> ESP["ESP32 Serial"]
```

## 2.1) Diagrama de Deploy (execucao local)

```mermaid
flowchart LR
    subgraph Host["PC Clinica / Windows"]
      App["Processo Python\nBrainBridge GUI"]
      Files["Filesystem\nCSV + modelos + logs"]
      Sql["SQLite local"]
    end

    subgraph Network["Rede local"]
      EEG["Streamer EEG UDP"]
      VR["Unity VR endpoint"]
    end

    MCU["ESP32 via USB Serial"]

    EEG -->|UDP 12345| App
    App <-->|TCP 12345 / UDP 12346 / ZMQ 5555| VR
    App <-->|Serial COMx| MCU
    App --> Files
    App --> Sql
```

## 3) Diagrama de Componentes (Runtime Principal)

```mermaid
flowchart LR
    Main["main.py"]
    MainWindow["gui.main_window.MainWindow"]
    StreamingWidget["gui.widgets.streaming.StreamingWidget"]
    PatientWidget["gui.widgets.patient_form.PatientRegistrationWidget"]
    DBM["database.manager.DatabaseManager"]
    ST["acquisition.streaming_thread.StreamingThread"]
    UR["acquisition.udp_receiver.UDPReceiver_BCI"]
    BF["processing.butter_filter.ButterworthFilter"]
    Logger["acquisition.data_logger.OpenBCICSVLogger"]
    Unity["communication.unity.UnityCommunicator + UDP_sender"]
    ESP32["communication.esp32.ESP32SerialCommunicator"]
    TF["ml.tensorflow_adapter.TensorFlowMLAdapter"]
    TrainDialog["gui.dialogs.training_dialog.TrainingDialog"]
    TrainThread["gui.training.model_trainer.ModelTrainerThread"]
    Trainer["ml.trainer.train_from_csvs"]

    Main --> MainWindow
    MainWindow --> DBM
    MainWindow --> PatientWidget
    MainWindow --> StreamingWidget
    PatientWidget --> DBM

    StreamingWidget --> ST
    ST --> UR
    ST --> BF
    ST --> StreamingWidget

    StreamingWidget --> Logger
    StreamingWidget --> Unity
    StreamingWidget --> ESP32
    StreamingWidget --> TF
    StreamingWidget --> DBM

    StreamingWidget --> TrainDialog
    TrainDialog --> TrainThread
    TrainThread --> Trainer
```

## 3.1) Diagrama de Classes (simplificado)

```mermaid
classDiagram
    class MainWindow {
      +db_manager: DatabaseManager
      +patient_widget: PatientRegistrationWidget
      +streaming_widget: StreamingWidget
    }
    class StreamingWidget {
      +toggle_connection()
      +toggle_recording()
      +predict_movement(eeg_data)
      +add_marker(marker)
      +send_udp_signal(direction)
      +send_esp32_signal(direction)
    }
    class StreamingThread {
      +start_streaming(host, port)
      +stop_streaming()
      +extract_eeg_from_udp(data)
    }
    class UDPReceiver_BCI {
      +start()
      +stop()
      +set_callback(cb)
    }
    class OpenBCICSVLogger {
      +log_sample(eeg_data, marker)
      +start_baseline()
      +stop_logging()
    }
    class UnityCommunicator {
      +start_server()
      +stop_server()
      +start_session(patient, task)
      +send_trigger()
      +end_task()
      +end_session()
    }
    class ESP32SerialCommunicator {
      +connect()
      +disconnect()
      +send_trigger_left()
      +send_trigger_right()
    }
    class DatabaseManager {
      +add_patient()
      +get_all_patients()
      +add_recording()
    }
    class TensorFlowMLAdapter {
      +load_model(path)
      +predict(data)
      +predict_on_window(window)
    }

    MainWindow --> StreamingWidget
    MainWindow --> DatabaseManager
    StreamingWidget --> StreamingThread
    StreamingWidget --> OpenBCICSVLogger
    StreamingWidget --> UnityCommunicator
    StreamingWidget --> ESP32SerialCommunicator
    StreamingWidget --> DatabaseManager
    StreamingWidget --> TensorFlowMLAdapter
    StreamingThread --> UDPReceiver_BCI
```

## 4) Diagrama de Camadas (AS-IS vs ideal clean)

```mermaid
flowchart TB
    subgraph L1["Camada de Apresentacao"]
      GUI["gui/*\nMainWindow, StreamingWidget, PatientForm, dialogs"]
    end

    subgraph L2["Camada de Aplicacao (orquestracao)"]
      Orq["StreamingWidget (orquestra quase tudo)\nmain.py bootstrap"]
    end

    subgraph L3["Camada de Dominio"]
      Domain["core/*\nPatient, EEGSample/EEGSession, Session"]
    end

    subgraph L4["Infraestrutura"]
      InfraA["acquisition/* (UDP, logging)"]
      InfraP["processing/* (filtro)"]
      InfraM["ml/* (treino/inferencia)"]
      InfraC["communication/* (Unity/ESP32)"]
      InfraD["database/* (sqlite)"]
    end

    GUI --> Orq
    Orq --> InfraA
    Orq --> InfraP
    Orq --> InfraM
    Orq --> InfraC
    Orq --> InfraD
    Orq -. uso parcial .-> Domain
```

Leitura pratica:
- Existe separacao por pastas/modulos, mas a orquestracao esta concentrada no `StreamingWidget`.
- `core/*` existe como dominio, porem tem baixo uso no fluxo principal GUI.

## 5) Sequencia - Streaming e Gravacao

```mermaid
sequenceDiagram
    actor User as Operador
    participant SW as StreamingWidget
    participant ST as StreamingThread
    participant UR as UDPReceiver_BCI
    participant BF as ButterworthFilter
    participant Log as OpenBCICSVLogger
    participant DB as DatabaseManager
    participant U as UnityCommunicator

    User->>SW: Conectar
    SW->>ST: start_streaming(host, port)
    ST->>UR: start() + callback
    UR-->>ST: payload UDP
    ST->>BF: apply_realtime_filter(sample)
    ST-->>SW: data_received(sample filtrado)

    User->>SW: Iniciar gravacao
    SW->>Log: cria arquivo CSV OpenBCI
    SW->>DB: add_recording(...)
    SW->>U: send_trigger() (tentativa)

    loop cada amostra
      ST-->>SW: data_received
      SW->>Log: log_sample(data, marker?)
    end

    User->>SW: Adicionar marcador T1/T2
    SW->>Log: pendencia de marker (T1/T2, auto T0)

    User->>SW: Parar gravacao
    SW->>Log: stop_logging()
    SW->>U: end_task() / end_session() se jogo
```

## 6) Sequencia - Modo Jogo (Predicao + Trigger Unity/ESP32)

```mermaid
sequenceDiagram
    actor User as Operador
    participant SW as StreamingWidget
    participant TF as TensorFlowMLAdapter
    participant U as UDP_sender/UnityCommunicator
    participant E as ESP32SerialCommunicator

    User->>SW: Seleciona tarefa "Jogo" e inicia
    SW->>SW: Carrega modelo .keras/.h5
    SW->>SW: Abre janela de IA (ai_prediction_enabled)

    loop janelas EEG (window_size)
      SW->>TF: predict_proba(X)
      TF-->>SW: probs + classe
      alt classe esquerda
        SW->>U: enviar_sinal("esquerda")
        SW->>E: send_trigger_left()
      else classe direita
        SW->>U: enviar_sinal("direita")
        SW->>E: send_trigger_right()
      end
      SW->>SW: prediction_locked=True
    end

    U-->>SW: mensagem CORRECT/WRONG/FLOWER
    SW->>SW: atualiza acuracia + libera proxima janela/sinal
```

## 7) Estado da Sessao Unity (implementacao atual)

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Setup: start_session(patient, task)
    Setup --> Ready: transition_to(READY) manual/externo
    Ready --> Active: send_trigger() + hand_close
    Active --> Ending: end_task()/end_session()
    Ending --> Idle: reset()
    Setup --> Idle: reset()/erro
    Ready --> Idle: reset()/cancelamento
    Active --> Idle: reset() direto (possivel no codigo)
```

Observacao:
- `SessionPhase` e `ServerState` existem no modulo, mas o controle efetivo de servidor ainda usa bastante `is_active` e `tcp_connected`.

## 8) Dependencias entre pacotes internos

```mermaid
flowchart LR
    gui --> acquisition
    gui --> processing
    gui --> ml
    gui --> communication
    gui --> database
    gui --> config

    acquisition --> processing
    acquisition --> acquisition

    ml --> processing
    ml --> config
    ml --> ml

    database --> config
    core --> core
    communication --> communication
```

Interpretacao:
- `gui` depende diretamente de quase tudo.
- Dependencia de `core` para regras de negocio no fluxo principal e baixa.

## 9) SOLID - Diagnostico objetivo (AS-IS)

| Principio | Estado atual | Evidencia no codigo | Impacto |
|---|---|---|---|
| S - Single Responsibility | Parcial / fraco em pontos criticos | `StreamingWidget` concentra UI, rede, log, inferencia, treino, protocolo | Alto acoplamento e manutencao dificil |
| O - Open/Closed | Parcial | Estrutura modular por pastas existe, mas extensoes exigem editar widgets centrais | Evolucao lenta sem quebrar fluxo |
| L - Liskov | Neutro | Pouca hierarquia/polimorfismo classico | Baixo impacto |
| I - Interface Segregation | Fraco | Nao ha interfaces pequenas para comunicacao, logger, inferencia | Componentes conhecem detalhes concretos |
| D - Dependency Inversion | Fraco no fluxo principal | GUI depende de classes concretas (`DatabaseManager`, `UnityCommunicator`, `ESP32...`) | Testabilidade e troca de adaptadores reduzidas |

## 10) Pontos fortes e riscos de arquitetura

### Pontos fortes
- Separacao fisica por modulos (`acquisition`, `ml`, `communication`, etc.).
- Fluxo end-to-end funcional no desktop (captura, grava, treina, prediz, envia trigger).
- Componentes de infraestrutura importantes ja isolados em arquivos proprios.

### Riscos principais
- `StreamingWidget` atua como "God Object" da aplicacao.
- Divergencia entre docs/testes e implementacao atual do estado de servidor Unity.
- `processing/preprocessing.py` referencia `.filters` que nao existe no pacote atual.
- Duplicidade de dialogos de treino (`gui/widgets/training.py` e `gui/dialogs/training_dialog.py`).
- Dominio (`core`) subutilizado na orquestracao principal.

## 11) Leitura final: "como o projeto esta hoje"

Arquitetura atual e **modular por pastas**, com **integracao funcional real**, mas com um **centro de gravidade excessivo na camada GUI** (especialmente `StreamingWidget`). O projeto esta numa fase entre "monolito modular" e "arquitetura em camadas de fato": os blocos existem, porem a inversao de dependencia e a distribuicao de responsabilidades ainda nao estao maduras.
