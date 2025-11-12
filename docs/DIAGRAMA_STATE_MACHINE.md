# DIAGRAMA DE MAQUINA DE ESTADOS

## ServerState (Servidor)

```
┌─────────┐
│ STOPPED │  (inicial)
└────┬────┘
     │ start_server()
     ↓
┌─────────┐
│ RUNNING │  (servidor iniciado, aguardando VR)
└────┬────┘
     │ (TCP conecta do VR)
     ↓
┌───────────┐
│ CONNECTED │  (VR conectado e pronto)
└────┬──────┘
     │ (TCP desconecta do VR)
     ↓
┌─────────┐
│ RUNNING │
└────┬────┘
     │ stop_server()
     ↓
┌─────────┐
│ STOPPED │
└─────────┘
```

### ServerState Checklist
- [x] Inicializa em STOPPED
- [x] start_server() → RUNNING
- [x] TCP conecta → CONNECTED
- [x] TCP desconecta → RUNNING
- [x] stop_server() → STOPPED
- [x] Transições bloqueadas (ex: IDLE → CONNECTED direto é impossível)

---

## SessionPhase (Sessão VR)

```
┌────────┐
│  IDLE  │  (inicial, sem sessao)
└───┬────┘
    │ start_session() com validacoes
    │
    ├─→ [FAIL] ← servidor nao ready
    │
    ↓
┌────────┐
│ SETUP  │  (enviando dados/tarefa ao VR)
└───┬────┘
    │
    ├─→ [TIMEOUT] ← se timeout, volta pra
    │              IDLE automaticamente
    │
    ↓
    │ VR responde com confirmacao
    │ (_process_vr_message detecta)
    ↓
┌────────┐
│ READY  │  (VR confirmou, esperando trigger)
└───┬────┘
    │
    ├─→ [CANCEL] → IDLE
    │
    ↓
    │ send_trigger()
    ↓
┌────────┐
│ ACTIVE │  (sessao em andamento)
└───┬────┘
    │ pode receber:
    │ - send_hand_close()
    │ - send_flower_action()
    │ - end_session()
    ↓
┌────────┐
│ ENDING │  (enviando finalizacao)
└───┬────┘
    │
    │ VR responde com confirmacao
    │ (_process_vr_message detecta)
    ↓
┌────────┐
│  IDLE  │  (volta ao inicial)
└────────┘
```

### SessionPhase Checklist
- [x] Inicializa em IDLE
- [x] Fluxo: IDLE → SETUP → READY → ACTIVE → ENDING → IDLE
- [x] Transições inválidas são bloqueadas
- [x] Cada estado só permite certas operacoes
- [x] Callbacks acionam transicoes (ex: confirmacao VR)

---

## Máquina de Estados Combinada

```
                    ┌─────────────────────────────────┐
                    │   SERVIDOR NAO PRONTO           │
                    │                                 │
                    │  ServerState != CONNECTED       │
                    │         OU                      │
                    │   SessionPhase != IDLE          │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────↓──────────────────┐
                    │   SERVIDOR PRONTO             │
                    │                               │
   ┌─ ServerState   │   ServerState = CONNECTED     │
   │   CONNECTED ───┤        AND                   │
   │                │   SessionPhase = IDLE       │
   └────────────────┤                               │
                    └───────────────────────────────┘


TIMELINE de uma Sessao Completa:

[USUARIO]                    [SERVIDOR]              [VR]
                            
                            ServerState=CONNECTED
                            SessionPhase=IDLE
                            
[clica "iniciar"]
                            start_session()
                            ├─ Verifica se pronto
                            └─ SessionPhase=SETUP
                                          ├─ envia dados
                                          ├─ envia tarefa
                                          │
                                          ├─→ [TCP] ─→ VR recebe
                                          │            e processa
                                          │
                                          ← [TCP] ← VR responde
                                          │        "Confirm"
                                          
                            _process_vr_message()
                            └─ SessionPhase=READY
                            
[clica "iniciar tarefa"]
                            send_trigger()
                            ├─ Verifica se em READY
                            └─ SessionPhase=ACTIVE
                                          ├─ envia "Trigger"
                                          │
                                          ├─→ [TCP] ─→ VR inicia
                                          │
[VR iniciou]                
                            
[durante a sessao]
                            send_hand_close('direita')
                            ├─ Verifica se ACTIVE
                            └─ envia "RIGHT_HAND_CLOSE"
                                          ├─→ [TCP] ─→ VR responde

[clica "terminar"]
                            end_session()
                            ├─ Verifica se ACTIVE
                            └─ SessionPhase=ENDING
                                          ├─ envia "Finalizar_tarefa"
                                          │
                                          ├─→ [TCP] ─→ VR encerra
                                          │
                                          ← [TCP] ← VR responde
                                          │        "Finalizar"
                                          
                            _process_vr_message()
                            └─ SessionPhase=IDLE
                            
[sessao encerrada]          SessionPhase=IDLE
                            SessionPhase=IDLE
                            Pronto pra nova sessao
```

---

## Transicoes Validas e Invalidas

### ServerState

| De | Para | Valido | Metodo |
|----|------|--------|--------|
| STOPPED | RUNNING | ✅ | start_server() |
| RUNNING | CONNECTED | ✅ | TCP conecta |
| CONNECTED | RUNNING | ✅ | TCP desconecta |
| RUNNING | STOPPED | ✅ | stop_server() |
| STOPPED | CONNECTED | ❌ | (impossivel) |
| RUNNING | RUNNING | ✅ | (ja esta) |
| Qualquer | Qualquer | ✅ | Se ja esta nesse |

### SessionPhase

| De | Para | Valido | Condicoes |
|----|------|--------|-----------|
| IDLE | SETUP | ✅ | start_session() chamado |
| SETUP | READY | ✅ | VR envia confirmacao |
| SETUP | IDLE | ✅ | Falha ao enviar dados |
| READY | ACTIVE | ✅ | send_trigger() chamado |
| READY | IDLE | ✅ | Cancelamento |
| ACTIVE | ENDING | ✅ | end_session() chamado |
| ENDING | IDLE | ✅ | VR envia confirmacao |
| IDLE | READY | ❌ | (impossivel, precisa SETUP) |
| IDLE | ACTIVE | ❌ | (impossivel, precisa SETUP + READY) |
| ACTIVE | IDLE | ❌ | (impossivel, precisa ENDING primeiro) |

---

## Estados Mutuamente Exclusivos

```
Nao eh possivel ter:

[ERRO 1] ❌ SessionPhase=IDLE + SessionPhase=SETUP
         (apenas UM phase por vez)

[ERRO 2] ❌ ServerState=STOPPED + mensagens TCP
         (servidor parado nao recebe)

[ERRO 3] ❌ SessionPhase=ACTIVE + status=IDLE
         (sessao nao pode ser ativa e idle)

[ERRO 4] ❌ ServerState=RUNNING + tcp_connected=True
         (se desconectou, volta pra RUNNING)
```

Graças a maquina de estados, esses erros sao **IMPOSSIVEIS** no codigo!

---

## Fluxo de Validacao

```
Operacao Solicitada (ex: send_trigger)
    │
    ├─ _is_session_waiting_trigger() ?
    │  └─ Verifica: phase == READY
    │
    ├─ Se FALSE → Retorna erro
    │
    ├─ Se TRUE → Processa operacao
    │
    ├─ _transition_session_phase(ACTIVE)
    │  └─ Verifica: READY.can_transition_to(ACTIVE) ?
    │
    ├─ Se FALSE → Volta atras, retorna erro
    │
    └─ Se TRUE → Transiciona, completa operacao
```

Cada validacao e em um lugar so!

---

## Resumo da Arquitetura

```
┌─────────────────────────────────────────────┐
│         UnityCommunicator                   │
├─────────────────────────────────────────────┤
│                                             │
│  ServerState ← Única fonte de verdade      │
│   (STOPPED/RUNNING/CONNECTED)              │
│                                             │
│  SessionPhase ← Única fonte de verdade     │
│   (IDLE/SETUP/READY/ACTIVE/ENDING)        │
│                                             │
│  Métodos Helpers:                          │
│   - _is_server_operational()               │
│   - _is_server_ready_for_session()         │
│   - _is_session_waiting_trigger()          │
│   - _is_session_active_for_commands()      │
│   - _transition_server_state()             │
│   - _transition_session_phase()            │
│                                             │
│  Métodos de Protocolo:                     │
│   - start_session()                        │
│   - send_trigger()                         │
│   - send_hand_close()                      │
│   - send_flower_action()                   │
│   - end_session()                          │
│                                             │
│  Tratamento de Mensagens:                  │
│   - _process_vr_message()                  │
│     └─ Aciona transicoes automaticas       │
│                                             │
└─────────────────────────────────────────────┘
```

Tudo passa por estados bem definidos e validados!
