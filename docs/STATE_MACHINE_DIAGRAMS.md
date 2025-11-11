# 🔄 Máquina de Estados - Diagrama Visual

## Fluxo Completo: Sistema → VR

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            MÁQUINA DE ESTADOS                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SERVIDOR                          SESSÃO                                   │
│  ───────────                        ───────                                 │
│                                                                              │
│  ┌──────────────┐                                                           │
│  │   STOPPED    │◄──────────────────────────────────────────┐               │
│  └──────────────┘                                          │               │
│         │                                                  │               │
│         │ start_server()                                  │               │
│         ▼                                                  │               │
│  ┌──────────────┐       IDLE          SETUP     READY    │               │
│  │   RUNNING    │       ────          ────      ─────    │               │
│  └──────────────┘        │             │          │      │               │
│         │                 │             │          │      │               │
│         │ (VR conecta)    │             │          │      │               │
│         ▼                 ▼             ▼          ▼      │               │
│  ┌──────────────┐       (idle)──────>(setup)──>(ready)   │               │
│  │  CONNECTED   │                      │          │       │               │
│  └──────────────┘                      │          │       │               │
│         │                              │          │       │               │
│         │ (VR desconecta)              │          │       │               │
│         ▼                              │          │       │               │
│  ┌──────────────┐                      │          ▼       │               │
│  │   RUNNING    │                      │       (active)   │               │
│  └──────────────┘                      │          │       │               │
│         │                              │          │       │               │
│         │ stop_server()                │          │       │               │
│         ▼                              │          │       │               │
│  ┌──────────────┐◄─────────────────────┴──────────┴───────┘               │
│  │   STOPPED    │          reset() / erro / cancelamento                  │
│  └──────────────┘                                                         │
│                                                                            │
│         ┌─────────────────────────────────────────────┐                   │
│         │ Fluxo Normal: IDLE→SETUP→READY→ACTIVE→     │                   │
│         │ ENDING→IDLE                                │                   │
│         └─────────────────────────────────────────────┘                   │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Transições Válidas

```
SessionPhase Transitions:
═══════════════════════════

IDLE (Estado Inicial)
  ├─→ SETUP .................... Iniciar comunicação
  └─→ Fim do processo (reset)

SETUP (Enviando dados)
  ├─→ READY .................... VR confirmou recepção
  ├─→ IDLE ..................... Erro na transmissão (fallback)
  └─→ Não pode ir para: ACTIVE, ENDING

READY (Aguardando comando)
  ├─→ ACTIVE ................... Trigger enviado
  ├─→ IDLE ..................... Cancelamento
  └─→ Não pode ir para: SETUP, ENDING

ACTIVE (Sessão em andamento)
  ├─→ ENDING ................... Finalizar sessão
  └─→ Não pode ir para: IDLE, SETUP, READY

ENDING (Finalizando)
  ├─→ IDLE ..................... VR confirmou finalização
  └─→ Não pode ir para: SETUP, READY, ACTIVE
```

## Transições Bloqueadas

```
❌ Transições INVÁLIDAS (Bloqueadas):
═════════════════════════════════════

IDLE
  ❌ → READY (pula SETUP)
  ❌ → ACTIVE (pula SETUP e READY)
  ❌ → ENDING (pula tudo)

SETUP
  ❌ → ACTIVE (pula READY)
  ❌ → ENDING (pula READY e ACTIVE)

READY
  ❌ → SETUP (não volta)
  ❌ → ENDING (pula ACTIVE)

ACTIVE
  ❌ → IDLE (pula ENDING)
  ❌ → SETUP (não volta)
  ❌ → READY (não volta)

ENDING
  ❌ → ACTIVE (não volta)
  ❌ → READY (não volta)
  ❌ → SETUP (não volta)
```

## Casos de Uso: Fluxos Felizes

### ✅ Fluxo 1: Sessão Completa

```
┌────────────────────────────────────────────────────────────────┐
│ FLUXO 1: Treino Completo                                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. start_server()                                            │
│     ServerState: STOPPED → RUNNING                            │
│                                                                │
│  2. (VR conecta na porta TCP)                                 │
│     ServerState: RUNNING → CONNECTED                          │
│                                                                │
│  3. start_session(patient, TaskType.TREINO)                   │
│     SessionPhase: IDLE → SETUP                                │
│     Ação: Enviar dados do paciente                            │
│                                                                │
│  4. (VR responde com "Confirm")                               │
│     SessionPhase: SETUP → READY                               │
│                                                                │
│  5. send_trigger()                                            │
│     SessionPhase: READY → ACTIVE                              │
│     Ação: VR inicia treino                                    │
│                                                                │
│  6. send_hand_close("direita")                                │
│     SessionPhase: ACTIVE (sem mudança)                        │
│     Ação: VR fecha mão direita                                │
│                                                                │
│  7. end_session("Parabéns!")                                  │
│     SessionPhase: ACTIVE → ENDING                             │
│     Ação: Enviar finalização                                  │
│                                                                │
│  8. (VR responde com "Finalizar")                             │
│     SessionPhase: ENDING → IDLE                               │
│     Pronto pra nova sessão!                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### ✅ Fluxo 2: Erro durante SETUP

```
┌────────────────────────────────────────────────────────────────┐
│ FLUXO 2: Erro na Transmissão                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. start_session(...)                                        │
│     SessionPhase: IDLE → SETUP                                │
│                                                                │
│  2. (Erro ao enviar dados)                                    │
│     Ação: Log de erro                                         │
│                                                                │
│  3. transition_to(SessionPhase.IDLE)  [FALLBACK]              │
│     SessionPhase: SETUP → IDLE                                │
│                                                                │
│  4. Pode retentar start_session()                             │
│     SessionPhase: IDLE → SETUP (novamente)                    │
│                                                                │
│  Benefício: Sem deixar em estado inconsistente! ✅            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### ✅ Fluxo 3: Cancelamento antes de iniciar

```
┌────────────────────────────────────────────────────────────────┐
│ FLUXO 3: Usuário Cancela Antes do Trigger                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. start_session(patient, TaskType.JOGO)                    │
│     SessionPhase: IDLE → SETUP                                │
│     Ação: Enviando dados                                      │
│                                                                │
│  2. (VR confirma)                                             │
│     SessionPhase: SETUP → READY                               │
│     Estado: Aguardando trigger                                │
│                                                                │
│  3. (Usuário muda de ideia)                                   │
│     Ação: Cancel button clicado                               │
│                                                                │
│  4. session.transition_to(SessionPhase.IDLE)                  │
│     SessionPhase: READY → IDLE                                │
│                                                                │
│  5. reset() (opcional, mas limpeza)                           │
│     session.patient = None                                    │
│     session.task_type = None                                  │
│                                                                │
│  Resultado: Voltou ao inicial limpo ✅                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Helpers: Verificações Centralizadas

```
┌─────────────────────────────────────────────────────────────────────────┐
│ HELPERS DE VERIFICAÇÃO                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ _is_server_operational()                                               │
│   ├─ Retorna: True se ServerState == RUNNING ou CONNECTED             │
│   ├─ Uso: Verificar se servidor está pronto                           │
│   └─ Exemplo: if comm._is_server_operational(): ...                   │
│                                                                         │
│ _is_server_ready_for_session()                                         │
│   ├─ Retorna: True se ServerState==CONNECTED E SessionPhase==IDLE     │
│   ├─ Uso: Verificar pré-requisitos pra start_session()               │
│   └─ Exemplo: if comm._is_server_ready_for_session(): ...             │
│                                                                         │
│ _is_session_waiting_trigger()                                          │
│   ├─ Retorna: True se SessionPhase == READY                           │
│   ├─ Uso: Saber se pode enviar trigger                                │
│   └─ Exemplo: if comm._is_session_waiting_trigger(): ...              │
│                                                                         │
│ _is_session_active_for_commands()                                      │
│   ├─ Retorna: True se SessionPhase == ACTIVE                          │
│   ├─ Uso: Verificar se pode enviar comandos (HAND_CLOSE, FLOWER)      │
│   └─ Exemplo: if comm._is_session_active_for_commands(): ...          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Proteções da Máquina de Estados

```
┌──────────────────────────────────────────────────────────────┐
│ PROTEÇÕES AUTOMÁTICAS                                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 1. Transições Inválidas                                     │
│    ✅ Bloqueadas na enum                                    │
│    ✅ transition_to() retorna False                         │
│    ✅ Estado não muda se inválido                          │
│                                                              │
│ 2. Estados Contraditórios                                   │
│    ✅ Impossível ter is_active=True E waiting_confirmation  │
│    ✅ SessionPhase é a ÚNICA fonte de verdade              │
│    ✅ ServerState é a ÚNICA fonte de verdade               │
│                                                              │
│ 3. Fallback Automático                                      │
│    ✅ Pode voltar de SETUP/READY para IDLE                 │
│    ✅ Facilita recuperação de erro                         │
│    ✅ Sem deixar em estado inválido                        │
│                                                              │
│ 4. Reset Atômico                                            │
│    ✅ session.reset() limpa tudo de uma vez                │
│    ✅ Sem deixar dados antigos                             │
│    ✅ Volta para IDLE garantido                            │
│                                                              │
│ 5. Validação de Dados                                       │
│    ✅ PatientData valida nivel (0-11)                      │
│    ✅ PatientData valida lado (Esquerdo/Direito)           │
│    ✅ TaskType restringe a (TREINO/JOGO)                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Sumário: Estados Válidos

```
╔════════════════════════════════════════════════════════════╗
║ ESTADOS POSSÍVEIS NA MÁQUINA                              ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║ ServerState:                                              ║
║   • STOPPED (3 caracteres) ..................... início  ║
║   • RUNNING (3 caracteres) ..................... meio    ║
║   • CONNECTED (1 caractere) .................... fim     ║
║                                                            ║
║ SessionPhase:                                             ║
║   • IDLE (1 caractere) ......................... Início  ║
║   • SETUP (1 caractere) ........................ Meio    ║
║   • READY (1 caractere) ........................ Meio    ║
║   • ACTIVE (1 caractere) ....................... Ação    ║
║   • ENDING (1 caractere) ........................ Fim     ║
║                                                            ║
║ TOTAL: 8 estados possíveis ✅                             ║
║                                                            ║
║ Alternativas antes da refatoração:                        ║
║   is_active: True/False (2 valores)                       ║
║   tcp_connected: True/False (2 valores)                   ║
║   session.is_active: True/False (2 valores)               ║
║   session.waiting_confirmation: True/False (2 valores)    ║
║   TOTAL: 2×2×2×2 = 16 combinações ❌                      ║
║   (mas APENAS 4 eram válidas!)                            ║
║                                                            ║
║ Redução de complexidade:                                  ║
║   Antes: 16 combinações (4 válidas) = 75% inválidas      ║
║   Depois: 8 estados (todos válidos) = 0% inválidos       ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

## Conclusão

A máquina de estados elimina:
- ✅ Combinações inválidas de estado
- ✅ Verificações cruzadas de múltiplas variáveis
- ✅ Bugs causados por inconsistência
- ✅ Código condicional complexo

E oferece:
- ✅ Transições claras e validadas
- ✅ Uma fonte de verdade por domínio
- ✅ Fallback automático
- ✅ Fácil de debugar e testar

**Resultado: O Ouroboros foi domesticado! 🐍→✨**
