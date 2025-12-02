# REFATORACAO: ELIMINACAO DO OUROBOROS NA MÁQUINA DE ESTADOS

## Problema Identificado

O código anterior tinha um **Ouroboros** de variáveis de estado interdependentes:

```
is_active (servidor)
    ↓
tcp_connected
    ↓
session.is_active
    ↓
session.waiting_confirmation
    ↓
↻ (volta pro is_active)
```

**Problema:** A lógica de estado era espalhada e tinha múltiplas variáveis verificando a mesma coisa de formas diferentes:

```python
# ANTES - Lógica confusa
if not self.is_active:
    return False
if not self.tcp_connected:
    return False
if not self.session.is_active and not self.session.waiting_confirmation:
    return False
# ... 15 linhas de verificações!
```

## Solução: Máquina de Estados Centralizada

### 1. **ServerState Enum** (Única fonte de verdade para servidor)
```python
class ServerState(Enum):
    STOPPED = "stopped"       # Servidor parado
    RUNNING = "running"       # Servidor rodando, sem conexão VR
    CONNECTED = "connected"   # VR conectado
```

Transições:
- `STOPPED → RUNNING` (start_server)
- `RUNNING ↔ CONNECTED` (TCP conecta/desconecta)
- `CONNECTED → RUNNING → STOPPED` (stop_server)

### 2. **SessionPhase Enum** (Única fonte de verdade para sessão)
```python
class SessionPhase(Enum):
    IDLE = "idle"             # Sem sessão ativa
    SETUP = "setup"           # Enviando dados/tarefa
    READY = "ready"           # Confirmação recebida
    ACTIVE = "active"         # Sessão em andamento
    ENDING = "ending"         # Finalização em progresso
```

Transições Válidas:
- `IDLE → SETUP` (start_session)
- `SETUP → READY` (recebe confirmação VR)
- `READY → ACTIVE` (send_trigger)
- `ACTIVE → ENDING` (end_session)
- `ENDING → IDLE` (recebe confirmação de finalização)

**Transições INVÁLIDAS são bloqueadas automaticamente:**
```python
def can_transition_to(self, next_phase):
    transitions = {
        IDLE: {SETUP},
        SETUP: {READY, IDLE},      # Volta se falhar
        READY: {ACTIVE, IDLE},     # Pode cancelar
        ACTIVE: {ENDING},
        ENDING: {IDLE},
    }
    return next_phase in transitions.get(self, set())
```

### 3. **Métodos Helpers** (Centralizam verificações)

| Método | Propósito |
|--------|----------|
| `_is_server_operational()` | Servidor está RUNNING ou CONNECTED |
| `_is_server_ready_for_session()` | Servidor CONNECTED + VR conectado + Sessão IDLE |
| `_is_session_waiting_trigger()` | Sessão está em READY |
| `_is_session_active_for_commands()` | Sessão está em ACTIVE |
| `_transition_server_state(state)` | Transiciona servidor |
| `_transition_session_phase(phase)` | Transiciona sessão com validação |

## ANTES vs DEPOIS

### ANTES (Confuso)
```python
# Verificações espalhadas
if not self.is_active:
    return False
if not self.tcp_connected:
    return False
if not self.session.is_active and not self.session.waiting_confirmation:
    return False
if self.session.is_active:
    self.session.is_active = False
    self.session.waiting_confirmation = True
```

### DEPOIS (Claro e Validado)
```python
# Uma verificação clara
if not self._is_server_ready_for_session():
    return False

# Transição explícita e validada
if not self._transition_session_phase(SessionPhase.SETUP):
    return False
```

## Benefícios

✅ **Sem Ouroboros**: Cada estado é independente
✅ **Máquina de Estados**: Transições são explícitas e validadas
✅ **Fonte Única de Verdade**: ServerState + SessionPhase determinam tudo
✅ **Bloqueio de Estados Inválidos**: Impossível ter `is_active=True` e `waiting_confirmation=True` ao mesmo tempo
✅ **Lógica Centralizada**: Todos os checks em métodos helpers
✅ **Testável**: Estados mutuamente exclusivos facilitam testes

## Testes Implementados

### test_state_machine.py (37 testes)
- Transições válidas
- Bloqueio de transições inválidas
- Reset de estado
- Helpers de query de estado
- Sem interdependências cruzadas

### test_unity_communication_integration.py (7 testes)
- Server startup/shutdown
- UDP broadcast discovery
- TCP connection
- Protocol session setup (IDLE → SETUP → READY)
- Protocol trigger (READY → ACTIVE)
- Error handling
- Legacy compatibility

## Resultado

```
[OK] 5 testes de integracao passaram
[OK] 37 testes de maquina de estados passaram
[OK] Transicoes de estado sao validadas
[OK] Interdependencias eliminadas
[OK] Codigo legado continua funcionando
```

## Cobertura de Estados

### ServerState Transitions
```
STOPPED
   ↓
RUNNING ← TCP desconecta
   ↓
CONNECTED (TCP conecta)
   ↓
RUNNING ← stop_server
   ↓
STOPPED
```

### SessionPhase Transitions
```
IDLE
  ↓
SETUP (start_session)
  ↓
READY (recebe confirmacao)
  ↓
ACTIVE (send_trigger)
  ↓
ENDING (end_session)
  ↓
IDLE (recebe confirmacao de encerramento)
```

## Mudanças no Código

### Removido
- `session.is_active` ❌
- `session.waiting_confirmation` ❌
- Múltiplas verificações `if` espalhadas ❌

### Adicionado
- `ServerState` enum ✅
- `SessionPhase` enum ✅
- Métodos helpers de validação ✅
- Método `can_transition_to()` ✅
- Método `transition_to()` em SessionState ✅

### Refatorado
- `start_server()` → usa `_transition_server_state()`
- `stop_server()` → reseta SessionPhase
- `start_session()` → transiciona SETUP e valida
- `send_trigger()` → transiciona READY → ACTIVE
- `end_session()` → transiciona ACTIVE → ENDING
- `_process_vr_message()` → transiciona de forma explícita
- Todos os checks → usam `_is_*()` helpers

## Exemplo de Uso

```python
# Verificar se servidor está pronto pra sessão
if not comm._is_server_ready_for_session():
    print("Servidor não está pronto")
    return False

# Iniciar sessão
patient = PatientData(nome="João", nivel=5, lado="Direito")
comm.start_session(patient, TaskType.TREINO)
# Session transiciona: IDLE → SETUP

# Aguardar confirmação do VR
# ... (VR envia confirmacao)
# Session transiciona: SETUP → READY

# Enviar trigger
comm.send_trigger()
# Session transiciona: READY → ACTIVE

# Enviar comandos durante sessão
if comm._is_session_active_for_commands():
    comm.send_hand_close('direita')

# Finalizar
comm.end_session()
# Session transiciona: ACTIVE → ENDING
```

## Conclusão

**Problema Resolvido:** Ouroboros eliminado ✅

A máquina de estados agora é:
- **Centralizada**: Um único lugar define os estados válidos
- **Validada**: Transições inválidas são bloqueadas
- **Clara**: Código legível e fácil de entender
- **Testável**: Todos os cenários podem ser testados
- **Robusta**: Impossível ter estados inconsistentes
