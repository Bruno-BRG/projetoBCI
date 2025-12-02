# 🎮 Novo Protocolo de Comunicação Sistema ↔️ VR Unity

## 📊 Diagrama do Protocolo

```
sequenceDiagram
    autonumber
    participant Sistema
    participant VR

    Sistema->>VR: Broadcast UDP
    VR-->>Sistema: Header: "Confirm"

    Sistema->>VR: Dados Paciente:\nNome: ...\nNível: ...\nLado: ...
    Sistema->>VR: Tarefa:\n"Treino"\n"Jogo"

    Sistema->>VR: Trigger
    Sistema->>VR: ****_HAND_CLOSE

    VR-->>Sistema: LEFT_FLOWER, RIGHT_FLOWER

    Sistema->>VR: Finalizar tarefa_treino\nEND_TASK
    Sistema->>VR: Finalizar tarefa_jogo\nEND_TASK, "Mensagem"

    VR-->>Sistema: Confirmar_finalização
```

---

## ✨ O que foi implementado

### 1. **Novos Enums e Classes de Dados**

#### `TaskType` (enum)
```python
class TaskType(Enum):
    TREINO = "Treino"
    JOGO = "Jogo"
```

#### `PatientData` (dataclass)
Representa dados do paciente a ser enviado ao VR:
- `nome`: Nome do paciente
- `nivel`: Nível de 0 a 11 (validado)
- `lado`: "Direito" ou "Esquerdo" (validado)
- `format_message()`: Formata dados para envio

#### `ActionCommand` (enum)
Comandos de ação:
- `LEFT_HAND_CLOSE` / `RIGHT_HAND_CLOSE`: Fechar mão
- `LEFT_FLOWER` / `RIGHT_FLOWER`: Ação de flor

#### `EndTaskCommand` (enum)
Comandos de finalização:
- `END_TRAINING`: "Finalizar_tarefa_treino"
- `END_GAME`: "Finalizar_tarefa_jogo"

#### `SessionState` (dataclass + máquina de estados)
Mantém estado da sessão:
- `patient`: Dados do paciente atual
- `task_type`: Tipo de tarefa
- `is_active`: Se sessão está ativa
- `waiting_confirmation`: Se aguardando confirmação do VR
- `phase`: Fase da máquina de estados (IDLE → SETUP → READY → ACTIVE → ENDING → IDLE)

### 2. **Novos Métodos em UnityCommunicator**

#### Session Management
```python
def start_session(self, patient: PatientData, task_type: TaskType) -> bool:
    """Inicia sessão: envia dados do paciente e tipo de tarefa"""
    
def send_trigger(self) -> bool:
    """Envia trigger para iniciar tarefa + comando de fechar mão"""
    
def end_task(self, message: str = "") -> bool:
    """Finaliza tarefa com mensagem opcional"""
```

#### Actions During Session
```python
def send_hand_close(self, lado: str) -> bool:
    """Envia comando de fechar mão (direita/esquerda)"""
    
def send_flower_action(self, lado: str) -> bool:
    """Envia comando de ação de flor"""
```

#### Callbacks para Eventos
```python
def set_confirmation_callback(self, callback: Callable[[], None]):
    """Callback quando VR confirma recebimento"""
    
def set_flower_callback(self, callback: Callable[[ActionCommand], None]):
    """Callback quando VR envia ação de flor"""
```

#### Helpers
```python
def _is_session_active_for_commands(self) -> bool:
    """Valida se sessão está pronta para receber comandos"""
    
def _process_vr_message(self, message: str):
    """Processa mensagens recebidas do VR"""
```

### 3. **Transições Automáticas de Mensagens do VR**

O método `_handle_tcp_connection` agora reconhece:
- ✅ `"confirm"` → dispara `on_confirmation` callback
- ✅ `"left_flower"` → dispara `on_flower_action` com LEFT_FLOWER
- ✅ `"right_flower"` → dispara `on_flower_action` com RIGHT_FLOWER
- ✅ `"confirmar_finalização"` → marca `is_active = False`

---

## 🔧 Compatibilidade Mantida

### ✅ Não há breaking changes!

1. **Classe `UDP_sender` ainda funciona** (compatibilidade legada)
2. **Métodos antigos preservados**:
   - `send_command()`
   - `send_hand_command()`
   - `send_trigger_command()`
3. **Novos métodos adicionados** (não remove os antigos)
4. **Máquina de estados** (SessionPhase, ServerState) adicionada para testes

---

## 📝 Exemplo de Uso

### Exemplo 1: Sessão Completa de Treino

```python
from brainbridge_v2.communication.unity import (
    UnityCommunicator,
    PatientData,
    TaskType,
    ActionCommand
)
import time

# 1. Criar comunicador
comm = UnityCommunicator()

# 2. Configurar callbacks
def on_connection(connected):
    print(f"VR {'conectado' if connected else 'desconectado'}")

def on_flower(action: ActionCommand):
    print(f"VR acionou: {action.value}")

comm.set_connection_callback(on_connection)
comm.set_flower_callback(on_flower)

# 3. Iniciar servidor
comm.start_server()

# 4. Aguardar conexão do VR
while not comm.tcp_connected:
    time.sleep(1)

# 5. Criar dados do paciente
patient = PatientData(
    nome="João Silva",
    nivel=5,
    lado="Direito"
)

# 6. Iniciar sessão
comm.start_session(patient, TaskType.TREINO)

# 7. Aguardar confirmação e enviar trigger
time.sleep(2)
comm.send_trigger()

# 8. Executar comandos durante sessão
time.sleep(1)
comm.send_hand_close("direita")
time.sleep(2)
comm.send_hand_close("esquerda")

# 9. Finalizar
comm.end_task("Treino concluído com sucesso!")

# 10. Parar servidor
comm.stop_server()
```

### Exemplo 2: Sessão Interativa

```python
# Ver em: brainbridge_v2/communication/example_protocol.py
# Função: exemplo_sessao_interativa()

from brainbridge_v2.communication.example_protocol import exemplo_sessao_interativa
exemplo_sessao_interativa()
```

---

## 🧪 Testes

Todos os testes passam com sucesso! ✅

```bash
# Rodar testes do protocolo
pytest brainbridge_v2/tests/test_unity_protocol.py -v

# Resultado:
# 23 passed in 0.07s ✅
```

Cobertura:
- ✅ Validação de PatientData (níveis 0-11, lado válido)
- ✅ TaskType enum
- ✅ ActionCommand enum
- ✅ EndTaskCommand enum
- ✅ SessionState com transições
- ✅ UnityCommunicator (singleton, server lifecycle)
- ✅ Protocol flow validation
- ✅ Backward compatibility (UDP_sender)

---

## 📦 Estrutura de Pastas

```
brainbridge_v2/
├── communication/
│   ├── __init__.py          (exporta novos tipos)
│   ├── unity.py             (implementação do protocolo)
│   └── example_protocol.py  (exemplos de uso)
├── tests/
│   └── test_unity_protocol.py  (23 testes ✅)
```

---

## 🎯 Principais Características

| Recurso | Status |
|---------|--------|
| Broadcast UDP com "Confirm" | ✅ Já existia |
| Enviar Dados Paciente | ✅ Implementado |
| Enviar Tipo de Tarefa | ✅ Implementado |
| Enviar Trigger + HAND_CLOSE | ✅ Implementado |
| Receber FLOWER responses | ✅ Implementado |
| Enviar END_TASK | ✅ Implementado |
| Confirmar finalização | ✅ Implementado |
| Validações robustas | ✅ PatientData valida dados |
| Máquina de estados | ✅ SessionState com transições |
| Backward compatibility | ✅ Nenhum breaking change |
| Testes unitários | ✅ 23 testes passando |

---

## 🚀 Fluxo Típico de Uma Sessão

```
┌─────────────────────────────────────────────────────────────┐
│                    SESSÃO VR NO SISTEMA                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1️⃣  UnityCommunicator.start_server()                     │
│      ├─ Inicia broadcast UDP em port 12346                │
│      ├─ Aguarda conexão TCP em port 12345                 │
│      └─ ZMQ publisher ativo em port 5555                  │
│                                                             │
│  2️⃣  VR conecta via TCP                                    │
│      └─ Callback: on_connection_changed(True)             │
│                                                             │
│  3️⃣  UnityCommunicator.start_session(patient, TaskType)  │
│      ├─ Envia: "Dados Paciente:\nNome: ...\nNível: ..."  │
│      └─ Envia: "Tarefa:\nTreino" (ou "Jogo")             │
│                                                             │
│  4️⃣  VR responde com "Confirm"                            │
│      └─ Callback: on_confirmation()                       │
│                                                             │
│  5️⃣  UnityCommunicator.send_trigger()                     │
│      ├─ Envia: "Trigger"                                   │
│      └─ Envia: "LEFT_HAND_CLOSE" (ou RIGHT)              │
│                                                             │
│  6️⃣  Sessão ativa - enviar comandos:                      │
│      ├─ send_hand_close("direita")                        │
│      └─ send_flower_action("esquerda")                    │
│                                                             │
│  7️⃣  VR responde com ações:                               │
│      ├─ "LEFT_FLOWER" → Callback: on_flower_action()    │
│      └─ "RIGHT_FLOWER" → Callback: on_flower_action()   │
│                                                             │
│  8️⃣  UnityCommunicator.end_task(message)                 │
│      ├─ Envia: "Finalizar_tarefa_treino"                  │
│      └─ Envia: mensagem opcional                          │
│                                                             │
│  9️⃣  VR confirma: "Confirmar_finalização"                 │
│      └─ Sessão marcada como finalizada                    │
│                                                             │
│  🔟 UnityCommunicator.stop_server()                       │
│      └─ Limpa recursos, encerra threads                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚡ Performance e Resiliência

- **Debounce**: `UDP_sender.enviar_sinal()` evita envios duplicados (200ms)
- **Timeouts**: Conexão TCP com timeout de 1.0s para detectar desconexões
- **Graceful degradation**: Se TCP falha, tenta ZMQ; se ZMQ falha, tenta TCP
- **Thread-safe**: Singleton thread-safe com lock
- **Cleanup**: Recursos limpos automaticamente ao parar servidor

---

## 📚 Referências

- Arquivo de exemplo: `brainbridge_v2/communication/example_protocol.py`
- Testes: `brainbridge_v2/tests/test_unity_protocol.py`
- Documentação antigo: `docs/STATE_MACHINE_SUMMARY.md`

---

## 🎓 Notas Importantes

### Mínimas Mudanças Aplicadas ✅

- ✅ Apenas **adicionado** novos métodos (nenhum removido)
- ✅ Apenas **adicionados** novos tipos (nenhum modificado)
- ✅ **Compatibilidade backward** com UDP_sender mantida 100%
- ✅ **Sem breaking changes** em código existente
- ✅ **Testes antigos** continuam passando

### O que Não Mudou

- Sistema de broadcast UDP
- Conexão TCP básica
- Classe UDP_sender
- Métodos send_command, send_hand_command, send_trigger_command
- Callbacks genéricos

---

## 🔐 Validações Robustas

```python
# PatientData valida automaticamente
patient = PatientData("João", 5, "Direito")  # ✅ OK

patient = PatientData("João", 15, "Direito")  # ❌ ValueError: nível fora de range

patient = PatientData("João", 5, "Centro")   # ❌ ValueError: lado inválido
```

---

## 📞 Suporte e Debugging

```python
# Ver logs detalhados:
comm = UnityCommunicator()
comm.start_server()  # Mostra: [UDP], [TCP], [ZMQ] logs

# Monitorar mensagens:
def debug_message(msg):
    print(f"DEBUG: {msg}")

comm.set_message_callback(debug_message)

# Monitorar conexão:
def debug_connection(connected):
    status = "✅ CONECTADO" if connected else "❌ DESCONECTADO"
    print(f"STATUS VR: {status}")

comm.set_connection_callback(debug_connection)
```

---

**Implementação Concluída!** 🎉
Data: 12 de Novembro de 2025
Status: ✅ 23/23 testes passando
Breaking Changes: ❌ Nenhum
