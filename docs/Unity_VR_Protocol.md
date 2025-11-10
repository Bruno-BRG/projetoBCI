# Protocolo de Comunicação Sistema ↔ VR

Este documento descreve o protocolo de comunicação implementado entre o sistema BrainBridge e o ambiente VR Unity.

## 📋 Visão Geral

O protocolo define uma sequência de mensagens estruturada para estabelecer e gerenciar sessões de reabilitação em VR, incluindo:
- Descoberta e conexão inicial
- Configuração de sessão com dados do paciente
- Controle de execução de tarefas
- Comandos de interação durante a sessão
- Finalização controlada

## 🔄 Fluxo do Protocolo

```
Sistema                                          VR
   |                                              |
   |--- 1. Broadcast UDP (Header: "Confirm") --->|
   |<--- 2. Confirm (TCP Connection) ------------|
   |                                              |
   |--- 3. Dados Paciente -------------------->  |
   |      Nome: ...                              |
   |      Nivel: ...                             |
   |      Lado: ...                              |
   |                                              |
   |--- 4. Tarefa: "Treino" ou "Jogo" -------->  |
   |                                              |
   |--- 5. Trigger ---------------------------->  |
   |                                              |
   |--- 6. Comandos de Ação ------------------>  |
   |      ****_HAND_CLOSE                        |
   |      LEFT_FLOWER, RIGHT_FLOWER              |
   |                                              |
   |--- 7. Finalizar_tarefa_treino ------------>  |
   |    ou Finalizar_tarefa_jogo                 |
   |                                              |
   |<--- 8. Confirmar_finalização ---------------|
   |                                              |
```

## 🌐 Fase 1: Descoberta e Conexão

### Broadcast UDP (Sistema → VR)

O sistema continuamente envia broadcasts UDP para descoberta:

```
Porta: 12346
Formato: "Confirm:<ip1>,<ip2>,..."
Exemplo: "Confirm:192.168.1.100,10.0.0.5"
Intervalo: 1 segundo
```

### Conexão TCP (VR → Sistema)

O VR estabelece conexão TCP ao receber o broadcast:

```
Porta: 12345 (TCP Server no Sistema)
O VR envia confirmação: "Confirm"
```

## 📤 Fase 2: Inicialização de Sessão

### 3. Envio de Dados do Paciente

```python
Formato:
Dados Paciente:
Nome: <nome_paciente>
Nivel: <nivel_paciente>
Lado: <Esquerdo|Direito>

Exemplo:
Dados Paciente:
Nome: João Silva
Nivel: Intermediário
Lado: Direito
```

**API Python:**
```python
from brainbridge_v2.communication.unity import UnityCommunicator, PatientData, TaskType

communicator = UnityCommunicator()
communicator.start_server()

# Criar dados do paciente
patient = PatientData(
    nome="João Silva",
    nivel="Intermediário",
    lado="Direito"
)

# Iniciar sessão
task_type = TaskType.TREINO  # ou TaskType.JOGO
communicator.start_session(patient, task_type)
```

### 4. Envio de Tarefa

```python
Formato:
Tarefa:
"<Treino|Jogo>"

Exemplo:
Tarefa:
"Treino"
```

**Valores permitidos:**
- `"Treino"` - Sessão de treinamento
- `"Jogo"` - Sessão de jogo/aplicação

## 🎯 Fase 3: Execução da Tarefa

### 5. Trigger - Iniciar Tarefa

Após o VR confirmar recebimento dos dados:

```python
Comando: "Trigger"
```

**API Python:**
```python
# Enviar trigger para iniciar
communicator.send_trigger()
```

### 6. Comandos Durante a Sessão

#### Fechar Mão

```python
Comandos:
- "LEFT_HAND_CLOSE"  # Fechar mão esquerda
- "RIGHT_HAND_CLOSE" # Fechar mão direita
```

**API Python:**
```python
# Fechar mão direita
communicator.send_hand_close("direita")  # ou "right"

# Fechar mão esquerda
communicator.send_hand_close("esquerda")  # ou "left"
```

#### Ação de Flor

```python
Comandos:
- "LEFT_FLOWER"  # Ação flor esquerda
- "RIGHT_FLOWER" # Ação flor direita
```

**API Python:**
```python
# Flor direita
communicator.send_flower_action("direita")

# Flor esquerda
communicator.send_flower_action("esquerda")
```

## 🏁 Fase 4: Finalização

### 7. Finalizar Tarefa

```python
Comandos:
- "Finalizar_tarefa_treino" # Para sessões de treino
- "Finalizar_tarefa_jogo"   # Para sessões de jogo

Formato com mensagem opcional:
Finalizar_tarefa_treino
END_TASK, "<mensagem>"
```

**API Python:**
```python
# Finalizar sem mensagem
communicator.end_session()

# Finalizar com mensagem
communicator.end_session("Sessão concluída com sucesso")
```

### 8. Confirmação de Finalização (VR → Sistema)

O VR deve responder confirmando a finalização:

```
Resposta esperada: mensagem contendo "Confirm" ou "Finalizar"
```

## 🔌 Portas e Protocolos

| Serviço | Protocolo | Porta | Direção | Descrição |
|---------|-----------|-------|---------|-----------|
| Broadcast IP | UDP | 12346 | Sistema → VR | Descoberta de rede |
| Comando/Controle | TCP | 12345 | Bidirecional | Mensagens do protocolo |
| Publisher ZMQ | TCP | 5555 | Sistema → VR | Mensagens alternativas |

## 💻 Exemplo Completo de Uso

```python
from brainbridge_v2.communication.unity import (
    UnityCommunicator, 
    PatientData, 
    TaskType
)

# Criar e configurar comunicador
comm = UnityCommunicator()

# Callbacks opcionais
def on_vr_connected(connected):
    if connected:
        print("✅ VR conectado!")
    else:
        print("❌ VR desconectado")

def on_confirmation():
    print("✅ VR confirmou - pronto para trigger")

comm.set_connection_callback(on_vr_connected)
comm.set_confirmation_callback(on_confirmation)

# Iniciar servidor
comm.start_server()

# Aguardar VR conectar...
input("Pressione ENTER após VR conectar...")

# Configurar sessão
patient = PatientData(
    nome="Maria Silva",
    nivel="Avançado",
    lado="Esquerdo"
)

# Iniciar sessão de treino
comm.start_session(patient, TaskType.TREINO)

# Aguardar confirmação do VR
input("Pressione ENTER após confirmação do VR...")

# Enviar trigger para iniciar
comm.send_trigger()

# Durante a sessão...
input("Pressione ENTER para fechar mão direita...")
comm.send_hand_close("direita")

input("Pressione ENTER para ação flor esquerda...")
comm.send_flower_action("esquerda")

# Finalizar
input("Pressione ENTER para finalizar...")
comm.end_session("Treino completado")

# Parar servidor
comm.stop_server()
```

## 🔧 API de Compatibilidade Legada

Para compatibilidade com código existente, a classe `UDP_sender` foi atualizada:

```python
from brainbridge_v2.communication.unity import UDP_sender

# Iniciar sistema
UDP_sender.init_zmq_socket()

# Iniciar sessão VR (novo)
UDP_sender.start_vr_session(
    patient_name="João",
    level="Intermediário", 
    affected_side="Direito",
    task="Treino"
)

# Enviar comandos (métodos legados adaptados)
UDP_sender.enviar_sinal("trigger")
UDP_sender.enviar_sinal("direita")  # Usa send_hand_close
UDP_sender.enviar_sinal("esquerda")
UDP_sender.enviar_sinal("left_flower")

# Finalizar sessão
UDP_sender.end_vr_session("Sessão concluída")

# Parar sistema
UDP_sender.stop_zmq_socket()
```

## 📊 Estados da Sessão

O comunicador mantém estado interno via `SessionState`:

```python
@dataclass
class SessionState:
    patient: Optional[PatientData] = None      # Dados do paciente atual
    task_type: Optional[TaskType] = None       # Tipo de tarefa (Treino/Jogo)
    is_active: bool = False                    # Sessão em execução
    waiting_confirmation: bool = False         # Aguardando confirmação VR
```

## ⚠️ Tratamento de Erros

Todos os métodos retornam `bool` indicando sucesso:

```python
if not comm.send_hand_close("direita"):
    print("❌ Falha ao enviar comando")
    
if not comm.start_session(patient, task_type):
    print("❌ Falha ao iniciar sessão")
```

## 🧪 Teste Manual

Execute o módulo diretamente para interface de teste:

```bash
cd brainbridge_v2
python -m communication.unity
```

**Comandos disponíveis:**
```
iniciar <nome> <nivel> <lado> <tarefa>
trigger
fechar <lado>
flor <lado>
fim [mensagem]
status
sair
```

**Exemplo de sessão:**
```
> iniciar João Intermediário Direito Treino
> trigger
> fechar direita
> flor esquerda
> fim Sessão completada
> sair
```

## 📝 Notas Importantes

1. **Ordem de Chamadas**: Sempre seguir a sequência do protocolo:
   - `start_server()` → `start_session()` → aguardar confirmação → `send_trigger()` → comandos → `end_session()`

2. **Confirmações**: O sistema aguarda confirmações do VR em pontos-chave:
   - Após envio de dados/tarefa
   - Após comando de finalização

3. **Thread-Safety**: A classe usa singleton e locks internos - seguro para uso multi-thread

4. **Compatibilidade**: API legada (`UDP_sender`) continua funcionando e usa o novo protocolo internamente

## 🔍 Debugging

Para debug detalhado, observe os logs no console:

- `[UDP]` - Mensagens de broadcast
- `[TCP]` - Mensagens TCP do servidor
- `[ZMQ]` - Mensagens ZMQ publisher
- `✅`/`❌` - Status de operações
- `📤`/`📥` - Envio/recebimento de mensagens

## 📚 Referências

- Implementação: `brainbridge_v2/communication/unity.py`
- Diagrama de Sequência: Anexo no repositório
- Copilot Instructions: `.github/copilot-instructions.md`
