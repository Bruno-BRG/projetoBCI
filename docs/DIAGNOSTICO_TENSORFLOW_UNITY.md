# Diagnóstico: Desconexão TensorFlow -> Unity

## Problema Relatado
Após mudar o protocolo de comunicação, a IA do TensorFlow perdeu a conexão com o sistema Unity. 
- ✓ Modelos carregam normalmente
- ✗ Previsões não são recebidas no VR/Unity

## Análise da Arquitetura

### Fluxo Esperado de Previsão (Correto)
```
Dados EEG
    ↓
StreamingWidget.on_data_received()
    ↓
StreamingWidget.eeg_buffer (acumula 250 amostras)
    ↓
StreamingWidget.predict_movement(eeg_buffer)  ← Predição com TensorFlow
    ↓
Predictor.predict_window() OU TensorFlowMLAdapter.predict_on_window()
    ↓
Resultado: {'label': 'left'|'right', 'probs': [...]}
    ↓
Mapear para comando: LEFT_HAND_CLOSE ou RIGHT_HAND_CLOSE
    ↓
StreamingWidget.send_udp_signal(direction)
    ↓
UDP_sender.enviar_sinal(direction)
    ↓
UnityCommunicator.send_hand_command(direction)
    ↓
UnityCommunicator.send_command(command_str)  ← Envia para Unity
    ↓
ZMQ + TCP para VR
```

### Pontos Potenciais de Falha Identificados

#### 1. **Previsões Não Sendo Feitas (predict_movement não chamado)**
   - `game_mode` não está ativado
   - `ai_prediction_enabled` está False
   - Janela de tempo (`ai_window_duration`) expirou
   - Modelo não foi carregado

#### 2. **Previsões Feitas mas Não Enviadas ao Unity**
   - `send_udp_signal()` em `predict_movement()` não é alcançado
   - `send_udp_signal()` retorna False (servidor UDP inativo)
   - `UDP_sender.enviar_sinal()` não mapeia corretamente a ação

#### 3. **Problema no Novo Protocolo**
   - `UnityCommunicator.send_hand_command()` não envia o comando correto
   - `send_command()` não está enviando via ZMQ/TCP
   - Conexão TCP não está estabelecida

## Checklist de Verificação

### ✓ Checklist do Carregamento do Modelo
```python
# Verificar que o modelo carrega
from ml.tensorflow_adapter import TensorFlowMLAdapter
adapter = TensorFlowMLAdapter()
adapter.load_model("path/to/model.keras")
# Se não lançar exceção, está OK
```

### ✓ Checklist da Predição
```python
# Verificar que predição funciona
import numpy as np
window = np.random.randn(250, 16)  # 250 timesteps, 16 channels
result = adapter.predict_on_window(window)
print(result)
# Output: {'probs': [0.3, 0.7], 'label': 'right'}
```

### ✓ Checklist do Envio para Unity
```python
# Verificar que comando vai para Unity
from communication.unity import UnityCommunicator
comm = UnityCommunicator()
comm.start_server()  # Inicia ZMQ e TCP
comm.send_hand_command('direita')  # Deve enviar RIGHT_HAND_CLOSE
```

### ✓ Checklist da Integração StreamingWidget
```python
# Verificar flags importantes durante o jogo:
# 1. self.game_mode == True
# 2. self.model is not None
# 3. self.udp_server_active == True
# 4. self.ai_prediction_enabled == True
# 5. self.task_start_time is not None
```

## Testes Criados

### 1. `test_tensorflow_unity_integration.py`
Suite completa com 20+ testes cobrindo:
- Carregamento de modelos TensorFlow
- Predictor.predict_window()
- Comandos ActionCommand
- Envio de dados para Unity
- Integração end-to-end
- Tratamento de erros
- Debouncing de comandos

**Executar:**
```bash
cd brainbridge_v2
python -m pytest tests/test_tensorflow_unity_integration.py -v
```

### 2. `diagnostic_tensorflow_unity.py`
Script diagnóstico que executa 8 checks sequenciais:
1. TensorFlow disponível?
2. Modelos Keras disponíveis?
3. TensorFlowMLAdapter funcionando?
4. Predictor funcionando?
5. UnityCommunicator pronto?
6. Fluxo predição -> comando?
7. UDP_sender compatível?
8. StreamingWidget integrado?

**Executar:**
```bash
cd brainbridge_v2/tests
python diagnostic_tensorflow_unity.py
```

## Debugging Passo-a-Passo

### Etapa 1: Verificar Logs do StreamingWidget
Ao iniciar o jogo, você deve ver logs como:
```
[SEND_COMMAND] Preparando para enviar (len=16): RIGHT_HAND_CLOSE
[ZMQ] Comando enviado: RIGHT_HAND_CLOSE
[TCP] Comando enviado: RIGHT_HAND_CLOSE
```

Se não ver, problema está em `send_udp_signal()` ou antes.

### Etapa 2: Verificar se game_mode está ativo
Adicionar print em `predict_movement()`:
```python
def predict_movement(self, eeg_data):
    print(f"[DEBUG] game_mode={self.game_mode}, model={self.model is not None}, ai_enabled={self.ai_prediction_enabled}")
    if not self.game_mode or self.model is None:
        return
```

### Etapa 3: Verificar se predição acontece
Adicionar print:
```python
# No final de predict_movement(), antes de send_udp_signal()
print(f"[PRED] label={pred_result['label']}, direction={'esquerda' if pred_result['label']=='left' else 'direita'}")
print(f"[PRED] Chamando send_udp_signal({direction})")
```

### Etapa 4: Verificar se send_udp_signal funciona
```python
def send_udp_signal(self, direction):
    print(f"[UDP] send_udp_signal({direction})")
    print(f"[UDP] server_active={self.udp_server_active}, auto_send={self.udp_auto_send_checkbox.isChecked()}")
    if self.udp_server_active and self.udp_auto_send_checkbox.isChecked():
        success = UDP_sender.enviar_sinal(direction)
        print(f"[UDP] enviar_sinal result: {success}")
        return success
    return False
```

## Causas Mais Comuns

### Causa 1: Servidor UDP não está ativo
```python
# Solução: Garantir que o servidor UDP foi iniciado
if not self.udp_server_active:
    self.unity_communicator.start_server()
```

### Causa 2: Checkbox "Auto-Send" não está marcado
```python
# Verificar em streaming.py se:
# - udp_auto_send_checkbox.isChecked() retorna True
# - esp32_auto_send_checkbox.isChecked() retorna True (se aplicável)
```

### Causa 3: send_udp_signal() não chama send_hand_command()
**Verificação:**
```python
# Em UDP_sender.enviar_sinal():
if action.lower() == 'direita':
    return cls._communicator.send_hand_command('direita')  # ← DEVE EXISTIR
```

### Causa 4: UnityCommunicator não tem conexão
```python
# Verificar se tcp_connected e zmq_socket estão ativos
if self.unity_communicator.tcp_connected and self.unity_communicator.zmq_socket:
    print("✓ Comunicador conectado e pronto")
else:
    print("✗ Comunicador não conectado")
```

### Causa 5: Janela de predição de IA expirou
```python
# ai_window_duration padrão é 2000ms (2 segundos)
# Se tarefa dura mais de 2s, janela fecha
# Solução: aumentar ou resetar a janela
self.ai_window_duration = 5000  # 5 segundos
```

## Métricas a Monitorar

1. **Número de predições**: `len(self.predictions)`
2. **Última predição**: `self.predictions[-1] if self.predictions else None`
3. **Tempo de predição**: armazenar timestamp e calcular latência
4. **Taxa de sucesso de envio**: contar quantas predições chegaram ao VR

## Exemplo de Fix Rápido

Se o problema é que `send_udp_signal()` não está sendo chamado:

```python
# Em predict_movement(), logo após calcular a predição:
print(f"[DEBUG] Predição: {pred_result['label']}")

# Garantir que direction está correto
if pred_result['label'] == 'left':
    direction = 'esquerda'
    print(f"[SEND] Enviando comando: esquerda (LEFT_HAND_CLOSE)")
else:
    direction = 'direita'
    print(f"[SEND] Enviando comando: direita (RIGHT_HAND_CLOSE)")

# Chamar send_udp_signal DEPOIS de atualizar a interface
self.send_udp_signal(direction)  # ← Garantir que está sendo chamado
self.send_esp32_signal(direction)  # ← E também ESP32 se aplicável

# Lock para evitar predições duplicadas
self.prediction_locked = True
```

## Próximos Passos

1. **Executar diagnóstico:**
   ```bash
   python brainbridge_v2/tests/diagnostic_tensorflow_unity.py
   ```

2. **Executar testes:**
   ```bash
   python -m pytest brainbridge_v2/tests/test_tensorflow_unity_integration.py -v
   ```

3. **Verificar logs em streaming.py:**
   - Procure por `[SEND_COMMAND]`, `[ZMQ]`, `[TCP]` quando fazer uma predição
   - Se não aparecer, problema está antes (predict_movement não chamado)
   - Se aparecer mas VR não responde, problema está na conexão

4. **Validar protocolo Unity:**
   - Testar com o script `tools/unity_device_simulator.py` se existir
   - Testar com VR real para confirmar comunicação

5. **Monitorar acurácia:**
   - Verificar se `self.accuracy_label` está se atualizando
   - Se não atualizar, mensagens não estão voltando do VR

## Referências

- [UnityCommunicator Protocol](./communication/unity.py#L1)
- [StreamingWidget.predict_movement()](./gui/widgets/streaming.py#L1515)
- [UDP_sender Compatibility](./communication/unity.py#L871)
- [Predictor Class](./ml/predictor.py)
- [TensorFlow Adapter](./ml/tensorflow_adapter.py)
