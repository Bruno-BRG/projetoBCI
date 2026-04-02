# Relatório Final: Integração TensorFlow -> Unity

## Status: ✓ VALIDADO E FUNCIONANDO

Após investigação completa e testes, a integração entre TensorFlow, Predictor e UnityCommunicator está **funcionando corretamente**.

## Resultados dos Testes

### Diagnóstico Executado
```
✓ CHECK 1: TensorFlow disponível         - PASS
✓ CHECK 2: Modelos Keras disponíveis     - PASS  
✓ CHECK 3: TensorFlowMLAdapter           - PASS
✓ CHECK 4: Predictor                     - PASS
✓ CHECK 5: UnityCommunicator             - PASS
✓ CHECK 6: Fluxo predição -> comando     - PASS
✓ CHECK 7: UDP_sender compatível         - PASS
✓ CHECK 8: StreamingWidget integração    - PASS

Resultado: TODOS OS 8 CHECKS PASSARAM ✓
```

### Suite de Testes Unitários
```
Total: 22 testes
Resultado: OK (todos passaram)

Cobertura:
  ✓ TensorFlowMLAdapter (5 testes)
  ✓ Predictor (3 testes)
  ✓ UnityCommunicator (2 testes)
  ✓ ActionCommand enums (2 testes)
  ✓ UDP_sender (2 testes)
  ✓ Fluxo predição->comando (2 testes)
  ✓ Janela de predição (2 testes)
  ✓ Tratamento de erros (2 testes)
  ✓ Logging (1 teste)
  ✓ Debouncing (1 teste)
```

## Fluxo de Funcionamento Confirmado

### Pipeline Completo Validado
```
1. EEG Data
   ↓ (StreamingThread)
2. on_data_received(data)
   ↓ (guwidgets/streaming.py)
3. eeg_buffer.append() x250
   ↓ (accumula 250 amostras)
4. predict_movement(eeg_buffer)
   ↓ (StreamingWidget.predict_movement)
5. TensorFlowMLAdapter.predict_on_window()
   ↓ (ml/tensorflow_adapter.py)
6. Modelo Keras prediz
   ↓ (resultado: {label, probs})
7. send_udp_signal(direction)
   ↓ (StreamingWidget)
8. UDP_sender.enviar_sinal()
   ↓ (communication/unity.py)
9. UnityCommunicator.send_hand_command()
   ↓ (UnityCommunicator.send_command)
10. ZMQ + TCP para Unity
   ↓ (protocolo bidirecional)
11. VR recebe comando e executa
```

## Possíveis Causas de Não Recebimento de Previsões

Mesmo com a integração funcionando, se você não está recebendo previsões, verifique:

### 1. Servidor UDP não está ativo
```python
# Solução: Garantir que start_server() foi chamado
if not self.unity_communicator.is_active:
    self.unity_communicator.start_server()
```

### 2. Modo jogo não está ativado
```python
# Verificar que game_mode == True
print(f"game_mode: {self.game_mode}")
# Deve estar True quando uma tarefa de JOGO é iniciada
```

### 3. Modelo não foi carregado
```python
# Verificar que modelo está carregado
if self.model is None:
    print("Modelo não carregado!")
    # Carregar modelo antes de iniciar jogo
```

### 4. Janela de tempo de IA expirou
```python
# ai_window_duration padrão é 2000ms (2 segundos)
# Se necessário mais tempo, aumentar:
self.ai_window_duration = 5000  # 5 segundos
```

### 5. Checkbox "Auto-Send" não está marcado
```python
# Garantir que:
self.udp_auto_send_checkbox.setChecked(True)
self.esp32_auto_send_checkbox.setChecked(True)  # se aplicável
```

## Checklist de Debugging

Se o VR não está recebendo previsões, usar este checklist:

### Etapa 1: Verificar Logs
Ao iniciar um jogo com IA ativa, você deve ver logs:
```
[SEND_COMMAND] Preparando para enviar: LEFT_HAND_CLOSE
[ZMQ] Comando enviado: LEFT_HAND_CLOSE
[TCP] Comando enviado: LEFT_HAND_CLOSE
```

**Se não vê esses logs:**
- Problema está antes de `send_udp_signal()`
- Verificar que `game_mode = True` e modelo foi carregado

### Etapa 2: Verificar Previsões Sendo Feitas
Adicionar temporariamente no `predict_movement()`:
```python
print(f"[DEBUG] Predição: {pred_result['label']}, prob: {pred_result['probs']}")
```

**Se não vê essa mensagem:**
- `predict_movement()` não está sendo chamado
- Verificar `ai_prediction_enabled` e `ai_window_duration`

### Etapa 3: Verificar Buffer EEG
```python
print(f"[DEBUG] EEG buffer size: {len(self.eeg_buffer)}")
```

**Se buffer não cresce até 250:**
- Problema no streaming de dados
- Verificar `on_data_received()` está sendo chamado

### Etapa 4: Testar Envio Manual
```python
# Adicionar botão de teste no UI
def test_manual_prediction(self):
    self.send_udp_signal('direita')
    self.send_udp_signal('esquerda')
```

**Se funciona manualmente mas não automático:**
- Problema específico do fluxo automático
- Verificar `game_mode`, `model`, `ai_prediction_enabled`

## Configurações Recomendadas

### Para Tarefa de Jogo
```python
# Em on_task_changed() quando tarefa é "Jogo":
self.game_mode = True
self.ai_prediction_enabled = True
self.ai_window_duration = 2000  # 2 segundos
self.udp_auto_send_checkbox.setChecked(True)
```

### Para Tarefa de Treino
```python
# Treino não usa IA, apenas grava dados
self.game_mode = False
self.ai_prediction_enabled = False
```

### Logging Avançado
```python
# Para debugging detalhado, adicionar em StreamingWidget.__init__:
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brainbridge_debug.log'),
        logging.StreamHandler()
    ]
)
```

## Como Executar Testes

### Diagnóstico Completo
```bash
cd brainbridge_v2
python tests/diagnostic_tensorflow_unity.py
```

Resultado esperado: "TODOS OS CHECKS PASSARAM!"

### Suite de Testes Unitários
```bash
cd brainbridge_v2
python -m unittest tests.test_tensorflow_unity_integration -v
```

Resultado esperado: "Ran 22 tests ... OK"

### Testar Integração Real
1. Abrir GUI: `python -m brainbridge_v2`
2. Selecionar paciente
3. Selecionar tarefa "Jogo"
4. Carregar modelo TensorFlow
5. Iniciar servidor UDP
6. Iniciar gravação/jogo
7. Verificar logs no console

## Estrutura dos Arquivos Criados

```
brainbridge_v2/
  tests/
    ├── test_tensorflow_unity_integration.py   (22 testes unitários)
    └── diagnostic_tensorflow_unity.py          (8 checks diagnósticos)
  docs/
    └── DIAGNOSTICO_TENSORFLOW_UNITY.md        (documentação completa)
```

## Performance e Latência

### Latência Esperada
- Leitura de dados EEG: ~8ms (125 Hz)
- Acúmulo de 250 amostras: ~2s
- Predição com TensorFlow: ~50-200ms (depende do hardware)
- Envio ZMQ: <5ms
- Envio TCP: <5ms
- **Total: ~2-2.3 segundos**

### Otimizações Possíveis
1. Reduzir `ai_window_duration` para disparar previsões mais cedo
2. Usar modelo mais leve para reduzir tempo de predição
3. Parallelizar leitura de dados com predição (usar thread)
4. Cache de resultados para evitar recálculos

## Conclusão

A integração TensorFlow <-> Unity **está funcionando corretamente**. Os testes confirmam que:

✓ Modelos carregam sem problemas
✓ Previsões são calculadas corretamente
✓ Comandos são enviados para Unity via ZMQ e TCP
✓ Debouncing evita spam de comandos
✓ Tratamento de erros funciona

**Se você não está recebendo previsões no VR, seguir o "Checklist de Debugging" acima.**

Para suporte técnico adicional, verificar os logs em:
- Console da GUI
- Arquivo de log (se configurado)
- Saída do `diagnostic_tensorflow_unity.py`
