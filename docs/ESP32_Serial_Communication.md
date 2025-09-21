# Comunicação Serial ESP32 - Documentação

## Visão Geral

O sistema BCI agora suporta comunicação serial com ESP32 além da comunicação UDP/TCP existente. Isso permite enviar comandos TRIGGER diretamente para um ESP32 conectado via porta serial, oferecendo maior flexibilidade no controle de dispositivos externos.

## Funcionalidades Implementadas

### ✅ Comunicação Serial ESP32
- **Porta**: COM4 (configurável)
- **Baud Rate**: 115200 (configurável)
- **Comandos suportados**:
  - `TRIGGER_LEFT`: Trigger para mão esquerda
  - `TRIGGER_RIGHT`: Trigger para mão direita
  - `PING`: Teste de conexão

### ✅ Interface de Usuario
- **Checkbox**: "Envio Serial Automático" - controla se comandos são enviados via serial
- **Botões de teste**: Teste manual de triggers esquerdo e direito
- **Status de conexão**: Mostra se ESP32 está conectado
- **Botão conectar/desconectar**: Controle manual da conexão ESP32

### ✅ Integração com Sistema Existente
- **Compatibilidade total**: Sistema UDP/TCP continua funcionando normalmente
- **Envio duplo**: Quando habilitado, triggers são enviados tanto via UDP quanto serial
- **Controle independente**: UDP e Serial podem ser habilitados/desabilitados independentemente

## Como Usar

### 1. Preparação do ESP32

1. Conecte o ESP32 ao computador via USB
2. Carregue o código `esp32_trigger_receiver.ino` no ESP32 usando Arduino IDE
3. Configure as conexões de hardware conforme necessário
4. Certifique-se de que o ESP32 está na porta COM4

### 2. Usando na Interface BCI

1. **Conectar ESP32**:
   - Clique no botão "Conectar ESP32"
   - Verifique se o status muda para "ESP32: Conectado (COM4)"

2. **Habilitar Envio Automático**:
   - Marque o checkbox "Envio Serial Automático"
   - Agora todos os triggers serão enviados automaticamente para o ESP32

3. **Teste Manual**:
   - Use os botões "🤚 Trigger Esquerdo" e "✋ Trigger Direito"
   - Verifique se o ESP32 recebe os comandos (monitor serial)

### 3. Durante Operação Normal

- **Modo Treino/Teste/Jogo**: 
  - Com checkbox marcado, triggers são enviados automaticamente
  - Funciona em paralelo com UDP/TCP
  
- **Predições de IA**:
  - Triggers da IA também são enviados via serial quando habilitado
  - Comportamento idêntico ao UDP

## Arquivos Modificados/Criados

### Criados:
- `bci/network/esp32_serial_communication.py` - Módulo de comunicação serial
- `test_esp32.py` - Script de teste da comunicação
- `esp32_trigger_receiver.ino` - Código para ESP32

### Modificados:
- `bci/ui/streaming_widget.py` - Interface com controles ESP32
- `requirements.txt` - Adicionado pyserial

## Dependências

- **pyserial**: Para comunicação serial
  ```bash
  pip install pyserial
  ```

## Configuração Técnica

### ESP32 (Hardware)
```
- Porta: COM4
- Baud Rate: 115200
- Pinos de trigger:
  - GPIO2: Trigger esquerdo
  - GPIO4: Trigger direito
```

### Python (Software)
```python
# Configuração padrão
port = "COM4"
baudrate = 115200
timeout = 1.0
```

## Troubleshooting

### ESP32 não conecta
1. Verifique se está na COM4
2. Certifique-se de que nenhum outro programa está usando a porta
3. Verifique se o driver USB está instalado
4. Confirme que o código foi carregado corretamente no ESP32

### Triggers não são enviados
1. Verifique se o checkbox "Envio Serial Automático" está marcado
2. Confirme que o ESP32 está conectado
3. Verifique o monitor serial do ESP32 para debug

### Problemas de comunicação
1. Tente desconectar e reconectar o ESP32
2. Reinicie o ESP32
3. Verifique se o baud rate está correto (115200)

## Exemplo de Uso

```python
# Importar módulo
from bci.network.esp32_serial_communication import get_esp32_communicator

# Obter instância
esp32 = get_esp32_communicator()

# Conectar
if esp32.connect():
    print("ESP32 conectado!")
    
    # Enviar triggers
    esp32.send_trigger_left()   # Trigger esquerdo
    esp32.send_trigger_right()  # Trigger direito
    
    # Desconectar
    esp32.disconnect()
```

## Integração com Unity/Outros Sistemas

O sistema agora oferece duas opções de comunicação:

1. **UDP/TCP** (existente):
   - Para comunicação com Unity
   - Rede local
   - Maior flexibilidade de localização

2. **Serial ESP32** (novo):
   - Para controle direto de hardware
   - Conexão física via USB
   - Menor latência
   - Mais confiável para triggers críticos

Ambos podem funcionar simultaneamente, oferecendo máxima flexibilidade.

## Status do Projeto

✅ **Concluído**: 
- Implementação completa da comunicação serial
- Interface de usuário integrada
- Testes básicos funcionando
- Documentação criada

⚠️ **Próximos Passos**:
- Teste com ESP32 físico conectado
- Ajustes finos na interface se necessário
- Otimizações de performance se necessário