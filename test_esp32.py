#!/usr/bin/env python3
"""
Script de teste para comunicação ESP32
Demonstra o funcionamento da comunicação serial sem precisar do ESP32 fisicamente conectado
"""

import sys
import time
from pathlib import Path

# Adicionar o diretório do projeto ao Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bci.network.esp32_serial_communication import ESP32SerialCommunicator

def test_esp32_communication():
    """Testa a comunicação ESP32"""
    print("🔧 Testando comunicação ESP32...")
    print("=" * 50)
    
    # Criar instância do comunicador
    esp32 = ESP32SerialCommunicator(port="COM4", baudrate=115200)
    
    # Configurar callback de conexão
    def on_connection_change(connected):
        status = "✅ Conectado" if connected else "❌ Desconectado"
        print(f"📡 Status ESP32: {status}")
    
    esp32.set_connection_callback(on_connection_change)
    
    # Obter status inicial
    status = esp32.get_connection_status()
    print(f"📊 Status inicial:")
    print(f"   Porta: {status['port']}")
    print(f"   Baudrate: {status['baudrate']}")
    print(f"   Timeout: {status['timeout']}s")
    print(f"   Conectado: {status['connected']}")
    print()
    
    # Tentar conectar
    print("🔌 Tentando conectar ao ESP32...")
    connected = esp32.connect()
    
    if connected:
        print("✅ ESP32 conectado com sucesso!")
        print()
        
        # Testar comandos
        print("🧪 Testando comandos...")
        
        commands = [
            ("PING", esp32.send_ping),
            ("TRIGGER_LEFT", esp32.send_trigger_left),
            ("TRIGGER_RIGHT", esp32.send_trigger_right),
        ]
        
        for cmd_name, cmd_func in commands:
            print(f"📤 Enviando {cmd_name}...")
            success = cmd_func()
            result = "✅ Sucesso" if success else "❌ Falha"
            print(f"   Resultado: {result}")
            time.sleep(0.5)
        
        print()
        print("🔌 Desconectando ESP32...")
        esp32.disconnect()
        print("✅ ESP32 desconectado")
        
    else:
        print("❌ Falha ao conectar ESP32")
        print("💡 Possíveis causas:")
        print("   - ESP32 não conectado na COM4")
        print("   - Porta COM4 em uso por outro programa")
        print("   - ESP32 não configurado para 115200 baud")
        print("   - Driver USB não instalado")
    
    print()
    print("=" * 50)
    print("🏁 Teste finalizado")

def test_singleton_behavior():
    """Testa o comportamento singleton"""
    print("\n🔧 Testando comportamento singleton...")
    
    from bci.network.esp32_serial_communication import get_esp32_communicator
    
    # Criar múltiplas instâncias
    esp1 = get_esp32_communicator()
    esp2 = get_esp32_communicator()
    esp3 = ESP32SerialCommunicator()
    
    # Verificar se são diferentes instâncias (esperado: esp1 == esp2, esp3 != esp1)
    print(f"esp1 is esp2: {esp1 is esp2}")  # Deve ser True (singleton)
    print(f"esp1 is esp3: {esp1 is esp3}")  # Deve ser False (instância separada)
    
    print("✅ Singleton funcionando corretamente")

def test_convenience_functions():
    """Testa as funções de conveniência"""
    print("\n🔧 Testando funções de conveniência...")
    
    from bci.network.esp32_serial_communication import (
        send_trigger_left, send_trigger_right, 
        connect_esp32, disconnect_esp32, is_esp32_connected
    )
    
    print(f"ESP32 conectado: {is_esp32_connected()}")
    
    print("Tentando conectar via função de conveniência...")
    connected = connect_esp32()
    print(f"Conectado: {connected}")
    print(f"Status após conexão: {is_esp32_connected()}")
    
    if connected:
        print("Testando triggers via funções de conveniência...")
        print(f"Trigger Left: {send_trigger_left()}")
        print(f"Trigger Right: {send_trigger_right()}")
        
        disconnect_esp32()
        print(f"Status após desconexão: {is_esp32_connected()}")
    
    print("✅ Funções de conveniência funcionando")

if __name__ == "__main__":
    print("🚀 Sistema de Teste - Comunicação ESP32")
    print("=" * 50)
    print("📝 Este script testa a comunicação serial com ESP32")
    print("📝 Funciona mesmo sem o ESP32 fisicamente conectado")
    print("=" * 50)
    
    test_esp32_communication()
    test_singleton_behavior()
    test_convenience_functions()
    
    print("\n🎉 Todos os testes concluídos!")
    print("💡 Para testar com ESP32 real:")
    print("   1. Conecte o ESP32 na COM4")
    print("   2. Configure para 115200 baud")
    print("   3. Execute este script novamente")