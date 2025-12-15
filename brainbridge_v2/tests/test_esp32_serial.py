"""
Script de teste para diagnóstico da comunicação ESP32
Valida conexão serial e envio de dados
"""

import sys
import time
from pathlib import Path

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from communication.esp32 import ESP32SerialCommunicator, get_esp32_communicator

def test_esp32_direct():
    """Testa instância direta"""
    print("\n" + "="*60)
    print("TESTE 1: Instância Direta do ESP32SerialCommunicator")
    print("="*60)
    
    esp32 = ESP32SerialCommunicator(port="COM3", baudrate=115200)
    
    # Mostrar status inicial
    print(f"\nStatus inicial:")
    print(f"  Porta: {esp32.port}")
    print(f"  Baudrate: {esp32.baudrate}")
    print(f"  Conectado: {esp32.is_connected}")
    
    # Tentar conectar
    print(f"\nConectando em COM3...")
    if esp32.connect():
        print("✓ Conectado com sucesso!")
        
        # Enviar teste
        print(f"\nEnviando PING...")
        result = esp32.send_ping()
        print(f"  Resultado: {'✓ Sucesso' if result else '✗ Falha'}")
        
        time.sleep(1)
        
        print(f"\nEnviando trigger esquerdo...")
        result = esp32.send_trigger_left()
        print(f"  Resultado: {'✓ Sucesso' if result else '✗ Falha'}")
        
        time.sleep(1)
        
        print(f"\nEnviando trigger direito...")
        result = esp32.send_trigger_right()
        print(f"  Resultado: {'✓ Sucesso' if result else '✗ Falha'}")
        
        # Desconectar
        print(f"\nDesconectando...")
        esp32.disconnect()
        print("✓ Desconectado")
        
    else:
        print("✗ Falha ao conectar!")
        print("\nPossíveis causas:")
        print("  1. COM3 não existe ou está em uso")
        print("  2. Drivers USB do ESP32 não instalados")
        print("  3. ESP32 desligado ou não conectado")
        print("  4. Outro programa usando COM3")
        
        # Verificar portas disponíveis
        print("\nVerificando portas seriais disponíveis...")
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            if ports:
                print(f"Portas encontradas ({len(ports)}):")
                for port in ports:
                    print(f"  - {port.device}: {port.description}")
            else:
                print("  Nenhuma porta serial encontrada!")
        except Exception as e:
            print(f"  Erro ao listar portas: {e}")


def test_esp32_singleton():
    """Testa instância singleton"""
    print("\n" + "="*60)
    print("TESTE 2: Singleton do ESP32")
    print("="*60)
    
    esp32_1 = get_esp32_communicator()
    esp32_2 = get_esp32_communicator()
    
    print(f"\nSame object? {esp32_1 is esp32_2} ✓" if esp32_1 is esp32_2 else "✗ Não é singleton!")
    print(f"Status: {esp32_1.get_connection_status()}")


def test_available_ports():
    """Lista todas as portas seriais disponíveis"""
    print("\n" + "="*60)
    print("TESTE 3: Portas Seriais Disponíveis")
    print("="*60)
    
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        
        if ports:
            print(f"\nEncontradas {len(ports)} porta(s):\n")
            for i, port in enumerate(ports, 1):
                print(f"{i}. {port.device}")
                print(f"   Descrição: {port.description}")
                print(f"   Serial: {port.serial_number}")
                print()
        else:
            print("\n✗ Nenhuma porta serial encontrada!")
            print("  Verifique:")
            print("  - USB conectado?")
            print("  - Drivers instalados?")
            print("  - Gerenciador de Dispositivos > Portas COM")
    except Exception as e:
        print(f"\n✗ Erro ao verificar portas: {e}")


if __name__ == "__main__":
    import logging
    
    # Configurar logging detalhado
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)-8s | %(name)s | %(message)s'
    )
    
    print("\n" + "▓"*60)
    print("▓  DIAGNÓSTICO DE COMUNICAÇÃO ESP32")
    print("▓"*60)
    
    # Executar testes
    test_available_ports()
    test_esp32_singleton()
    test_esp32_direct()
    
    print("\n" + "▓"*60)
    print("▓  FIM DOS TESTES")
    print("▓"*60 + "\n")
