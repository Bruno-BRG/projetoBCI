"""
Teste do sistema de trigger com duração de 3 segundos
Valida bloqueio de trigger e liberação automática
"""

import sys
import time
from pathlib import Path

# Adicionar o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from communication.esp32 import ESP32SerialCommunicator

def test_trigger_duration():
    """Testa duração do trigger de 3 segundos"""
    print("\n" + "="*60)
    print("TESTE: Sistema de Trigger com Duração de 3 Segundos")
    print("="*60)
    
    esp32 = ESP32SerialCommunicator(port="COM3", baudrate=115200)
    
    # Conectar
    print(f"\nConectando em COM3...")
    if not esp32.connect():
        print("✗ Falha ao conectar")
        return
    
    print("✓ Conectado com sucesso!")
    
    # Teste 1: Enviar trigger esquerdo
    print(f"\n--- Teste 1: Enviar trigger esquerdo ---")
    print(f"Status antes: {esp32.is_trigger_ready()=}")
    result = esp32.send_trigger_left()
    print(f"Resultado: {'✓ Enviado' if result else '✗ Falhou'}")
    print(f"Status depois: {esp32.is_trigger_ready()=}")
    print(f"Tempo bloqueado: {esp32.get_trigger_remaining_time():.2f}s")
    
    # Teste 2: Tentar enviar trigger direito imediatamente (deve falhar)
    print(f"\n--- Teste 2: Tentar enviar trigger direito imediatamente ---")
    result = esp32.send_trigger_right()
    print(f"Resultado: {'✓ Enviado (BUG!)' if result else '✗ Bloqueado (correto)'}")
    print(f"Tempo restante: {esp32.get_trigger_remaining_time():.2f}s")
    
    # Teste 3: Aguardar 1.5 segundos e tentar novamente
    print(f"\n--- Teste 3: Aguardar 1.5s e tentar novamente ---")
    print(f"Aguardando 1.5 segundos...")
    time.sleep(1.5)
    print(f"Tempo restante: {esp32.get_trigger_remaining_time():.2f}s")
    result = esp32.send_trigger_right()
    print(f"Resultado: {'✓ Enviado (BUG!)' if result else '✗ Bloqueado (correto)'}")
    
    # Teste 4: Aguardar até liberar completamente
    print(f"\n--- Teste 4: Aguardar até liberar completamente ---")
    tempo_restante = esp32.get_trigger_remaining_time()
    print(f"Tempo restante: {tempo_restante:.2f}s")
    print(f"Aguardando {tempo_restante + 0.5:.2f} segundos...")
    time.sleep(tempo_restante + 0.5)
    
    print(f"Status: {esp32.is_trigger_ready()=}")
    result = esp32.send_trigger_right()
    print(f"Resultado: {'✓ Enviado (correto)' if result else '✗ Bloqueado (BUG!)'}")
    
    # Teste 5: Múltiplos triggers em sequência com espera
    print(f"\n--- Teste 5: Sequência de triggers com espera ---")
    for i in range(3):
        hand = 'esquerda' if i % 2 == 0 else 'direita'
        print(f"\n  Trigger {i+1}: {hand}")
        
        if not esp32.is_trigger_ready():
            tempo = esp32.get_trigger_remaining_time()
            print(f"    Aguardando {tempo:.2f}s...")
            time.sleep(tempo + 0.1)
        
        result = esp32.send_trigger_command(hand)
        print(f"    Resultado: {'✓' if result else '✗'}")
        print(f"    Próximo trigger em: {esp32.get_trigger_remaining_time():.2f}s")
    
    # Desconectar
    print(f"\n\nDesconectando...")
    esp32.disconnect()
    print("✓ Teste concluído!")


if __name__ == "__main__":
    import logging
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)-8s | %(name)s | %(message)s'
    )
    
    test_trigger_duration()
