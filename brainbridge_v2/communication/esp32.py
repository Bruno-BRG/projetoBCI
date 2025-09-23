"""
Módulo de comunicação serial com ESP32
Envia comandos TRIGGER para ESP32 via porta serial COM4
"""

import serial
import threading
import time
from typing import Optional, Callable
import logging

class ESP32SerialCommunicator:
    """
    Classe para comunicação serial com ESP32
    Envia comandos TRIGGER_LEFT e TRIGGER_RIGHT para ESP32 na COM4
    """
    
    def __init__(self, port: str = "COM4", baudrate: int = 115200, timeout: float = 1.0):
        """
        Inicializa o comunicador serial
        
        Args:
            port: Porta serial (padrão COM4)
            baudrate: Taxa de transmissão (padrão 115200)
            timeout: Timeout para comunicação (padrão 1.0s)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        
        # Estado da conexão
        self.is_connected = False
        self.serial_connection: Optional[serial.Serial] = None
        
        # Lock para thread safety
        self._lock = threading.Lock()
        
        # Callback para mudanças de conexão
        self.on_connection_changed: Optional[Callable[[bool], None]] = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """
        Conecta à porta serial
        
        Returns:
            bool: True se conectado com sucesso
        """
        with self._lock:
            if self.is_connected:
                self.logger.info("ESP32 já conectado")
                return True
            
            try:
                self.serial_connection = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.timeout
                )
                
                # Aguardar um pouco para estabilizar a conexão
                time.sleep(0.5)
                
                # Teste de comunicação
                if self.serial_connection.is_open:
                    self.is_connected = True
                    self.logger.info(f"ESP32 conectado em {self.port} @ {self.baudrate}")
                    
                    # Enviar comando de teste
                    self._send_raw_command("PING")
                    
                    # Notificar mudança de conexão
                    if self.on_connection_changed:
                        self.on_connection_changed(True)
                    
                    return True
                else:
                    self.logger.error("Falha ao abrir porta serial")
                    return False
                    
            except serial.SerialException as e:
                self.logger.error(f"Erro de comunicação serial: {e}")
                self.serial_connection = None
                return False
            except Exception as e:
                self.logger.error(f"Erro inesperado ao conectar ESP32: {e}")
                self.serial_connection = None
                return False
    
    def disconnect(self):
        """
        Desconecta da porta serial
        """
        with self._lock:
            if self.serial_connection and self.serial_connection.is_open:
                try:
                    self.serial_connection.close()
                    self.logger.info("ESP32 desconectado")
                except Exception as e:
                    self.logger.error(f"Erro ao desconectar ESP32: {e}")
                finally:
                    self.serial_connection = None
                    self.is_connected = False
                    
                    # Notificar mudança de conexão
                    if self.on_connection_changed:
                        self.on_connection_changed(False)
    
    def _send_raw_command(self, command: str) -> bool:
        """
        Envia comando bruto para ESP32
        
        Args:
            command: Comando a ser enviado
            
        Returns:
            bool: True se enviado com sucesso
        """
        if not self.is_connected or not self.serial_connection:
            self.logger.warning("ESP32 não conectado - comando ignorado")
            return False
        
        try:
            # Adicionar quebra de linha se não houver
            if not command.endswith('\n'):
                command += '\n'
            
            # Enviar comando
            self.serial_connection.write(command.encode('utf-8'))
            self.serial_connection.flush()
            
            self.logger.debug(f"Comando enviado para ESP32: {command.strip()}")
            return True
            
        except serial.SerialException as e:
            self.logger.error(f"Erro ao enviar comando serial: {e}")
            # Tentar reconectar em caso de erro
            self.is_connected = False
            return False
        except Exception as e:
            self.logger.error(f"Erro inesperado ao enviar comando: {e}")
            return False
    
    def send_trigger_command(self, hand: str) -> bool:
        """
        Envia comando de trigger para ESP32
        
        Args:
            hand: 'direita'/'right' ou 'esquerda'/'left'
            
        Returns:
            bool: True se enviado com sucesso
        """
        if hand.lower() in ['direita', 'right']:
            return self._send_raw_command("TRIGGER_RIGHT")
        elif hand.lower() in ['esquerda', 'left']:
            return self._send_raw_command("TRIGGER_LEFT")
        else:
            self.logger.error(f"Comando de trigger inválido: {hand}")
            return False
    
    def send_trigger_left(self) -> bool:
        """
        Envia trigger para mão esquerda
        
        Returns:
            bool: True se enviado com sucesso
        """
        return self.send_trigger_command('esquerda')
    
    def send_trigger_right(self) -> bool:
        """
        Envia trigger para mão direita
        
        Returns:
            bool: True se enviado com sucesso
        """
        return self.send_trigger_command('direita')
    
    def send_ping(self) -> bool:
        """
        Envia comando PING para testar conexão
        
        Returns:
            bool: True se enviado com sucesso
        """
        return self._send_raw_command("PING")
    
    def set_connection_callback(self, callback: Callable[[bool], None]):
        """
        Define callback para mudanças de conexão
        
        Args:
            callback: Função a ser chamada quando conexão muda
        """
        self.on_connection_changed = callback
    
    def get_connection_status(self) -> dict:
        """
        Retorna status da conexão
        
        Returns:
            dict: Informações sobre a conexão
        """
        return {
            'connected': self.is_connected,
            'port': self.port,
            'baudrate': self.baudrate,
            'timeout': self.timeout
        }


# Instância singleton para fácil acesso
_esp32_communicator: Optional[ESP32SerialCommunicator] = None
_communicator_lock = threading.Lock()

def get_esp32_communicator() -> ESP32SerialCommunicator:
    """
    Retorna instância singleton do comunicador ESP32
    
    Returns:
        ESP32SerialCommunicator: Instância do comunicador
    """
    global _esp32_communicator
    
    with _communicator_lock:
        if _esp32_communicator is None:
            _esp32_communicator = ESP32SerialCommunicator()
        return _esp32_communicator


# Funções de conveniência para compatibilidade
def send_trigger_left() -> bool:
    """Envia trigger esquerdo via ESP32"""
    return get_esp32_communicator().send_trigger_left()

def send_trigger_right() -> bool:
    """Envia trigger direito via ESP32"""
    return get_esp32_communicator().send_trigger_right()

def connect_esp32() -> bool:
    """Conecta ao ESP32"""
    return get_esp32_communicator().connect()

def disconnect_esp32():
    """Desconecta do ESP32"""
    get_esp32_communicator().disconnect()

def is_esp32_connected() -> bool:
    """Verifica se ESP32 está conectado"""
    return get_esp32_communicator().is_connected


if __name__ == "__main__":
    """Teste básico do módulo"""
    import logging
    
    # Configurar logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Teste de comunicação
    esp32 = ESP32SerialCommunicator()
    
    print("Testando comunicação com ESP32...")
    
    if esp32.connect():
        print("✓ Conectado ao ESP32")
        
        # Testar comandos
        print("Testando PING...")
        esp32.send_ping()
        time.sleep(1)
        
        print("Testando TRIGGER_LEFT...")
        esp32.send_trigger_left()
        time.sleep(1)
        
        print("Testando TRIGGER_RIGHT...")
        esp32.send_trigger_right()
        time.sleep(1)
        
        esp32.disconnect()
        print("✓ Desconectado do ESP32")
    else:
        print("✗ Falha ao conectar ESP32")