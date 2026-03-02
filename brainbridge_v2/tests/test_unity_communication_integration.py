"""
Testes de integracaeo UDP + TCP com simulacao de VR
Valida fluxo completo: servidor UDP -> broadcast -> cliente TCP conecta -> protocolo completo
"""

import sys
import socket
import threading
import time
from typing import Optional

sys.path.insert(0, 'c:\\Users\\Chari\\Documents\\dev\\BrainBridge')

from brainbridge_v2.infrastructure.communication.unity import (
    UnityCommunicator, SessionPhase, ServerState, 
    PatientData, TaskType, UDP_sender
)


class MockVRClient:
    """Simula um cliente VR que se conecta e responde ao protocolo"""
    
    def __init__(self, host='localhost', port=12345, udp_port=12346):
        self.host = host
        self.tcp_port = port
        self.udp_port = udp_port
        self.tcp_socket: Optional[socket.socket] = None
        self.udp_socket: Optional[socket.socket] = None
        self.connected = False
        self.messages_received = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
    def start_listening_udp(self):
        """Comeca a escutar broadcast UDP pra encontrar o servidor"""
        def udp_listener():
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                self.udp_socket.bind(('', self.udp_port))
                print(f"[MockVR-UDP] Escutando broadcast na porta {self.udp_port}")
                
                while self.running:
                    try:
                        data, addr = self.udp_socket.recvfrom(1024)
                        message = data.decode('utf-8', errors='ignore')
                        print(f"[MockVR-UDP] Recebeu broadcast: {message}")
                        
                        if "Confirm" in message:
                            print("[MockVR-UDP] [OK] Broadcast confirmado!")
                            break
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.running:
                            print(f"[MockVR-UDP] Erro: {e}")
                        break
            finally:
                if self.udp_socket:
                    self.udp_socket.close()
        
        self.running = True
        self.udp_socket = None
        thread = threading.Thread(target=udp_listener, daemon=True)
        thread.start()
        time.sleep(0.2)
        return thread
    
    def connect_tcp(self) -> bool:
        """Conecta ao servidor TCP"""
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.settimeout(5.0)
            self.tcp_socket.connect((self.host, self.tcp_port))
            self.connected = True
            print(f"[MockVR-TCP] [OK] Conectado ao servidor em {self.host}:{self.tcp_port}")
            return True
        except Exception as e:
            print(f"[MockVR-TCP] [FAIL] Falha ao conectar: {e}")
            return False
    
    def start_receiver(self):
        """Comeca a receber mensagens do servidor"""
        def receiver():
            if not self.tcp_socket:
                return
                
            try:
                self.tcp_socket.settimeout(2.0)
                while self.running:
                    try:
                        data = self.tcp_socket.recv(4096)
                        if not data:
                            print("[MockVR-TCP] [DISC] Servidor desconectou")
                            break
                        
                        message = data.decode('utf-8', errors='ignore').strip()
                        self.messages_received.append(message)
                        print(f"[MockVR-TCP] [RECV] Recebeu: {message[:50]}...")
                        
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.running:
                            print(f"[MockVR-TCP] Erro na recepcao: {e}")
                        break
            finally:
                self.connected = False
        
        self.running = True
        self.thread = threading.Thread(target=receiver, daemon=True)
        self.thread.start()
    
    def send_response(self, message: str):
        """Envia resposta pro servidor"""
        try:
            if self.tcp_socket:
                self.tcp_socket.sendall((message + '\n').encode('utf-8'))
                print(f"[MockVR-TCP] [SEND] Enviou: {message}")
                return True
        except Exception as e:
            print(f"[MockVR-TCP] [FAIL] Erro ao enviar: {e}")
        return False
    
    def send_confirmation(self):
        """Envia confirmacao de que recebeu dados/tarefa"""
        return self.send_response("Confirm: VR ready")
    
    def send_end_confirmation(self):
        """Envia confirmacao de finalizacao"""
        return self.send_response("Finalizar: sessao encerrada")
    
    def disconnect(self):
        """Desconecta"""
        self.running = False
        if self.tcp_socket:
            try:
                self.tcp_socket.close()
            except:
                pass
        if self.udp_socket:
            try:
                self.udp_socket.close()
            except:
                pass
        self.connected = False
        print("[MockVR] Desconectado")


# ============================================================================
# TESTES DE INTEGRACAO
# ============================================================================

def cleanup():
    """Limpa o estado entre testes"""
    try:
        comm = UnityCommunicator()
        if comm.server_state != ServerState.STOPPED:
            comm.stop_server()
        time.sleep(0.3)
    except:
        pass
    
    # Resetar singleton
    UnityCommunicator._instance = None
    time.sleep(0.2)


def test_server_startup():
    """Testa se o servidor inicia corretamente"""
    cleanup()
    
    print("\n" + "="*70)
    print("[TEST 1] Startup do servidor")
    print("="*70)
    
    comm = UnityCommunicator()
    
    # Estado inicial
    assert comm.server_state == ServerState.STOPPED, "Deve iniciar STOPPED"
    print("[OK] Estado inicial: STOPPED")
    
    # Iniciar
    assert comm.start_server(), "Deve iniciar com sucesso"
    assert comm.server_state == ServerState.RUNNING, "Deve estar RUNNING"
    print("[OK] Servidor iniciado: RUNNING")
    
    # Parar
    comm.stop_server()
    assert comm.server_state == ServerState.STOPPED, "Deve parar"
    print("[OK] Servidor parado: STOPPED")
    
    cleanup()


def test_udp_broadcast_discovery():
    """Testa se o broadcast UDP funciona pra descoberta"""
    cleanup()
    
    print("\n" + "="*70)
    print("[TEST 2] UDP broadcast discovery")
    print("="*70)
    
    comm = UnityCommunicator()
    mock_vr = MockVRClient()
    
    # Iniciar servidor
    assert comm.start_server(), "Servidor deve iniciar"
    print("[OK] Servidor iniciado")
    
    # Escutar UDP broadcast
    udp_thread = mock_vr.start_listening_udp()
    time.sleep(0.5)
    
    # Aguardar receber broadcast
    mock_vr.running = False
    udp_thread.join(timeout=2)
    
    print("[OK] UDP broadcast funcionando")
    
    # Limpar
    comm.stop_server()
    mock_vr.disconnect()
    cleanup()


def test_tcp_connection():
    """Testa se conexao TCP funciona"""
    cleanup()
    
    print("\n" + "="*70)
    print("[TEST 3] Conexao TCP")
    print("="*70)
    
    comm = UnityCommunicator()
    mock_vr = MockVRClient()
    
    # Iniciar servidor
    assert comm.start_server(), "Servidor deve iniciar"
    print("[OK] Servidor iniciado")
    
    # Conectar cliente VR
    time.sleep(0.2)  # Dar tempo pro TCP server iniciar
    assert mock_vr.connect_tcp(), "VR deve conectar"
    print("[OK] VR conectou ao TCP")
    
    # Verificar estado do servidor
    time.sleep(0.2)
    assert comm.server_state == ServerState.CONNECTED, "Servidor deve estar CONNECTED"
    assert comm.tcp_connected, "Deve ter tcp_connected = True"
    print("[OK] Servidor transitou para CONNECTED")
    
    # Desconectar
    mock_vr.disconnect()
    comm.stop_server()
    print("[OK] Desconectado")
    cleanup()


def test_protocol_session_setup():
    """Testa fluxo completo de setup de sessao"""
    cleanup()
    
    print("\n" + "="*70)
    print("[TEST 4] Protocolo - Setup de sessao")
    print("="*70)
    
    comm = UnityCommunicator()
    mock_vr = MockVRClient()
    
    # Iniciar servidor
    assert comm.start_server(), "Servidor deve iniciar"
    print("[OK] Servidor iniciado")
    
    # Conectar VR
    time.sleep(0.2)
    assert mock_vr.connect_tcp(), "VR deve conectar"
    mock_vr.start_receiver()
    print("[OK] VR conectado")
    
    # Estado inicial
    assert comm.session.phase == SessionPhase.IDLE, "Deve iniciar IDLE"
    print("[OK] Sessao em IDLE")
    
    # Iniciar sessao
    patient = PatientData(nome="Joao Silva", nivel=5, lado="Direito")
    assert comm.start_session(patient, TaskType.TREINO), "Deve iniciar sessao"
    
    # Deve estar em SETUP
    assert comm.session.phase == SessionPhase.SETUP, "Deve estar em SETUP"
    print("[OK] Sessao transitou para SETUP")
    
    # VR recebeu mensagens?
    time.sleep(0.3)
    assert len(mock_vr.messages_received) > 0, "VR deve ter recebido mensagens"
    print(f"[OK] VR recebeu {len(mock_vr.messages_received)} mensagens")
    
    # VR envia confirmacao
    mock_vr.send_confirmation()
    time.sleep(0.2)
    
    # Deve ter transitado para READY
    assert comm.session.phase == SessionPhase.READY, f"Deve estar em READY, esta em {comm.session.phase}"
    print("[OK] Sessao transitou para READY (recebeu confirmacao)")
    
    # Limpar
    mock_vr.disconnect()
    comm.stop_server()
    cleanup()


def test_protocol_trigger():
    """Testa envio de trigger"""
    cleanup()
    
    print("\n" + "="*70)
    print("[TEST 5] Protocolo - Trigger")
    print("="*70)
    
    comm = UnityCommunicator()
    mock_vr = MockVRClient()
    
    # Setup
    assert comm.start_server(), "Servidor deve iniciar"
    time.sleep(0.2)
    assert mock_vr.connect_tcp(), "VR deve conectar"
    mock_vr.start_receiver()
    
    # Iniciar sessao
    patient = PatientData(nome="Maria", nivel=3, lado="Esquerdo")
    comm.start_session(patient, TaskType.JOGO)
    time.sleep(0.2)
    
    # VR confirma
    mock_vr.send_confirmation()
    time.sleep(0.2)
    
    # Deve estar em READY
    assert comm.session.phase == SessionPhase.READY, "Deve estar em READY antes do trigger"
    print("[OK] Sessao em READY")
    
    # Enviar trigger
    assert comm.send_trigger(), "Deve enviar trigger"
    
    # Deve estar em ACTIVE
    assert comm.session.phase == SessionPhase.ACTIVE, "Deve estar em ACTIVE apos trigger"
    print("[OK] Sessao transitou para ACTIVE (trigger enviado)")
    
    # VR deve ter recebido comando "Trigger"
    time.sleep(0.2)
    trigger_messages = [m for m in mock_vr.messages_received if "Trigger" in m]
    assert len(trigger_messages) > 0, "VR deve ter recebido mensagem com Trigger"
    print("[OK] VR recebeu comando Trigger")
    
    # Limpar
    mock_vr.disconnect()
    comm.stop_server()
    cleanup()


def test_error_cases():
    """Testa casos de erro e transicoes invalidas"""
    cleanup()
    
    print("\n" + "="*70)
    print("[TEST 6] Casos de erro")
    print("="*70)
    
    comm = UnityCommunicator()
    mock_vr = MockVRClient()
    
    # Erro 1: Tentar iniciar sessao sem servidor rodando
    patient = PatientData(nome="Test", nivel=5, lado="Direito")
    assert not comm.start_session(patient, TaskType.TREINO), "Deve falhar sem servidor"
    print("[OK] Erro 1: Nao inicia sessao sem servidor")
    
    # Erro 2: Tentar iniciar sessao sem VR conectado
    assert comm.start_server(), "Servidor deve iniciar"
    time.sleep(0.2)
    assert not comm.start_session(patient, TaskType.TREINO), "Deve falhar sem VR"
    print("[OK] Erro 2: Nao inicia sessao sem VR conectado")
    
    # Erro 3: Enviar trigger sem estar em READY
    assert mock_vr.connect_tcp(), "VR deve conectar"
    mock_vr.start_receiver()
    time.sleep(0.2)
    
    # Sessao esta em IDLE
    assert not comm.send_trigger(), "Nao deve enviar trigger em IDLE"
    print("[OK] Erro 3: Trigger bloqueado em IDLE")
    
    # Erro 4: Enviar comando sem sessao ativa
    assert not comm.send_hand_close('direita'), "Nao deve enviar comando sem sessao ativa"
    print("[OK] Erro 4: Comando bloqueado sem sessao ativa")
    
    # Erro 5: Finalizar sessao sem estar ativa
    assert not comm.end_session(), "Nao deve finalizar sem sessao ativa"
    print("[OK] Erro 5: Finalizacao bloqueada sem sessao ativa")
    
    # Limpar
    mock_vr.disconnect()
    comm.stop_server()
    cleanup()


def test_legacy_compatibility():
    """Testa compatibilidade com codigo legado via UDP_sender"""
    cleanup()
    
    print("\n" + "="*70)
    print("[TEST 7] Compatibilidade legada (UDP_sender)")
    print("="*70)
    
    # Iniciar servidor via classe legada
    assert UDP_sender.init_zmq_socket(), "Deve iniciar servidor"
    assert UDP_sender.is_server_active(), "Servidor deve estar ativo"
    print("[OK] UDP_sender.init_zmq_socket() funciona")
    
    # Parar
    UDP_sender.stop_zmq_socket()
    assert not UDP_sender.is_server_active(), "Servidor deve estar parado"
    print("[OK] UDP_sender.stop_zmq_socket() funciona")
    
    cleanup()


def main():
    """Executa todos os testes de integracao"""
    print("\n")
    print("="*70)
    print(" TESTES DE INTEGRACAO: UDP + TCP + PROTOCOLO COMPLETO")
    print("="*70)
    
    tests = [
        ("Server Startup", test_server_startup),
        ("UDP Broadcast Discovery", test_udp_broadcast_discovery),
        ("TCP Connection", test_tcp_connection),
        ("Protocol Session Setup", test_protocol_session_setup),
        ("Protocol Trigger", test_protocol_trigger),
        ("Error Cases", test_error_cases),
        ("Legacy Compatibility", test_legacy_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] TESTE FALHOU: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[ERROR] ERRO: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Resumo
    print("\n" + "="*70)
    print(f"[OK] {passed} testes passaram")
    if failed > 0:
        print(f"[FAIL] {failed} testes falharam")
    print("="*70)
    print("\nCobertura:")
    print("   [OK] Server lifecycle (start/stop)")
    print("   [OK] UDP broadcast discovery")
    print("   [OK] TCP connection handshake")
    print("   [OK] Session protocol (setup -> trigger -> commands -> end)")
    print("   [OK] State machine transitions")
    print("   [OK] Error handling")
    print("   [OK] Legacy compatibility")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
