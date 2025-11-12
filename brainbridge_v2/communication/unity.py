# unity_communication.py
"""
Sistema unificado de comunicação com Unity
Substitui UDP_sender.py e udp_receiver.py por uma abordagem mais simples e robusta

Protocolo:
1. Sistema: Broadcast UDP com "Confirm"
2. VR: Responde com "Header: Confirm"
3. Sistema: Envia Dados Paciente (Nome, Nível, Lado)
4. Sistema: Envia Tarefa ("Treino" ou "Jogo")
5. Sistema: Envia Trigger + LEFT/RIGHT_HAND_CLOSE
6. VR: Responde com LEFT_FLOWER ou RIGHT_FLOWER
7. Sistema: Envia END_TASK ou END_GAME com mensagem
8. VR: Confirma finalização
"""

import socket
import threading
import time
import zmq
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Callable


# ============================================================================
# ENUMS e CLASSES DE DADOS DO PROTOCOLO
# ============================================================================

class TaskType(Enum):
    """Tipo de tarefa no VR"""
    TREINO = "Treino"
    JOGO = "Jogo"


class SessionPhase(Enum):
    """Fases da sessão (para máquina de estados)"""
    IDLE = "idle"
    SETUP = "setup"
    READY = "ready"
    ACTIVE = "active"
    ENDING = "ending"


class ServerState(Enum):
    """Estados do servidor"""
    STOPPED = "stopped"
    RUNNING = "running"
    CONNECTED = "connected"


class ActionCommand(Enum):
    """Comandos de ação do sistema"""
    LEFT_HAND_CLOSE = "LEFT_HAND_CLOSE"
    RIGHT_HAND_CLOSE = "RIGHT_HAND_CLOSE"
    LEFT_FLOWER = "LEFT_FLOWER"
    RIGHT_FLOWER = "RIGHT_FLOWER"


class EndTaskCommand(Enum):
    """Comandos de finalização de tarefa"""
    END_TRAINING = "Finalizar_tarefa_treino"
    END_GAME = "Finalizar_tarefa_jogo"


@dataclass
class PatientData:
    """Dados do paciente para enviar ao VR"""
    nome: str
    nivel: int  # 0-11
    lado: str   # "Direito" ou "Esquerdo"
    
    def __post_init__(self):
        """Valida dados do paciente"""
        if not (0 <= self.nivel <= 11):
            raise ValueError(f"Nível deve estar entre 0 e 11, recebido: {self.nivel}")
        
        if self.lado not in ["Direito", "Esquerdo"]:
            raise ValueError(f"Lado deve ser 'Direito' ou 'Esquerdo', recebido: {self.lado}")
    
    def format_message(self) -> str:
        """Formata dados do paciente para envio ao VR"""
        return f"Dados Paciente:\nNome: {self.nome}\nNivel: {self.nivel}\nLado: {self.lado}"


@dataclass
class SessionState:
    """Estado da sessão atual"""
    patient: Optional[PatientData] = None
    task_type: Optional[TaskType] = None
    is_active: bool = False
    waiting_confirmation: bool = False
    phase: 'SessionPhase' = None  # Suporte para máquina de estados
    
    def __post_init__(self):
        """Inicializa a fase"""
        if self.phase is None:
            self.phase = SessionPhase.IDLE
    
    def reset(self):
        """Reseta o estado da sessão"""
        self.patient = None
        self.task_type = None
        self.is_active = False
        self.waiting_confirmation = False
        self.phase = SessionPhase.IDLE
    
    def transition_to(self, new_phase: 'SessionPhase') -> bool:
        """
        Tenta transicionar para nova fase
        Retorna True se transição foi válida
        """
        # Transições válidas
        valid_transitions = {
            SessionPhase.IDLE: [SessionPhase.SETUP],
            SessionPhase.SETUP: [SessionPhase.READY, SessionPhase.IDLE],
            SessionPhase.READY: [SessionPhase.ACTIVE, SessionPhase.IDLE],
            SessionPhase.ACTIVE: [SessionPhase.ENDING, SessionPhase.IDLE],
            SessionPhase.ENDING: [SessionPhase.IDLE],
        }
        
        current = self.phase
        if current in valid_transitions and new_phase in valid_transitions[current]:
            self.phase = new_phase
            return True
        return False

class UnityCommunicator:
    """
    Classe unificada para comunicação com Unity usando TCP + ZMQ
    Combina as funcionalidades de UDP_sender e udp_receiver em uma única interface
    """
    
    # Configurações
    UDP_PORT = 12346      # porta para broadcast de IPs
    TCP_PORT = 12345      # porta para servidor TCP Unity
    ZMQ_PORT = 5555       # porta para ZMQ publisher
    BROADCAST_INTERVAL = 1.0
    BUFFER_SIZE = 4096
    
    # Protocolo
    CONFIRM_HEADER = "Confirm"
    
    # Variáveis de classe para singleton
    _instance: Optional['UnityCommunicator'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implementa padrão singleton"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(UnityCommunicator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inicializa o comunicador"""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        
        # Estado da conexão
        self.is_active = False
        self.tcp_connected = False
        
        # Estado da sessão (novo protocolo)
        self.session = SessionState()
        
        # Sockets e contextos
        self.zmq_context: Optional[zmq.Context] = None
        self.zmq_socket: Optional[zmq.Socket] = None
        self.tcp_connection: Optional[socket.socket] = None
        
        # Threads de controle
        self.broadcast_thread: Optional[threading.Thread] = None
        self.tcp_server_thread: Optional[threading.Thread] = None
        self.tcp_handler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Callbacks para eventos
        self.on_message_received: Optional[Callable[[str], None]] = None
        self.on_connection_changed: Optional[Callable[[bool], None]] = None
        
        # Callbacks para novo protocolo
        self.on_confirmation: Optional[Callable[[], None]] = None  # VR confirmou
        self.on_flower_action: Optional[Callable[[ActionCommand], None]] = None  # VR enviou FLOWER
    
    @staticmethod
    def get_all_ips() -> List[str]:
        """
        Retorna lista de IPs IPv4 locais usando stdlib.
        """
        ips = set()
        try:
            hostname = socket.gethostname()
            for res in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ips.add(res[4][0])
        except Exception:
            pass
        if not ips:
            ips.add('127.0.0.1')
        return list(ips)
    
    def start_server(self) -> bool:
        """
        Inicia o servidor de comunicação
        Retorna True se iniciado com sucesso
        """
        if self.is_active:
            print("Servidor já está ativo")
            return True
            
        try:
            # Inicializar ZMQ
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PUB)
            self.zmq_socket.bind(f"tcp://*:{self.ZMQ_PORT}")
            
            # Reset do evento de parada
            self.stop_event.clear()
            
            # Iniciar broadcast UDP
            self.broadcast_thread = threading.Thread(
                target=self._broadcast_ips, 
                daemon=True
            )
            self.broadcast_thread.start()
            
            # Iniciar servidor TCP
            self.tcp_server_thread = threading.Thread(
                target=self._tcp_server, 
                daemon=True
            )
            self.tcp_server_thread.start()
            
            self.is_active = True
            print(f"Servidor iniciado - ZMQ: {self.ZMQ_PORT}, TCP: {self.TCP_PORT}, UDP: {self.UDP_PORT}")
            return True
            
        except Exception as e:
            print(f"Erro ao iniciar servidor: {e}")
            self.stop_server()
            return False
    
    def stop_server(self):
        """Para o servidor e limpa recursos"""
        self.is_active = False
        self.stop_event.set()
        
        # Aguardar threads terminarem
        if self.broadcast_thread and self.broadcast_thread.is_alive():
            self.broadcast_thread.join(timeout=2.0)
            
        if self.tcp_server_thread and self.tcp_server_thread.is_alive():
            self.tcp_server_thread.join(timeout=2.0)
            
        if self.tcp_handler_thread and self.tcp_handler_thread.is_alive():
            self.tcp_handler_thread.join(timeout=2.0)
        
        # Fechar conexão TCP
        if self.tcp_connection:
            try:
                self.tcp_connection.close()
            except Exception:
                pass
            self.tcp_connection = None
            
        # Fechar ZMQ
        if self.zmq_socket:
            try:
                self.zmq_socket.close()
            except Exception:
                pass
            self.zmq_socket = None
            
        if self.zmq_context:
            try:
                self.zmq_context.term()
            except Exception:
                pass
            self.zmq_context = None
        
        # Atualizar estado
        if self.tcp_connected:
            self.tcp_connected = False
            if self.on_connection_changed:
                self.on_connection_changed(False)
        
        print("Servidor parado e recursos limpos")
    
    def send_command(self, command: str) -> bool:
        """
        Envia comando para Unity via ZMQ e TCP
        Retorna True se enviado com sucesso
        """
        if not self.is_active:
            print("Servidor não está ativo")
            return False
            
        success = False
        
        # Enviar via ZMQ (sempre disponível quando servidor ativo)
        if self.zmq_socket:
            try:
                self.zmq_socket.send_string(command)
                print(f"[ZMQ] Comando enviado: {command}")
                success = True
            except Exception as e:
                print(f"[ZMQ] Erro ao enviar: {e}")
        
        # Enviar via TCP se conectado
        if self.tcp_connected and self.tcp_connection:
            try:
                message = command + '\n'
                self.tcp_connection.sendall(message.encode('utf-8'))
                print(f"[TCP] Comando enviado: {command}")
                success = True
            except Exception as e:
                print(f"[TCP] Erro ao enviar: {e}")
                self.tcp_connected = False
                if self.on_connection_changed:
                    self.on_connection_changed(False)
        
        return success
    
    def _send_protocol_message(self, message: str) -> bool:
        """
        Método legado para compatibilidade com testes anteriores
        Alias para send_command
        """
        return self.send_command(message)
    
    def send_hand_command(self, hand: str) -> bool:
        """
        Envia comando de mão (direita/esquerda)
        """
        if hand.lower() in ['direita', 'right']:
            return self.send_command("RIGHT_HAND_CLOSE")
        elif hand.lower() in ['esquerda', 'left']:
            return self.send_command("LEFT_HAND_CLOSE")
        else:
            print(f"Comando de mão inválido: {hand}")
            return False
    
    def send_trigger_command(self, hand: str) -> bool:
        """
        Envia comando de trigger
        """
        if hand.lower() in ['direita', 'right']:
            return self.send_command("TRIGGER_RIGHT")
        elif hand.lower() in ['esquerda', 'left']:
            return self.send_command("TRIGGER_LEFT")
        else:
            print(f"Comando de trigger inválido: {hand}")
            return False
    
    # ========================================================================
    # NOVO PROTOCOLO - MÉTODOS DE SESSÃO E TAREFA
    # ========================================================================
    
    def start_session(self, patient: PatientData, task_type: TaskType) -> bool:
        """
        Inicia uma nova sessão com paciente e tipo de tarefa
        Envia dados do paciente e tipo de tarefa ao VR
        """
        if not self.is_active:
            print("Erro: Servidor não está ativo")
            return False
        
        if not self.tcp_connected:
            print("Erro: VR não está conectado")
            return False
        
        # Validar paciente
        try:
            if not isinstance(patient, PatientData):
                raise ValueError("Paciente deve ser uma instância de PatientData")
        except Exception as e:
            print(f"Erro ao validar paciente: {e}")
            return False
        
        # Atualizar estado da sessão
        self.session.patient = patient
        self.session.task_type = task_type
        self.session.waiting_confirmation = True
        
        # 1. Enviar dados do paciente
        print(f"\n📋 Enviando dados do paciente...")
        patient_message = patient.format_message()
        if not self.send_command(patient_message):
            print("Erro ao enviar dados do paciente")
            self.session.reset()
            return False
        
        time.sleep(0.5)  # Pequeno delay entre mensagens
        
        # 2. Enviar tipo de tarefa
        print(f"📌 Enviando tipo de tarefa: {task_type.value}")
        if not self.send_command(f"Tarefa:\n{task_type.value}"):
            print("Erro ao enviar tipo de tarefa")
            self.session.reset()
            return False
        
        print("✅ Sessão iniciada com sucesso")
        return True
    
    def send_trigger(self) -> bool:
        """
        Envia trigger para iniciar tarefa no VR
        Segue com comando de fechar mão
        """
        if not self._is_session_active_for_commands():
            print("Erro: Sessão não está ativa para enviar trigger")
            return False
        
        print(f"🎯 Enviando trigger...")
        if not self.send_command("Trigger"):
            return False
        
        time.sleep(0.3)
        
        # Enviar comando de fechar mão baseado no lado do paciente
        hand_command = ActionCommand.RIGHT_HAND_CLOSE if self.session.patient.lado == "Direito" else ActionCommand.LEFT_HAND_CLOSE
        print(f"👋 Enviando comando: {hand_command.value}")
        
        success = self.send_command(hand_command.value)
        
        if success:
            self.session.is_active = True
            self.session.waiting_confirmation = False
        
        return success
    
    def send_hand_close(self, lado: str) -> bool:
        """
        Envia comando de fechar mão para o lado especificado
        Compatível com novo protocolo
        """
        if not self._is_session_active_for_commands():
            print("Erro: Sessão não está ativa")
            return False
        
        if lado.lower() in ['direita', 'right']:
            return self.send_command(ActionCommand.RIGHT_HAND_CLOSE.value)
        elif lado.lower() in ['esquerda', 'left']:
            return self.send_command(ActionCommand.LEFT_HAND_CLOSE.value)
        else:
            print(f"Lado inválido: {lado}")
            return False
    
    def send_flower_action(self, lado: str) -> bool:
        """
        Envia comando de ação de flor
        (simulação de resposta do VR)
        """
        if lado.lower() in ['direita', 'right']:
            return self.send_command(ActionCommand.RIGHT_FLOWER.value)
        elif lado.lower() in ['esquerda', 'left']:
            return self.send_command(ActionCommand.LEFT_FLOWER.value)
        else:
            print(f"Lado inválido: {lado}")
            return False
    
    def end_task(self, message: str = "") -> bool:
        """
        Finaliza a tarefa atual
        Envia END_TASK com mensagem opcional
        """
        if not self.session.is_active:
            print("Erro: Nenhuma tarefa ativa para finalizar")
            return False
        
        if not self.tcp_connected:
            print("Erro: VR não está conectado")
            return False
        
        # Escolher comando baseado no tipo de tarefa
        if self.session.task_type == TaskType.TREINO:
            end_command = EndTaskCommand.END_TRAINING.value
        else:
            end_command = EndTaskCommand.END_GAME.value
        
        # Enviar comando de finalização
        end_message = f"{end_command}\n{message}" if message else end_command
        print(f"✋ Finalizando tarefa: {end_message}")
        
        success = self.send_command(end_message)
        
        if success:
            self.session.reset()
        
        return success
    
    def set_confirmation_callback(self, callback: Callable[[], None]):
        """Define callback para confirmação do VR"""
        self.on_confirmation = callback
    
    def set_flower_callback(self, callback: Callable[[ActionCommand], None]):
        """Define callback para ações de flor do VR"""
        self.on_flower_action = callback
    
    def _is_session_active_for_commands(self) -> bool:
        """
        Verifica se sessão está ativa e pronta para comandos
        Retorna True se: servidor ativo, VR conectado e sessão ativa
        """
        return (self.is_active and 
                self.tcp_connected and 
                self.session.patient is not None and
                self.session.task_type is not None)
    
    def _broadcast_ips(self):
        """
        Thread para broadcast dos IPs via UDP
        """
        ips = self.get_all_ips()
        message = ','.join(ips).encode('utf-8')
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        print(f"[UDP] Iniciando broadcast: {ips}")
        
        try:
            while not self.stop_event.is_set():
                sock.sendto(message, ('<broadcast>', self.UDP_PORT))
                time.sleep(self.BROADCAST_INTERVAL)
        except Exception as e:
            print(f"[UDP] Erro no broadcast: {e}")
        finally:
            sock.close()
            print("[UDP] Broadcast parado")
    
    def _tcp_server(self):
        """
        Thread para servidor TCP
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)  # Timeout para permitir verificação de stop_event
        
        try:
            sock.bind(('', self.TCP_PORT))
            sock.listen(1)
            print(f"[TCP] Servidor ouvindo na porta {self.TCP_PORT}")
            
            while not self.stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                    print(f"[TCP] Unity conectado de {addr}")
                    
                    self.tcp_connection = conn
                    self.tcp_connected = True
                    
                    if self.on_connection_changed:
                        self.on_connection_changed(True)
                    
                    # Iniciar thread para lidar com esta conexão
                    self.tcp_handler_thread = threading.Thread(
                        target=self._handle_tcp_connection,
                        args=(conn, addr),
                        daemon=True
                    )
                    self.tcp_handler_thread.start()
                    
                    # Aguardar conexão terminar antes de aceitar nova
                    self.tcp_handler_thread.join()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"[TCP] Erro no servidor: {e}")
                    break
                    
        except Exception as e:
            print(f"[TCP] Erro ao iniciar servidor: {e}")
        finally:
            sock.close()
            print("[TCP] Servidor TCP parado")
    
    def _handle_tcp_connection(self, conn: socket.socket, addr):
        """
        Lida com uma conexão TCP específica
        Processa mensagens do VR e dispara callbacks apropriados
        """
        try:
            conn.settimeout(1.0)
            
            while not self.stop_event.is_set() and self.tcp_connected:
                try:
                    data = conn.recv(self.BUFFER_SIZE)
                    if not data:
                        print("[TCP] Unity desconectou")
                        break
                        
                    message = data.decode('utf-8', errors='ignore').strip()
                    print(f"[TCP] Recebido: {message}")
                    
                    # Processar mensagens do novo protocolo
                    self._process_vr_message(message)
                    
                    # Callback genérico (compatibilidade)
                    if self.on_message_received:
                        self.on_message_received(message)
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[TCP] Erro na recepção: {e}")
                    break
                    
        finally:
            try:
                conn.close()
            except Exception:
                pass
            
            self.tcp_connection = None
            self.tcp_connected = False
            
            if self.on_connection_changed:
                self.on_connection_changed(False)
            
            print("[TCP] Conexão encerrada")
    
    def _process_vr_message(self, message: str):
        """
        Processa mensagens recebidas do VR conforme o novo protocolo
        """
        msg_lower = message.lower()
        
        # Reconhecer confirmação de header
        if "confirm" in msg_lower:
            print("✅ VR confirmou recebimento!")
            if self.on_confirmation:
                self.on_confirmation()
            if self.session.waiting_confirmation:
                self.session.waiting_confirmation = False
        
        # Reconhecer ações de flor
        if "left_flower" in msg_lower:
            print("🌸 VR acionou FLOR ESQUERDA")
            if self.on_flower_action:
                self.on_flower_action(ActionCommand.LEFT_FLOWER)
        
        if "right_flower" in msg_lower:
            print("🌸 VR acionou FLOR DIREITA")
            if self.on_flower_action:
                self.on_flower_action(ActionCommand.RIGHT_FLOWER)
        
        # Reconhecer confirmação de finalização
        if "confirmar_finalização" in msg_lower or "confirm_end" in msg_lower:
            print("✅ VR confirmou finalização!")
            self.session.is_active = False
    
    def set_message_callback(self, callback: Callable[[str], None]):
        """Define callback para mensagens recebidas"""
        self.on_message_received = callback
    
    def set_connection_callback(self, callback: Callable[[bool], None]):
        """Define callback para mudanças de conexão"""
        self.on_connection_changed = callback


# Classe para compatibilidade com código existente
class UDP_sender:
    """Classe de compatibilidade que mapeia para UnityCommunicator"""
    
    _communicator = UnityCommunicator()
    # simple debounce state to avoid duplicate rapid sends
    _last_sent_times = {}
    _debounce_seconds = 0.2  # ignore same action within 200 ms
    
    @classmethod
    def init_zmq_socket(cls, broadcast_duration=3.0):
        """Inicializa o sistema de comunicação"""
        return cls._communicator.start_server()
    
    @classmethod
    def stop_zmq_socket(cls):
        """Para o sistema de comunicação"""
        cls._communicator.stop_server()
    
    @classmethod
    def enviar_sinal(cls, action: str) -> bool:
        """Envia sinal de ação"""
        # debounce: avoid sending the same action repeatedly in a short window
        try:
            now = time.time()
            key = action.lower()
            last = cls._last_sent_times.get(key)
            if last is not None and (now - last) < cls._debounce_seconds:
                # skip duplicate
                print(f"Debounce: skipping duplicate action '{action}' (last sent {now-last:.3f}s ago)")
                return False
            cls._last_sent_times[key] = now
        except Exception:
            # if anything goes wrong in debounce, proceed to send (fail-open)
            pass
        if action.lower() == 'direita':
            return cls._communicator.send_hand_command('direita')
        elif action.lower() == 'esquerda':
            return cls._communicator.send_hand_command('esquerda')
        elif action.lower() == 'trigger_right':
            return cls._communicator.send_trigger_command('direita')
        elif action.lower() == 'trigger_left':
            return cls._communicator.send_trigger_command('esquerda')
        else:
            return cls._communicator.send_command(action)
    
    @classmethod
    def is_server_active(cls) -> bool:
        """Verifica se o servidor está ativo"""
        return cls._communicator.is_active
    
    @classmethod
    def restart_broadcast(cls, duration=3.0):
        """Reinicia o broadcast (não necessário na nova implementação)"""
        return True  # Broadcast é contínuo na nova implementação
    
    # Métodos legacy mantidos para compatibilidade
    @staticmethod
    def get_all_ips():
        return UnityCommunicator.get_all_ips()
    
    @staticmethod
    def get_local_ip():
        all_ips = UnityCommunicator.get_all_ips()
        for ip in all_ips:
            if ip != '127.0.0.1':
                return ip
        return all_ips[0] if all_ips else '127.0.0.1'

# Função principal para demonstração
def main():
    """Função principal para teste do sistema"""
    communicator = UnityCommunicator()
    
    def on_message(message):
        print(f"Mensagem recebida: {message}")
    
    def on_connection(connected):
        print(f"Conexão: {'Conectado' if connected else 'Desconectado'}")
    
    # Configurar callbacks
    communicator.set_message_callback(on_message)
    communicator.set_connection_callback(on_connection)
    
    # Iniciar servidor
    if not communicator.start_server():
        print("Falha ao iniciar servidor")
        return
    
    print("\n" + "="*50)
    print("Sistema de Comunicação Unity Ativo")
    print("="*50)
    print("Comandos disponíveis:")
    print("  - direita       : Controla mão direita") 
    print("  - esquerda      : Controla mão esquerda")
    print("  - trigger_right : Gatilho mão direita")
    print("  - trigger_left  : Gatilho mão esquerda")
    print("  - <comando>     : Comando personalizado")
    print("  - sair          : Encerra o programa")
    print("="*50)
    
    try:
        while True:
            comando = input("\nDigite um comando: ").strip()
            
            if comando.lower() == 'sair':
                break
            elif comando.lower() == 'direita':
                communicator.send_hand_command('direita')
            elif comando.lower() == 'esquerda':
                communicator.send_hand_command('esquerda')
            elif comando.lower() == 'trigger_right':
                communicator.send_trigger_command('direita')
            elif comando.lower() == 'trigger_left':
                communicator.send_trigger_command('esquerda')
            elif comando:
                communicator.send_command(comando)
                
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
    finally:
        communicator.stop_server()
        print("Programa encerrado")


if __name__ == '__main__':
    main()