# unity_communication.py
"""
Sistema unificado de comunicação com Unity
Substitui UDP_sender.py e udp_receiver.py por uma abordagem mais simples e robusta
Implementa protocolo de comunicação Sistema <-> VR conforme especificação
"""

import socket
import threading
import time
import zmq
from typing import Optional, List, Callable, Dict
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# PROTOCOLO DE COMUNICAÇÃO SISTEMA <-> VR
# ============================================================================

class ServerState(Enum):
    """Estados mutuamente exclusivos do servidor"""
    STOPPED = "stopped"           # Servidor parado
    RUNNING = "running"           # Servidor rodando, sem conexão VR
    CONNECTED = "connected"       # VR conectado

class SessionPhase(Enum):
    """Fases mutuamente exclusivas da sessão - máquina de estados"""
    IDLE = "idle"                 # Sem sessão ativa
    SETUP = "setup"               # Enviando dados/tarefa, aguardando confirmação
    READY = "ready"               # Confirmação recebida, aguardando trigger
    ACTIVE = "active"             # Sessão em andamento (após trigger)
    ENDING = "ending"             # Finalização em progresso, aguardando confirmação
    
    def can_transition_to(self, next_phase: 'SessionPhase') -> bool:
        """Define transições válidas da máquina de estados"""
        transitions = {
            SessionPhase.IDLE: {SessionPhase.SETUP},
            SessionPhase.SETUP: {SessionPhase.READY, SessionPhase.IDLE},  # volta se falhar
            SessionPhase.READY: {SessionPhase.ACTIVE, SessionPhase.IDLE},  # pode cancelar
            SessionPhase.ACTIVE: {SessionPhase.ENDING},
            SessionPhase.ENDING: {SessionPhase.IDLE},
        }
        return next_phase in transitions.get(self, set())

class TaskType(Enum):
    """Tipos de tarefa conforme protocolo"""
    TREINO = "Treino"
    JOGO = "Jogo"

class TriggerCommand(Enum):
    """Comandos de trigger conforme protocolo"""
    START = "Trigger"
    HAND_CLOSE = "****_HAND_CLOSE"  # será substituído por LEFT/RIGHT

class ActionCommand(Enum):
    """Comandos de ação durante sessão"""
    LEFT_FLOWER = "LEFT_FLOWER"
    RIGHT_FLOWER = "RIGHT_FLOWER"
    LEFT_HAND_CLOSE = "LEFT_HAND_CLOSE"
    RIGHT_HAND_CLOSE = "RIGHT_HAND_CLOSE"

class EndTaskCommand(Enum):
    """Comandos de finalização"""
    END_TRAINING = "Finalizar_tarefa_treino"
    END_GAME = "Finalizar_tarefa_jogo"

@dataclass
class PatientData:
    """Dados do paciente conforme protocolo"""
    nome: str
    nivel: int  # Nível de 0 a 11
    lado: str   # "Esquerdo" ou "Direito"
    
    def __post_init__(self):
        """Valida os dados após inicialização"""
        if not isinstance(self.nivel, int) or not (0 <= self.nivel <= 11):
            raise ValueError(f"Nível deve ser um número inteiro entre 0 e 11. Recebido: {self.nivel}")
        
        if self.lado not in ["Esquerdo", "Direito"]:
            raise ValueError(f"Lado deve ser 'Esquerdo' ou 'Direito'. Recebido: {self.lado}")
    
    def format_message(self) -> str:
        """Formata mensagem de dados do paciente conforme protocolo"""
        return f"Dados Paciente:\nNome: {self.nome}\nNivel: {self.nivel}\nLado: {self.lado}"

@dataclass
class SessionState:
    """Estado da sessão atual com máquina de estados"""
    phase: SessionPhase = SessionPhase.IDLE
    patient: Optional[PatientData] = None
    task_type: Optional[TaskType] = None
    
    def transition_to(self, next_phase: SessionPhase) -> bool:
        """
        Transiciona para próxima fase se for válido
        Retorna True se transição bem-sucedida, False caso contrário
        """
        if not self.phase.can_transition_to(next_phase):
            return False
        self.phase = next_phase
        return True
    
    def reset(self):
        """Reseta sessão para estado inicial"""
        self.phase = SessionPhase.IDLE
        self.patient = None
        self.task_type = None

# ============================================================================
# CLASSE PRINCIPAL DE COMUNICAÇÃO
# ============================================================================

class UnityCommunicator:
    """
    Classe unificada para comunicação com Unity usando TCP + ZMQ
    Implementa protocolo completo Sistema <-> VR conforme especificação
    
    Fluxo do Protocolo:
    1. Broadcast UDP com header "Confirm"
    2. Envio de dados do paciente
    3. Envio de tarefa (Treino/Jogo)
    4. Trigger para iniciar
    5. Comandos durante sessão (HAND_CLOSE, FLOWER, etc)
    6. Finalização de tarefa
    7. Confirmação de finalização
    """
    
    # Configurações
    UDP_PORT = 12346      # porta para broadcast de IPs
    TCP_PORT = 12345      # porta para servidor TCP Unity
    ZMQ_PORT = 5555       # porta para ZMQ publisher
    BROADCAST_INTERVAL = 1.0
    BUFFER_SIZE = 4096
    CONFIRM_HEADER = "Confirm"  # Header para broadcast conforme protocolo
    
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
        
        # Estado do servidor (máquina de estados)
        self.server_state = ServerState.STOPPED
        self.tcp_connected = False
        
        # Estado da sessão (máquina de estados)
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
        self.on_confirmation_received: Optional[Callable[[], None]] = None
    
    # ========================================================================
    # HELPERS DE TRANSIÇÃO DE ESTADO (MÁQUINA DE ESTADOS CENTRALIZADA)
    # ========================================================================
    
    def _transition_server_state(self, next_state: ServerState) -> bool:
        """Transiciona servidor para novo estado com validação"""
        if self.server_state == next_state:
            return True  # Já nesse estado
        self.server_state = next_state
        return True
    
    def _transition_session_phase(self, next_phase: SessionPhase) -> bool:
        """
        Transiciona sessão para nova fase com validação.
        Retorna False se transição for inválida.
        """
        if not self.session.transition_to(next_phase):
            return False
        return True
    
    def _is_server_ready_for_session(self) -> bool:
        """Verifica se servidor está pronto para iniciar sessão"""
        return (
            self.server_state == ServerState.CONNECTED and
            self.tcp_connected and
            self.session.phase == SessionPhase.IDLE
        )
    
    def _is_session_waiting_trigger(self) -> bool:
        """Verifica se sessão aguarda trigger"""
        return self.session.phase == SessionPhase.READY
    
    def _is_session_active_for_commands(self) -> bool:
        """Verifica se sessão está ativa para aceitar comandos"""
        return self.session.phase == SessionPhase.ACTIVE
    
    def _is_server_operational(self) -> bool:
        """Verifica se servidor está operacional (rodando)"""
        return self.server_state in [ServerState.RUNNING, ServerState.CONNECTED]
    
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
        if self.server_state != ServerState.STOPPED:
            print("Servidor já está rodando")
            return True
            
        try:
            # Inicializar ZMQ
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PUB)
            self.zmq_socket.bind(f"tcp://*:{self.ZMQ_PORT}")
            
            # Reset do evento de parada
            self.stop_event.clear()
            
            # Transicionar servidor para estado RUNNING
            self._transition_server_state(ServerState.RUNNING)
            
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
            
            print(f"Servidor iniciado - ZMQ: {self.ZMQ_PORT}, TCP: {self.TCP_PORT}, UDP: {self.UDP_PORT}")
            return True
            
        except Exception as e:
            print(f"Erro ao iniciar servidor: {e}")
            self.stop_server()
            return False
    
    def stop_server(self):
        """Para o servidor e limpa recursos"""
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
        
        # Resetar sessão se ativa
        if self.session.phase != SessionPhase.IDLE:
            self.session.reset()
        
        # Transicionar servidor para estado STOPPED
        self._transition_server_state(ServerState.STOPPED)
        
        # Atualizar conexão TCP
        if self.tcp_connected:
            self.tcp_connected = False
            if self.on_connection_changed:
                self.on_connection_changed(False)
        
        print("Servidor parado e recursos limpos")
    
    # ========================================================================
    # MÉTODOS DO PROTOCOLO SISTEMA <-> VR
    # ========================================================================
    
    def start_session(self, patient_data: PatientData, task_type: TaskType) -> bool:
        """
        Inicia uma sessão completa seguindo o protocolo:
        1. Envia dados do paciente
        2. Envia tipo de tarefa
        3. Aguarda confirmação do VR
        
        Retorna True se a sessão foi iniciada com sucesso
        """
        # Validar precondições usando helpers de estado
        if not self._is_server_ready_for_session():
            if self.server_state == ServerState.STOPPED:
                print("❌ Erro: Servidor não está ativo")
            elif not self.tcp_connected:
                print("❌ Erro: VR não está conectado")
            elif self.session.phase != SessionPhase.IDLE:
                print("❌ Erro: Já existe uma sessão em progresso")
            return False
        
        print("\n" + "="*60)
        print("🚀 INICIANDO SESSÃO VR - PROTOCOLO SISTEMA <-> VR")
        print("="*60)
        
        # Transicionar para SETUP
        if not self._transition_session_phase(SessionPhase.SETUP):
            print("❌ Erro: Não foi possível transicionar para SETUP")
            return False
        
        # Passo 1: Enviar dados do paciente
        print("\n📤 [1/3] Enviando dados do paciente...")
        patient_msg = patient_data.format_message()
        if not self._send_protocol_message(patient_msg):
            print("❌ Falha ao enviar dados do paciente")
            self.session.transition_to(SessionPhase.IDLE)
            return False
        print(f"✅ Dados enviados:\n{patient_msg}")
        time.sleep(0.5)
        
        # Passo 2: Enviar tipo de tarefa
        print(f"\n📤 [2/3] Enviando tarefa: {task_type.value}")
        task_msg = f'Tarefa:\n"{task_type.value}"'
        if not self._send_protocol_message(task_msg):
            print("❌ Falha ao enviar tarefa")
            self.session.transition_to(SessionPhase.IDLE)
            return False
        print(f"✅ Tarefa enviada: {task_type.value}")
        time.sleep(0.5)
        
        # Passo 3: Aguardar confirmação (implementado via callback)
        print("\n⏳ [3/3] Aguardando confirmação do VR...")
        self.session.patient = patient_data
        self.session.task_type = task_type
        
        print("="*60)
        print("✅ Sessão configurada - Aguardando confirmação do VR")
        print("="*60 + "\n")
        
        return True
    
    def send_trigger(self) -> bool:
        """
        Envia comando de trigger para iniciar a tarefa no VR
        Deve ser chamado após receber confirmação do VR
        """
        if not self._is_session_waiting_trigger():
            print("❌ Erro: Sessão não está aguardando trigger")
            return False
        
        print("\n🎯 Enviando TRIGGER para iniciar tarefa...")
        if self._send_protocol_message(TriggerCommand.START.value):
            if self._transition_session_phase(SessionPhase.ACTIVE):
                print("✅ Trigger enviado - Tarefa iniciada no VR")
                return True
        
        print("❌ Falha ao enviar trigger")
        return False
    
    def send_hand_close(self, side: str) -> bool:
        """
        Envia comando de fechar mão (LEFT ou RIGHT)
        Args:
            side: "left", "right", "esquerda" ou "direita"
        """
        if not self._is_session_active_for_commands():
            print("❌ Erro: Sessão não está ativa para aceitar comandos")
            return False
        
        # Normalizar entrada
        side_normalized = side.lower()
        if side_normalized in ['left', 'esquerda']:
            command = ActionCommand.LEFT_HAND_CLOSE.value
            side_label = "ESQUERDA"
        elif side_normalized in ['right', 'direita']:
            command = ActionCommand.RIGHT_HAND_CLOSE.value
            side_label = "DIREITA"
        else:
            print(f"❌ Erro: Lado inválido '{side}'")
            return False
        
        print(f"\n✊ Enviando comando: Fechar mão {side_label}")
        if self._send_protocol_message(command):
            print(f"✅ Comando enviado: {command}")
            return True
        
        print(f"❌ Falha ao enviar comando")
        return False
    
    def send_flower_action(self, side: str) -> bool:
        """
        Envia comando de ação de flor (LEFT_FLOWER ou RIGHT_FLOWER)
        Args:
            side: "left", "right", "esquerda" ou "direita"
        """
        if not self._is_session_active_for_commands():
            print("❌ Erro: Sessão não está ativa para aceitar comandos")
            return False
        
        # Normalizar entrada
        side_normalized = side.lower()
        if side_normalized in ['left', 'esquerda']:
            command = ActionCommand.LEFT_FLOWER.value
            side_label = "ESQUERDA"
        elif side_normalized in ['right', 'direita']:
            command = ActionCommand.RIGHT_FLOWER.value
            side_label = "DIREITA"
        else:
            print(f"❌ Erro: Lado inválido '{side}'")
            return False
        
        print(f"\n🌸 Enviando comando: Flor {side_label}")
        if self._send_protocol_message(command):
            print(f"✅ Comando enviado: {command}")
            return True
        
        print(f"❌ Falha ao enviar comando")
        return False
    
    def end_session(self, message: Optional[str] = None) -> bool:
        """
        Finaliza a sessão atual enviando comando de finalização
        Args:
            message: Mensagem opcional para enviar junto com a finalização
                    Se None, usa mensagem padrão de encorajamento
        """
        if not self._is_session_active_for_commands():
            print("❌ Erro: Sessão não está ativa para finalizar")
            return False
        
        print("\n" + "="*60)
        print("🏁 FINALIZANDO SESSÃO VR")
        print("="*60)
        
        # Transicionar para ENDING
        if not self._transition_session_phase(SessionPhase.ENDING):
            print("❌ Erro: Não foi possível transicionar para ENDING")
            return False
        
        # Determinar comando de finalização baseado no tipo de tarefa
        if self.session.task_type == TaskType.TREINO:
            end_command = EndTaskCommand.END_TRAINING.value
            task_name = "TREINO"
        elif self.session.task_type == TaskType.JOGO:
            end_command = EndTaskCommand.END_GAME.value
            task_name = "JOGO"
        else:
            print("❌ Erro: Tipo de tarefa desconhecido")
            self.session.transition_to(SessionPhase.ACTIVE)  # Voltar para ACTIVE
            return False
        
        # Usar mensagem padrão se nenhuma for fornecida
        if message is None:
            message = "Parabens voce esta mandando muito bem"
        
        # Enviar comando de finalização
        print(f"\n📤 Enviando comando de finalização: {task_name}")
        final_msg = f"{end_command}\nEND_TASK, \"{message}\""
        
        if self._send_protocol_message(final_msg):
            print(f"✅ Comando de finalização enviado")
            print("\n⏳ Aguardando confirmação de finalização do VR...")
            print("="*60 + "\n")
            return True
        
        print("❌ Falha ao enviar comando de finalização")
        self.session.transition_to(SessionPhase.ACTIVE)  # Voltar para ACTIVE se falhar
        return False
    
    def _send_protocol_message(self, message: str) -> bool:
        """
        Envia mensagem seguindo o protocolo (via ZMQ e TCP)
        Método interno usado pelos métodos públicos do protocolo
        """
        if not self._is_server_operational():
            return False
        
        success = False
        
        # Enviar via ZMQ
        if self.zmq_socket:
            try:
                self.zmq_socket.send_string(message)
                success = True
            except Exception as e:
                print(f"⚠️ [ZMQ] Erro ao enviar: {e}")
        
        # Enviar via TCP (prioritário)
        if self.tcp_connected and self.tcp_connection:
            try:
                encoded_msg = message + '\n'
                self.tcp_connection.sendall(encoded_msg.encode('utf-8'))
                success = True
            except Exception as e:
                print(f"⚠️ [TCP] Erro ao enviar: {e}")
                self.tcp_connected = False
                if self.on_connection_changed:
                    self.on_connection_changed(False)
        
        return success
    
    # ========================================================================
    # MÉTODOS LEGADOS (mantidos para compatibilidade)
    # ========================================================================
    
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
    
    def _broadcast_ips(self):
        """
        Thread para broadcast dos IPs via UDP com header "Confirm"
        Conforme protocolo: Broadcast UDP com header "Confirm"
        """
        ips = self.get_all_ips()
        # Formatar mensagem com header "Confirm" conforme protocolo
        message_content = ','.join(ips)
        message = f"{self.CONFIRM_HEADER}:{message_content}".encode('utf-8')
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        print(f"[UDP] Iniciando broadcast com header '{self.CONFIRM_HEADER}': {ips}")
        
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
                    
                    # Transicionar servidor para CONNECTED quando VR conecta
                    self._transition_server_state(ServerState.CONNECTED)
                    
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
        Processa mensagens do VR incluindo confirmações
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
                    
                    # Processar mensagem recebida
                    self._process_vr_message(message)
                    
                    # Callback genérico
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
            
            # Transicionar servidor de volta para RUNNING quando VR desconecta
            if self.server_state == ServerState.CONNECTED:
                self._transition_server_state(ServerState.RUNNING)
            
            if self.on_connection_changed:
                self.on_connection_changed(False)
            
            print("[TCP] Conexão encerrada")
    
    def _process_vr_message(self, message: str):
        """
        Processa mensagens recebidas do VR conforme protocolo
        Gerencia transições de estado automáticas via máquina de estados
        """
        message_lower = message.lower().strip()
        
        # Detectar confirmação de inicialização (VR recebeu dados/tarefa)
        if "confirm" in message_lower and self.session.phase == SessionPhase.SETUP:
            print("\n" + "="*60)
            print("✅ CONFIRMAÇÃO RECEBIDA DO VR")
            print("="*60)
            print("📥 VR está pronto e confirmou recebimento dos dados")
            print("🎯 Aguardando trigger para iniciar a tarefa...")
            print("="*60 + "\n")
            
            # Transicionar para READY
            self._transition_session_phase(SessionPhase.READY)
            
            if self.on_confirmation_received:
                self.on_confirmation_received()
        
        # Detectar confirmação de finalização
        elif ("finalizar" in message_lower or "end" in message_lower) and self.session.phase == SessionPhase.ENDING:
            print("\n" + "="*60)
            print("✅ CONFIRMAÇÃO DE FINALIZAÇÃO RECEBIDA")
            print("="*60)
            print("📥 VR confirmou finalização da sessão")
            
            # Transicionar para IDLE
            self._transition_session_phase(SessionPhase.IDLE)
            
            print("🏁 Sessão encerrada com sucesso")
            print("="*60 + "\n")
        
        # Log de outras mensagens
        else:
            print(f"[TCP] 📨 Mensagem do VR: {message}")
    
    def set_message_callback(self, callback: Callable[[str], None]):
        """Define callback para mensagens recebidas"""
        self.on_message_received = callback
    
    def set_connection_callback(self, callback: Callable[[bool], None]):
        """Define callback para mudanças de conexão"""
        self.on_connection_changed = callback
    
    def set_confirmation_callback(self, callback: Callable[[], None]):
        """Define callback para confirmações do VR"""
        self.on_confirmation_received = callback


# ============================================================================
# CLASSES DE COMPATIBILIDADE COM CÓDIGO LEGADO
# ============================================================================

class UDP_sender:
    """
    Classe de compatibilidade que mapeia para UnityCommunicator
    Mantém API legada enquanto usa o novo protocolo internamente
    """
    
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
        """
        Envia sinal de ação (método legado)
        Agora usa os métodos do protocolo quando possível
        """
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
        
        # Mapear ações legadas para novos métodos do protocolo
        action_lower = action.lower()
        
        if action_lower in ['direita', 'right']:
            return cls._communicator.send_hand_close('direita')
        elif action_lower in ['esquerda', 'left']:
            return cls._communicator.send_hand_close('esquerda')
        elif action_lower in ['trigger_right', 'trigger']:
            return cls._communicator.send_trigger()
        elif action_lower == 'trigger_left':
            return cls._communicator.send_trigger()
        elif action_lower == 'left_flower':
            return cls._communicator.send_flower_action('esquerda')
        elif action_lower == 'right_flower':
            return cls._communicator.send_flower_action('direita')
        else:
            # Comando genérico - usar método legado
            return cls._communicator.send_command(action)
    
    @classmethod
    def is_server_active(cls) -> bool:
        """Verifica se o servidor está ativo"""
        return cls._communicator.server_state != ServerState.STOPPED
    
    @classmethod
    def restart_broadcast(cls, duration=3.0):
        """Reinicia o broadcast (não necessário na nova implementação)"""
        return True  # Broadcast é contínuo na nova implementação
    
    # Métodos para usar o protocolo completo
    @classmethod
    def start_vr_session(cls, patient_name: str, level: int, affected_side: str, task: str) -> bool:
        """
        Inicia sessão VR usando o protocolo completo
        Args:
            patient_name: Nome do paciente
            level: Nível do paciente (0-11)
            affected_side: "Esquerdo" ou "Direito"
            task: "Treino" ou "Jogo"
        """
        patient_data = PatientData(
            nome=patient_name,
            nivel=level,
            lado=affected_side
        )
        
        task_type = TaskType.TREINO if task.lower() == "treino" else TaskType.JOGO
        
        return cls._communicator.start_session(patient_data, task_type)
    
    @classmethod
    def end_vr_session(cls, message: Optional[str] = None) -> bool:
        """Finaliza sessão VR usando o protocolo"""
        return cls._communicator.end_session(message)
    
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

# ============================================================================
# FUNÇÃO PRINCIPAL PARA DEMONSTRAÇÃO E TESTES
# ============================================================================

def main():
    """
    Função principal para teste do sistema
    Demonstra o uso completo do protocolo Sistema <-> VR
    """
    communicator = UnityCommunicator()
    
    def on_message(message):
        print(f"📨 Callback - Mensagem recebida: {message}")
    
    def on_connection(connected):
        status = "🟢 Conectado" if connected else "🔴 Desconectado"
        print(f"🔌 Callback - Conexão VR: {status}")
    
    def on_confirmation():
        print("✅ Callback - Confirmação recebida!")
    
    # Configurar callbacks
    communicator.set_message_callback(on_message)
    communicator.set_connection_callback(on_connection)
    communicator.set_confirmation_callback(on_confirmation)
    
    # Iniciar servidor
    if not communicator.start_server():
        print("❌ Falha ao iniciar servidor")
        return
    
    print("\n" + "="*70)
    print(" SISTEMA DE COMUNICAÇÃO UNITY - PROTOCOLO COMPLETO IMPLEMENTADO")
    print("="*70)
    print("\n📋 COMANDOS DO PROTOCOLO:")
    print("="*70)
    print("\n🚀 Iniciar Sessão:")
    print("  iniciar <nome> <nivel> <lado> <tarefa>")
    print("  Exemplo: iniciar João 5 Direito Treino")
    print("  (nivel: 0-11, lado: Esquerdo/Direito, tarefa: Treino/Jogo)")
    print("\n🎯 Durante Sessão:")
    print("  trigger          - Envia trigger para iniciar tarefa")
    print("  fechar <lado>    - Fecha mão (esquerda/direita)")
    print("  flor <lado>      - Ação de flor (esquerda/direita)")
    print("\n🏁 Finalizar:")
    print("  fim [mensagem]   - Finaliza sessão (mensagem opcional)")
    print("\n💡 Comandos Gerais:")
    print("  status           - Mostra estado atual")
    print("  sair             - Encerra programa")
    print("="*70 + "\n")
    
    try:
        while True:
            comando = input("\n> ").strip()
            
            if not comando:
                continue
            
            parts = comando.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # Comando: sair
            if cmd == 'sair':
                print("\n👋 Encerrando...")
                break
            
            # Comando: status
            elif cmd == 'status':
                print("\n📊 ESTADO DO SISTEMA:")
                print(f"  Servidor ativo: {communicator.is_active}")
                print(f"  VR conectado: {communicator.tcp_connected}")
                print(f"  Sessão ativa: {communicator.session.is_active}")
                print(f"  Aguardando confirmação: {communicator.session.waiting_confirmation}")
                if communicator.session.patient:
                    print(f"  Paciente: {communicator.session.patient.nome}")
                    print(f"  Nível: {communicator.session.patient.nivel}")
                    print(f"  Lado: {communicator.session.patient.lado}")
                if communicator.session.task_type:
                    print(f"  Tarefa: {communicator.session.task_type.value}")
            
            # Comando: iniciar sessão
            elif cmd == 'iniciar':
                try:
                    parts = args.split()
                    if len(parts) < 4:
                        print("❌ Uso: iniciar <nome> <nivel> <lado> <tarefa>")
                        print("   Exemplo: iniciar João 5 Direito Treino")
                        print("   Nível: 0-11")
                        continue
                    
                    nome = parts[0]
                    try:
                        nivel = int(parts[1])
                        if not (0 <= nivel <= 11):
                            print("❌ Nível deve ser um número entre 0 e 11")
                            continue
                    except ValueError:
                        print("❌ Nível deve ser um número inteiro (0-11)")
                        continue
                    
                    lado = parts[2].capitalize()
                    tarefa = parts[3].capitalize()
                    
                    if lado not in ["Esquerdo", "Direito"]:
                        print("❌ Lado deve ser 'Esquerdo' ou 'Direito'")
                        continue
                    
                    if tarefa not in ["Treino", "Jogo"]:
                        print("❌ Tarefa deve ser 'Treino' ou 'Jogo'")
                        continue
                    
                    patient_data = PatientData(nome=nome, nivel=nivel, lado=lado)
                    task_type = TaskType.TREINO if tarefa == "Treino" else TaskType.JOGO
                    
                    communicator.start_session(patient_data, task_type)
                    
                except Exception as e:
                    print(f"❌ Erro ao iniciar sessão: {e}")
            
            # Comando: trigger
            elif cmd == 'trigger':
                communicator.send_trigger()
            
            # Comando: fechar mão
            elif cmd == 'fechar':
                if not args:
                    print("❌ Especifique o lado: fechar esquerda|direita")
                else:
                    communicator.send_hand_close(args.lower())
            
            # Comando: flor
            elif cmd == 'flor':
                if not args:
                    print("❌ Especifique o lado: flor esquerda|direita")
                else:
                    communicator.send_flower_action(args.lower())
            
            # Comando: fim
            elif cmd == 'fim':
                message = args if args else None
                communicator.end_session(message)
            
            # Comandos legados (compatibilidade)
            elif cmd == 'direita':
                communicator.send_hand_close('direita')
            elif cmd == 'esquerda':
                communicator.send_hand_close('esquerda')
            
            # Comando desconhecido
            else:
                print(f"⚠️  Comando desconhecido: '{cmd}'")
                print("   Digite 'sair' para ver todos os comandos disponíveis")
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
    finally:
        communicator.stop_server()
        print("\n✅ Programa encerrado\n")


if __name__ == '__main__':
    main()