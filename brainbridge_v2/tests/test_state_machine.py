"""
Testes da máquina de estados para a comunicação com Unity
Valida transições válidas/inválidas e interdependências
"""

import sys
sys.path.insert(0, 'c:\\Users\\Chari\\Documents\\dev\\BrainBridge')

from brainbridge_v2.communication.unity import (
    SessionPhase, ServerState, SessionState, PatientData, 
    TaskType, UnityCommunicator
)


def test_session_phase_transitions():
    """Testa se as transições de fase são válidas"""
    print("\n" + "="*70)
    print("🧪 TESTE 1: Transições de fase válidas")
    print("="*70)
    
    session = SessionState()
    
    # Teste 1: IDLE -> SETUP
    assert session.phase == SessionPhase.IDLE, f"Inicial deve ser IDLE, got {session.phase}"
    assert session.transition_to(SessionPhase.SETUP), "Deve transicionar IDLE -> SETUP"
    assert session.phase == SessionPhase.SETUP, f"Deve estar em SETUP, got {session.phase}"
    print("✅ IDLE -> SETUP: OK")
    
    # Teste 2: SETUP -> READY
    assert session.transition_to(SessionPhase.READY), "Deve transicionar SETUP -> READY"
    assert session.phase == SessionPhase.READY, f"Deve estar em READY, got {session.phase}"
    print("✅ SETUP -> READY: OK")
    
    # Teste 3: READY -> ACTIVE
    assert session.transition_to(SessionPhase.ACTIVE), "Deve transicionar READY -> ACTIVE"
    assert session.phase == SessionPhase.ACTIVE, f"Deve estar em ACTIVE, got {session.phase}"
    print("✅ READY -> ACTIVE: OK")
    
    # Teste 4: ACTIVE -> ENDING
    assert session.transition_to(SessionPhase.ENDING), "Deve transicionar ACTIVE -> ENDING"
    assert session.phase == SessionPhase.ENDING, f"Deve estar em ENDING, got {session.phase}"
    print("✅ ACTIVE -> ENDING: OK")
    
    # Teste 5: ENDING -> IDLE
    assert session.transition_to(SessionPhase.IDLE), "Deve transicionar ENDING -> IDLE"
    assert session.phase == SessionPhase.IDLE, f"Deve estar em IDLE, got {session.phase}"
    print("✅ ENDING -> IDLE: OK")


def test_invalid_transitions():
    """Testa se transições inválidas são bloqueadas"""
    print("\n" + "="*70)
    print("🧪 TESTE 2: Transições inválidas devem ser bloqueadas")
    print("="*70)
    
    session = SessionState()
    
    # IDLE não pode ir pra READY diretamente
    assert not session.transition_to(SessionPhase.READY), "IDLE -> READY deve ser inválido"
    print("✅ IDLE -> READY bloqueado (correto)")
    
    # IDLE não pode ir pra ACTIVE
    assert not session.transition_to(SessionPhase.ACTIVE), "IDLE -> ACTIVE deve ser inválido"
    print("✅ IDLE -> ACTIVE bloqueado (correto)")
    
    # Transicionar pra SETUP
    session.transition_to(SessionPhase.SETUP)
    
    # SETUP não pode ir pra ACTIVE (precisa ir pra READY primeiro)
    assert not session.transition_to(SessionPhase.ACTIVE), "SETUP -> ACTIVE deve ser inválido"
    print("✅ SETUP -> ACTIVE bloqueado (correto)")
    
    # SETUP não pode ir pra ENDING
    assert not session.transition_to(SessionPhase.ENDING), "SETUP -> ENDING deve ser inválido"
    print("✅ SETUP -> ENDING bloqueado (correto)")


def test_reset():
    """Testa se reset volta ao estado inicial"""
    print("\n" + "="*70)
    print("🧪 TESTE 3: Reset limpa o estado")
    print("="*70)
    
    session = SessionState()
    patient = PatientData(nome="João", nivel=5, lado="Direito")
    
    # Preencher dados
    session.patient = patient
    session.task_type = TaskType.TREINO
    session.transition_to(SessionPhase.SETUP)
    
    # Validar estado cheio
    assert session.patient is not None, "Deve ter paciente"
    assert session.task_type == TaskType.TREINO, "Deve ter tarefa"
    assert session.phase == SessionPhase.SETUP, "Deve estar em SETUP"
    print("✅ Estado preenchido corretamente")
    
    # Reset
    session.reset()
    
    # Validar reset
    assert session.patient is None, "Paciente deve ser None após reset"
    assert session.task_type is None, "Tarefa deve ser None após reset"
    assert session.phase == SessionPhase.IDLE, "Fase deve ser IDLE após reset"
    print("✅ Reset limpou o estado corretamente")


def test_communicator_state_transitions():
    """Testa transições de estado do comunicador"""
    print("\n" + "="*70)
    print("🧪 TESTE 4: Transições de estado do servidor")
    print("="*70)
    
    comm = UnityCommunicator()
    
    # Initial state
    assert comm.server_state == ServerState.STOPPED, f"Inicial deve ser STOPPED, got {comm.server_state}"
    print("✅ Estado inicial: STOPPED")
    
    # Iniciar servidor
    comm.start_server()
    assert comm.server_state == ServerState.RUNNING, f"Após start deve ser RUNNING, got {comm.server_state}"
    print("✅ Após start_server(): RUNNING")
    
    # Parar servidor
    comm.stop_server()
    assert comm.server_state == ServerState.STOPPED, f"Após stop deve ser STOPPED, got {comm.server_state}"
    print("✅ Após stop_server(): STOPPED")


def test_helpers_state_queries():
    """Testa métodos helpers de queries de estado"""
    print("\n" + "="*70)
    print("🧪 TESTE 5: Helpers de query de estado")
    print("="*70)
    
    comm = UnityCommunicator()
    
    # Servidor não operational
    assert not comm._is_server_operational(), "Servidor parado não é operational"
    print("✅ _is_server_operational() = False (servidor parado)")
    
    # Iniciar
    comm.start_server()
    assert comm._is_server_operational(), "Servidor rodando é operational"
    print("✅ _is_server_operational() = True (servidor rodando)")
    
    # Não pronto para sessão sem VR conectado
    assert not comm._is_server_ready_for_session(), "Servidor não pronto sem VR"
    print("✅ _is_server_ready_for_session() = False (sem VR)")
    
    # Sessão não aguarda trigger quando em IDLE
    assert not comm._is_session_waiting_trigger(), "Não aguarda trigger em IDLE"
    print("✅ _is_session_waiting_trigger() = False (IDLE)")
    
    # Sessão não ativa para comandos em IDLE
    assert not comm._is_session_active_for_commands(), "Não ativa em IDLE"
    print("✅ _is_session_active_for_commands() = False (IDLE)")
    
    # Limpar
    comm.stop_server()


def test_no_cross_dependencies():
    """Testa se removemos interdependências entre variáveis"""
    print("\n" + "="*70)
    print("🧪 TESTE 6: Sem interdependências cruzadas")
    print("="*70)
    
    comm = UnityCommunicator()
    session = comm.session
    
    # SessionPhase é a única fonte de verdade para estado de sessão
    # Não deve haver is_active nem waiting_confirmation
    assert not hasattr(session, 'is_active'), "SessionState não deve ter is_active"
    assert not hasattr(session, 'waiting_confirmation'), "SessionState não deve ter waiting_confirmation"
    print("✅ SessionState não tem is_active ou waiting_confirmation")
    
    # ServerState é a única fonte de verdade para servidor
    # Pode ter tcp_connected como detalhe de implementação, mas não é a verdade
    assert hasattr(comm, 'server_state'), "Deve ter server_state"
    assert isinstance(comm.server_state, ServerState), "server_state deve ser ServerState enum"
    print("✅ UnityCommunicator usa ServerState enum (fonte única de verdade)")
    
    # Session.phase é a única fonte para se a sessão está em qual estágio
    assert hasattr(session, 'phase'), "Deve ter phase"
    assert isinstance(session.phase, SessionPhase), "phase deve ser SessionPhase enum"
    print("✅ SessionState usa SessionPhase enum (fonte única de verdade)")
    
    print("\n💡 Arquitetura: Estados são mutuamente exclusivos e não podem ocorrer em paralelo")
    print("   - Reduz bugs causados por verificações de múltiplas variáveis")
    print("   - Transições são explícitas e validadas")


def main():
    """Executa todos os testes"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  TESTE: MÁQUINA DE ESTADOS DA COMUNICAÇÃO UNITY".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        test_session_phase_transitions()
        test_invalid_transitions()
        test_reset()
        test_communicator_state_transitions()
        test_helpers_state_queries()
        test_no_cross_dependencies()
        
        print("\n" + "="*70)
        print("✅ TODOS OS TESTES PASSARAM!")
        print("="*70)
        print("\n📊 Resumo da refatoração:")
        print("   - SessionPhase: IDLE -> SETUP -> READY -> ACTIVE -> ENDING -> IDLE")
        print("   - ServerState: STOPPED -> RUNNING <-> CONNECTED")
        print("   - Transições são validadas (máquina de estados)")
        print("   - Sem interdependências: cada estado é independente")
        print("   - Helpers centralizam lógica de verificação")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TESTE FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
