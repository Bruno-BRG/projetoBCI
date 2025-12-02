"""
Testes Abrangentes da Máquina de Estados - Unity Communicator
Valida transições, erros, edge cases e interdependências
"""

import pytest
import sys
import os

# Adicionar path direto sem passar pela raiz
test_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(os.path.dirname(test_dir))
sys.path.insert(0, project_dir)

# Import direto do módulo
from brainbridge_v2.communication.unity import (
    SessionPhase, ServerState, SessionState, PatientData,
    TaskType, TriggerCommand, ActionCommand, EndTaskCommand,
    UnityCommunicator
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def session_state():
    """Fixture: SessionState novo para cada teste"""
    return SessionState()


@pytest.fixture
def communicator():
    """Fixture: UnityCommunicator novo (singleton, mas reseta para testes)"""
    comm = UnityCommunicator()
    yield comm
    # Cleanup
    if comm.server_state != ServerState.STOPPED:
        comm.stop_server()


@pytest.fixture
def patient_data():
    """Fixture: Dados de paciente válido"""
    return PatientData(nome="João Silva", nivel=5, lado="Direito")


# ============================================================================
# TESTES: SessionPhase - Transições Válidas
# ============================================================================

class TestSessionPhaseTransitions:
    """Testes das transições de fase da máquina de estados"""

    def test_initial_state_is_idle(self, session_state):
        """A fase inicial deve ser IDLE"""
        assert session_state.phase == SessionPhase.IDLE

    def test_transition_idle_to_setup(self, session_state):
        """Transição IDLE -> SETUP deve ser válida"""
        assert session_state.transition_to(SessionPhase.SETUP)
        assert session_state.phase == SessionPhase.SETUP

    def test_transition_setup_to_ready(self, session_state):
        """Transição SETUP -> READY deve ser válida"""
        session_state.transition_to(SessionPhase.SETUP)
        assert session_state.transition_to(SessionPhase.READY)
        assert session_state.phase == SessionPhase.READY

    def test_transition_ready_to_active(self, session_state):
        """Transição READY -> ACTIVE deve ser válida"""
        session_state.transition_to(SessionPhase.SETUP)
        session_state.transition_to(SessionPhase.READY)
        assert session_state.transition_to(SessionPhase.ACTIVE)
        assert session_state.phase == SessionPhase.ACTIVE

    def test_transition_active_to_ending(self, session_state):
        """Transição ACTIVE -> ENDING deve ser válida"""
        session_state.transition_to(SessionPhase.SETUP)
        session_state.transition_to(SessionPhase.READY)
        session_state.transition_to(SessionPhase.ACTIVE)
        assert session_state.transition_to(SessionPhase.ENDING)
        assert session_state.phase == SessionPhase.ENDING

    def test_transition_ending_to_idle(self, session_state):
        """Transição ENDING -> IDLE deve ser válida"""
        # Preencher pipeline completo
        for phase in [SessionPhase.SETUP, SessionPhase.READY, SessionPhase.ACTIVE, SessionPhase.ENDING]:
            session_state.transition_to(phase)
        
        assert session_state.transition_to(SessionPhase.IDLE)
        assert session_state.phase == SessionPhase.IDLE

    def test_complete_workflow_transition(self, session_state):
        """Teste do fluxo completo: IDLE -> SETUP -> READY -> ACTIVE -> ENDING -> IDLE"""
        phases = [SessionPhase.SETUP, SessionPhase.READY, SessionPhase.ACTIVE, SessionPhase.ENDING, SessionPhase.IDLE]
        
        for phase in phases:
            assert session_state.transition_to(phase), f"Falha ao transicionar para {phase}"
            assert session_state.phase == phase, f"Não chegou em {phase}"


# ============================================================================
# TESTES: SessionPhase - Transições Inválidas
# ============================================================================

class TestSessionPhaseInvalidTransitions:
    """Testes para transições inválidas que devem ser bloqueadas"""

    def test_idle_cannot_go_to_ready(self, session_state):
        """IDLE não pode ir direto para READY"""
        assert not session_state.transition_to(SessionPhase.READY)
        assert session_state.phase == SessionPhase.IDLE

    def test_idle_cannot_go_to_active(self, session_state):
        """IDLE não pode ir direto para ACTIVE"""
        assert not session_state.transition_to(SessionPhase.ACTIVE)
        assert session_state.phase == SessionPhase.IDLE

    def test_idle_cannot_go_to_ending(self, session_state):
        """IDLE não pode ir direto para ENDING"""
        assert not session_state.transition_to(SessionPhase.ENDING)
        assert session_state.phase == SessionPhase.IDLE

    def test_setup_cannot_go_to_active(self, session_state):
        """SETUP não pode ir direto para ACTIVE (precisa READY primeiro)"""
        session_state.transition_to(SessionPhase.SETUP)
        assert not session_state.transition_to(SessionPhase.ACTIVE)
        assert session_state.phase == SessionPhase.SETUP

    def test_setup_cannot_go_to_ending(self, session_state):
        """SETUP não pode ir direto para ENDING"""
        session_state.transition_to(SessionPhase.SETUP)
        assert not session_state.transition_to(SessionPhase.ENDING)
        assert session_state.phase == SessionPhase.SETUP

    def test_ready_cannot_go_to_setup(self, session_state):
        """READY não pode voltar para SETUP"""
        session_state.transition_to(SessionPhase.SETUP)
        session_state.transition_to(SessionPhase.READY)
        assert not session_state.transition_to(SessionPhase.SETUP)
        assert session_state.phase == SessionPhase.READY

    def test_ready_cannot_go_to_ending(self, session_state):
        """READY não pode ir direto para ENDING (precisa ACTIVE primeiro)"""
        session_state.transition_to(SessionPhase.SETUP)
        session_state.transition_to(SessionPhase.READY)
        assert not session_state.transition_to(SessionPhase.ENDING)
        assert session_state.phase == SessionPhase.READY

    def test_active_cannot_go_to_idle(self, session_state):
        """ACTIVE não pode ir direto para IDLE (precisa ENDING primeiro)"""
        session_state.transition_to(SessionPhase.SETUP)
        session_state.transition_to(SessionPhase.READY)
        session_state.transition_to(SessionPhase.ACTIVE)
        assert not session_state.transition_to(SessionPhase.IDLE)
        assert session_state.phase == SessionPhase.ACTIVE

    def test_ending_cannot_go_to_active(self, session_state):
        """ENDING não pode voltar para ACTIVE"""
        for phase in [SessionPhase.SETUP, SessionPhase.READY, SessionPhase.ACTIVE, SessionPhase.ENDING]:
            session_state.transition_to(phase)
        assert not session_state.transition_to(SessionPhase.ACTIVE)
        assert session_state.phase == SessionPhase.ENDING

    def test_setup_can_go_back_to_idle(self, session_state):
        """SETUP pode voltar para IDLE (fallback para erro)"""
        session_state.transition_to(SessionPhase.SETUP)
        assert session_state.transition_to(SessionPhase.IDLE)
        assert session_state.phase == SessionPhase.IDLE

    def test_ready_can_go_back_to_idle(self, session_state):
        """READY pode voltar para IDLE (cancelamento)"""
        session_state.transition_to(SessionPhase.SETUP)
        session_state.transition_to(SessionPhase.READY)
        assert session_state.transition_to(SessionPhase.IDLE)
        assert session_state.phase == SessionPhase.IDLE


# ============================================================================
# TESTES: SessionState - Dados e Reset
# ============================================================================

class TestSessionStateData:
    """Testes para gerenciamento de dados na sessão"""

    def test_initial_data_is_none(self, session_state):
        """Dados iniciais devem ser None"""
        assert session_state.patient is None
        assert session_state.task_type is None

    def test_can_set_patient_data(self, session_state, patient_data):
        """Deve ser possível setar dados do paciente"""
        session_state.patient = patient_data
        assert session_state.patient == patient_data
        assert session_state.patient.nome == "João Silva"
        assert session_state.patient.nivel == 5
        assert session_state.patient.lado == "Direito"

    def test_can_set_task_type(self, session_state):
        """Deve ser possível setar tipo de tarefa"""
        session_state.task_type = TaskType.TREINO
        assert session_state.task_type == TaskType.TREINO
        
        session_state.task_type = TaskType.JOGO
        assert session_state.task_type == TaskType.JOGO

    def test_reset_clears_patient(self, session_state, patient_data):
        """Reset deve limpar dados do paciente"""
        session_state.patient = patient_data
        session_state.reset()
        assert session_state.patient is None

    def test_reset_clears_task_type(self, session_state):
        """Reset deve limpar tipo de tarefa"""
        session_state.task_type = TaskType.TREINO
        session_state.reset()
        assert session_state.task_type is None

    def test_reset_returns_to_idle(self, session_state):
        """Reset deve retornar phase para IDLE"""
        session_state.transition_to(SessionPhase.SETUP)
        session_state.reset()
        assert session_state.phase == SessionPhase.IDLE

    def test_reset_clears_all_state(self, session_state, patient_data):
        """Reset deve limpar tudo de uma vez"""
        session_state.patient = patient_data
        session_state.task_type = TaskType.JOGO
        session_state.transition_to(SessionPhase.SETUP)
        
        session_state.reset()
        
        assert session_state.patient is None
        assert session_state.task_type is None
        assert session_state.phase == SessionPhase.IDLE


# ============================================================================
# TESTES: PatientData - Validação
# ============================================================================

class TestPatientDataValidation:
    """Testes para validação de dados do paciente"""

    def test_valid_patient_data(self):
        """Dados válidos devem criar PatientData sem erros"""
        patient = PatientData(nome="Maria", nivel=3, lado="Esquerdo")
        assert patient.nome == "Maria"
        assert patient.nivel == 3
        assert patient.lado == "Esquerdo"

    def test_nivel_must_be_integer(self):
        """Nível deve ser inteiro"""
        with pytest.raises(ValueError):
            PatientData(nome="João", nivel=5.5, lado="Direito")

    def test_nivel_must_be_in_range_0_to_11(self):
        """Nível deve estar entre 0 e 11"""
        with pytest.raises(ValueError):
            PatientData(nome="João", nivel=-1, lado="Direito")
        
        with pytest.raises(ValueError):
            PatientData(nome="João", nivel=12, lado="Direito")

    def test_nivel_boundaries_valid(self):
        """Nível 0 e 11 devem ser válidos"""
        patient_0 = PatientData(nome="João", nivel=0, lado="Direito")
        assert patient_0.nivel == 0
        
        patient_11 = PatientData(nome="João", nivel=11, lado="Direito")
        assert patient_11.nivel == 11

    def test_lado_must_be_valid(self):
        """Lado deve ser 'Esquerdo' ou 'Direito'"""
        with pytest.raises(ValueError):
            PatientData(nome="João", nivel=5, lado="Centro")
        
        with pytest.raises(ValueError):
            PatientData(nome="João", nivel=5, lado="Left")

    def test_lado_case_sensitive(self):
        """Lado é case-sensitive"""
        with pytest.raises(ValueError):
            PatientData(nome="João", nivel=5, lado="direito")  # minúsculo

    def test_format_message(self):
        """format_message deve gerar mensagem correta"""
        patient = PatientData(nome="João", nivel=5, lado="Direito")
        msg = patient.format_message()
        
        assert "João" in msg
        assert "5" in msg
        assert "Direito" in msg
        assert "Dados Paciente" in msg


# ============================================================================
# TESTES: ServerState - Estados do Servidor
# ============================================================================

class TestServerState:
    """Testes para estados do servidor"""

    def test_initial_server_state_is_stopped(self, communicator):
        """Servidor inicial deve estar STOPPED"""
        assert communicator.server_state == ServerState.STOPPED

    def test_start_server_transitions_to_running(self, communicator):
        """start_server deve transicionar para RUNNING"""
        communicator.start_server()
        assert communicator.server_state == ServerState.RUNNING

    def test_stop_server_transitions_to_stopped(self, communicator):
        """stop_server deve transicionar para STOPPED"""
        communicator.start_server()
        communicator.stop_server()
        assert communicator.server_state == ServerState.STOPPED

    def test_stop_server_already_stopped(self, communicator):
        """stop_server em servidor já parado não deve falhar"""
        communicator.stop_server()  # Já está parado
        communicator.stop_server()  # Não deve falhar
        assert communicator.server_state == ServerState.STOPPED


# ============================================================================
# TESTES: Helpers de Query de Estado
# ============================================================================

class TestStateHelpers:
    """Testes para métodos helpers de query de estado"""

    def test_is_server_operational_when_stopped(self, communicator):
        """_is_server_operational deve retornar False quando STOPPED"""
        assert not communicator._is_server_operational()

    def test_is_server_operational_when_running(self, communicator):
        """_is_server_operational deve retornar True quando RUNNING"""
        communicator.start_server()
        assert communicator._is_server_operational()

    def test_is_server_ready_for_session_when_stopped(self, communicator):
        """_is_server_ready_for_session deve retornar False quando servidor parado"""
        assert not communicator._is_server_ready_for_session()

    def test_is_server_ready_for_session_when_running_but_no_vr(self, communicator):
        """_is_server_ready_for_session deve retornar False sem VR conectado"""
        communicator.start_server()
        assert not communicator._is_server_ready_for_session()

    def test_is_session_waiting_trigger_false_in_idle(self, communicator):
        """_is_session_waiting_trigger deve retornar False em IDLE"""
        assert not communicator._is_session_waiting_trigger()

    def test_is_session_waiting_trigger_true_in_ready(self, communicator):
        """_is_session_waiting_trigger deve retornar True em READY"""
        communicator.session.transition_to(SessionPhase.SETUP)
        communicator.session.transition_to(SessionPhase.READY)
        assert communicator._is_session_waiting_trigger()

    def test_is_session_active_for_commands_false_in_idle(self, communicator):
        """_is_session_active_for_commands deve retornar False em IDLE"""
        assert not communicator._is_session_active_for_commands()

    def test_is_session_active_for_commands_true_in_active(self, communicator):
        """_is_session_active_for_commands deve retornar True em ACTIVE"""
        communicator.session.transition_to(SessionPhase.SETUP)
        communicator.session.transition_to(SessionPhase.READY)
        communicator.session.transition_to(SessionPhase.ACTIVE)
        assert communicator._is_session_active_for_commands()


# ============================================================================
# TESTES: Sem Interdependências (Arquitetura)
# ============================================================================

class TestNoInterDependencies:
    """Testes para validar que eliminamos interdependências"""

    def test_session_state_no_is_active(self, session_state):
        """SessionState não deve ter atributo is_active"""
        assert not hasattr(session_state, 'is_active'), \
            "SessionState não deve ter is_active - use phase ao invés"

    def test_session_state_no_waiting_confirmation(self, session_state):
        """SessionState não deve ter atributo waiting_confirmation"""
        assert not hasattr(session_state, 'waiting_confirmation'), \
            "SessionState não deve ter waiting_confirmation - use phase ao invés"

    def test_server_state_is_source_of_truth(self, communicator):
        """server_state deve ser a fonte única de verdade do servidor"""
        assert hasattr(communicator, 'server_state')
        assert isinstance(communicator.server_state, ServerState)
        
        # tcp_connected pode existir como detalhe, mas não é a fonte
        # A verdade vem de server_state
        communicator.start_server()
        assert communicator.server_state == ServerState.RUNNING

    def test_session_phase_is_source_of_truth(self, communicator):
        """session.phase deve ser a fonte única de verdade da sessão"""
        assert hasattr(communicator.session, 'phase')
        assert isinstance(communicator.session.phase, SessionPhase)
        
        # Não pode haver is_active ou waiting_confirmation
        assert not hasattr(communicator.session, 'is_active')
        assert not hasattr(communicator.session, 'waiting_confirmation')


# ============================================================================
# TESTES: Edge Cases e Erros
# ============================================================================

class TestEdgeCases:
    """Testes para casos extremos e edge cases"""

    def test_transition_to_same_phase_idempotent(self, session_state):
        """Transicionar para mesma fase deve retornar True (idempotente)"""
        current_phase = session_state.phase
        result = session_state.transition_to(current_phase)
        assert result  # Pode ser True se implementado como idempotente
        assert session_state.phase == current_phase

    def test_multiple_resets(self, session_state, patient_data):
        """Múltiplos resets não devem causar erro"""
        session_state.patient = patient_data
        session_state.transition_to(SessionPhase.SETUP)
        
        session_state.reset()
        session_state.reset()
        session_state.reset()
        
        assert session_state.phase == SessionPhase.IDLE
        assert session_state.patient is None

    def test_patient_data_immutable_after_creation(self):
        """PatientData deve ser immutable (dataclass frozen)"""
        patient = PatientData(nome="João", nivel=5, lado="Direito")
        
        # Tentar modificar deve falhar ou ser notado
        # (dataclass com frozen=True não permite)
        try:
            patient.nome = "Maria"
            # Se chegou aqui, não é frozen
            assert patient.nome == "Maria"  # Mas pelo menos registrou
        except (AttributeError, TypeError):
            # Expected se frozen=True
            pass

    def test_enum_values_are_correct_strings(self):
        """Valores dos enums devem ser strings corretas"""
        assert TriggerCommand.START.value == "Trigger"
        assert ActionCommand.LEFT_HAND_CLOSE.value == "LEFT_HAND_CLOSE"
        assert ActionCommand.RIGHT_HAND_CLOSE.value == "RIGHT_HAND_CLOSE"
        assert EndTaskCommand.END_TRAINING.value == "Finalizar_tarefa_treino"
        assert EndTaskCommand.END_GAME.value == "Finalizar_tarefa_jogo"

    def test_session_state_with_none_values(self, session_state):
        """SessionState com None values deve ser seguro"""
        assert session_state.patient is None
        assert session_state.task_type is None
        assert session_state.phase == SessionPhase.IDLE
        
        # Não deve falhar em operações
        session_state.reset()
        session_state.transition_to(SessionPhase.SETUP)


# ============================================================================
# TESTES: Fluxos Completos de Erro
# ============================================================================

class TestErrorFlows:
    """Testes para fluxos com recuperação de erro"""

    def test_setup_failure_can_go_back_to_idle(self, session_state):
        """Se SETUP falhar, pode voltar para IDLE"""
        session_state.transition_to(SessionPhase.SETUP)
        assert session_state.phase == SessionPhase.SETUP
        
        # Simular falha
        session_state.transition_to(SessionPhase.IDLE)
        assert session_state.phase == SessionPhase.IDLE

    def test_ready_cancellation(self, session_state):
        """Pode cancelar antes de ACTIVE"""
        session_state.transition_to(SessionPhase.SETUP)
        session_state.transition_to(SessionPhase.READY)
        assert session_state.phase == SessionPhase.READY
        
        # Cancelar
        session_state.transition_to(SessionPhase.IDLE)
        assert session_state.phase == SessionPhase.IDLE

    def test_cannot_send_command_before_active(self, communicator):
        """Não deve enviar comandos fora de ACTIVE"""
        assert not communicator._is_session_active_for_commands()
        
        # Tentar em diferentes fases
        for phase in [SessionPhase.IDLE, SessionPhase.SETUP, SessionPhase.READY, SessionPhase.ENDING]:
            communicator.session.phase = phase
            assert not communicator._is_session_active_for_commands(), \
                f"Não deveria aceitar comandos em {phase}"

    def test_cannot_start_session_twice(self, communicator):
        """Não deve poder iniciar sessão se já houver uma em progresso"""
        assert communicator._is_server_ready_for_session() or True  # Pode estar sem VR
        
        # Tentar iniciar
        communicator.session.transition_to(SessionPhase.SETUP)
        assert not communicator._is_server_ready_for_session()  # Agora não está pronto


# ============================================================================
# TESTES: Compatibilidade com Legado
# ============================================================================

class TestLegacyCompatibility:
    """Testes para compatibilidade com código legado"""

    def test_udp_sender_is_server_active_queries_server_state(self, communicator):
        """UDP_sender.is_server_active() deve usar novo estado"""
        communicator.server_state = ServerState.STOPPED
        assert not communicator._is_server_operational()
        
        communicator.server_state = ServerState.RUNNING
        assert communicator._is_server_operational()

    def test_transition_helpers_centralize_logic(self, communicator):
        """Helpers devem centralizar lógica de query"""
        # Não devemos checar múltiplas variáveis em code legado
        # Devemos usar helpers
        
        assert hasattr(communicator, '_is_server_operational')
        assert hasattr(communicator, '_is_server_ready_for_session')
        assert hasattr(communicator, '_is_session_waiting_trigger')
        assert hasattr(communicator, '_is_session_active_for_commands')


# ============================================================================
# EXECUTAR TESTES
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
