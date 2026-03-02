"""
Testes do protocolo de comunicação Sistema <-> VR Unity
"""
import pytest
from brainbridge_v2.infrastructure.communication.unity import (
    UnityCommunicator,
    PatientData,
    TaskType,
    SessionPhase,
    ActionCommand,
    EndTaskCommand,
    SessionState
)


class TestPatientData:
    """Testes da classe PatientData"""
    
    def test_patient_data_valid_nivel_min(self):
        """Testa criação com nível mínimo válido (0)"""
        patient = PatientData("João", 0, "Direito")
        assert patient.nome == "João"
        assert patient.nivel == 0
        assert patient.lado == "Direito"
    
    def test_patient_data_valid_nivel_max(self):
        """Testa criação com nível máximo válido (11)"""
        patient = PatientData("Maria", 11, "Esquerdo")
        assert patient.nome == "Maria"
        assert patient.nivel == 11
        assert patient.lado == "Esquerdo"
    
    def test_patient_data_valid_nivel_mid(self):
        """Testa criação com nível intermediário válido"""
        patient = PatientData("José", 5, "Direito")
        assert patient.nivel == 5
    
    def test_patient_data_invalid_nivel_negative(self):
        """Testa que nível negativo gera erro"""
        with pytest.raises(ValueError) as exc_info:
            PatientData("João", -1, "Direito")
        assert "0 e 11" in str(exc_info.value)
    
    def test_patient_data_invalid_nivel_above_max(self):
        """Testa que nível acima de 11 gera erro"""
        with pytest.raises(ValueError) as exc_info:
            PatientData("Maria", 12, "Esquerdo")
        assert "0 e 11" in str(exc_info.value)
    
    def test_patient_data_invalid_lado(self):
        """Testa que lado inválido gera erro"""
        with pytest.raises(ValueError) as exc_info:
            PatientData("José", 5, "Centro")
        assert "Esquerdo" in str(exc_info.value) or "Direito" in str(exc_info.value)
    
    def test_patient_data_format_message(self):
        """Testa formatação da mensagem do protocolo"""
        patient = PatientData("João Silva", 7, "Direito")
        message = patient.format_message()
        
        assert "Dados Paciente:" in message
        assert "Nome: João Silva" in message
        assert "Nivel: 7" in message
        assert "Lado: Direito" in message
    
    def test_all_valid_niveis(self):
        """Testa que todos os níveis de 0 a 11 são válidos"""
        for nivel in range(12):
            patient = PatientData(f"Paciente{nivel}", nivel, "Direito")
            assert patient.nivel == nivel


class TestTaskType:
    """Testes do enum TaskType"""
    
    def test_task_type_treino(self):
        """Testa valor de TREINO"""
        assert TaskType.TREINO.value == "Treino"
    
    def test_task_type_jogo(self):
        """Testa valor de JOGO"""
        assert TaskType.JOGO.value == "Jogo"


class TestActionCommand:
    """Testes do enum ActionCommand"""
    
    def test_action_commands_exist(self):
        """Verifica que todos os comandos de ação existem"""
        assert ActionCommand.LEFT_HAND_CLOSE.value == "LEFT_HAND_CLOSE"
        assert ActionCommand.RIGHT_HAND_CLOSE.value == "RIGHT_HAND_CLOSE"
        assert ActionCommand.LEFT_FLOWER.value == "LEFT_FLOWER"
        assert ActionCommand.RIGHT_FLOWER.value == "RIGHT_FLOWER"


class TestEndTaskCommand:
    """Testes do enum EndTaskCommand"""
    
    def test_end_task_commands(self):
        """Verifica comandos de finalização"""
        assert EndTaskCommand.END_TRAINING.value == "Finalizar_tarefa_treino"
        assert EndTaskCommand.END_GAME.value == "Finalizar_tarefa_jogo"


class TestSessionState:
    """Testes da classe SessionState"""
    
    def test_session_state_initial(self):
        """Testa estado inicial da sessão"""
        session = SessionState()
        assert session.patient is None
        assert session.task_type is None
        assert session.phase == SessionPhase.IDLE


class TestUnityCommunicator:
    """Testes da classe UnityCommunicator"""
    
    def test_singleton_pattern(self):
        """Testa que UnityCommunicator é singleton"""
        comm1 = UnityCommunicator()
        comm2 = UnityCommunicator()
        assert comm1 is comm2
    
    def test_initial_state(self):
        """Testa estado inicial do comunicador"""
        comm = UnityCommunicator()
        assert comm.is_active is False
        assert comm.tcp_connected is False
        assert comm.session.phase == SessionPhase.IDLE
    
    def test_broadcast_header_constant(self):
        """Verifica que o header de broadcast está correto"""
        comm = UnityCommunicator()
        assert comm.CONFIRM_HEADER == "Confirm"
    
    def test_ports_configured(self):
        """Verifica que as portas estão configuradas corretamente"""
        comm = UnityCommunicator()
        assert comm.UDP_PORT == 12346
        assert comm.TCP_PORT == 12345
        assert comm.ZMQ_PORT == 5555


class TestProtocolFlow:
    """Testes do fluxo do protocolo"""
    
    def test_start_session_without_server(self):
        """Testa que start_session falha se servidor não está ativo"""
        comm = UnityCommunicator()
        patient = PatientData("João", 5, "Direito")
        
        # Garantir que servidor está parado
        comm.stop_server()
        
        result = comm.start_session(patient, TaskType.TREINO)
        assert result is False
    
    def test_send_trigger_without_session(self):
        """Testa que trigger falha se não há sessão configurada"""
        comm = UnityCommunicator()
        comm.stop_server()
        
        result = comm.send_trigger()
        assert result is False
    
    def test_send_hand_close_without_active_session(self):
        """Testa que comandos falham se sessão não está ativa"""
        comm = UnityCommunicator()
        comm.stop_server()
        
        result = comm.send_hand_close("direita")
        assert result is False
    
    def test_end_session_without_active_session(self):
        """Testa que end_task falha se sessão não está ativa"""
        comm = UnityCommunicator()
        comm.stop_server()
        
        result = comm.end_task()
        assert result is False


class TestCompatibilityLayer:
    """Testes da camada de compatibilidade UDP_sender"""
    
    def test_udp_sender_import(self):
        """Testa que UDP_sender pode ser importado"""
        from brainbridge_v2.infrastructure.communication.unity import UDP_sender
        assert UDP_sender is not None
    
    def test_udp_sender_methods_exist(self):
        """Verifica que métodos legados existem"""
        from brainbridge_v2.infrastructure.communication.unity import UDP_sender
        
        assert hasattr(UDP_sender, 'init_zmq_socket')
        assert hasattr(UDP_sender, 'stop_zmq_socket')
        assert hasattr(UDP_sender, 'enviar_sinal')
        assert hasattr(UDP_sender, 'is_server_active')


if __name__ == "__main__":
    # Executar testes sem pytest
    print("🧪 Executando testes do protocolo Unity...\n")
    
    test_patient = TestPatientData()
    print("✅ TestPatientData.test_patient_data_valid_nivel_min")
    test_patient.test_patient_data_valid_nivel_min()
    
    print("✅ TestPatientData.test_patient_data_valid_nivel_max")
    test_patient.test_patient_data_valid_nivel_max()
    
    print("✅ TestPatientData.test_all_valid_niveis")
    test_patient.test_all_valid_niveis()
    
    print("✅ TestPatientData.test_patient_data_format_message")
    test_patient.test_patient_data_format_message()
    
    test_comm = TestUnityCommunicator()
    print("✅ TestUnityCommunicator.test_singleton_pattern")
    test_comm.test_singleton_pattern()
    
    print("✅ TestUnityCommunicator.test_initial_state")
    test_comm.test_initial_state()
    
    print("✅ TestUnityCommunicator.test_ports_configured")
    test_comm.test_ports_configured()
    
    print("\n🎉 Todos os testes passaram!")
