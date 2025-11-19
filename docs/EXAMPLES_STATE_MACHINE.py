"""
EXEMPLOS DE USO: Máquina de Estados na Prática
===============================================

Demonstra como usar a nova arquitetura refatorada
"""

from brainbridge_v2.communication.unity import (
    UnityCommunicator, SessionPhase, ServerState, PatientData,
    TaskType, UDP_sender
)


# ===========================================================================
# EXEMPLO 1: Iniciar e Parar Servidor
# ===========================================================================

def exemplo_1_servidor():
    """Exemplo: Ciclo de vida do servidor"""
    print("=" * 70)
    print("EXEMPLO 1: Ciclo de Vida do Servidor")
    print("=" * 70)
    
    comm = UnityCommunicator()
    
    # Verificar estado inicial
    print(f"\n1. Estado inicial: {comm.server_state}")
    assert comm.server_state == ServerState.STOPPED
    
    # Iniciar servidor
    print("\n2. Iniciando servidor...")
    comm.start_server()
    print(f"   Estado: {comm.server_state}")
    assert comm.server_state == ServerState.RUNNING
    
    # Verificar se está operacional
    print(f"\n3. Servidor operacional? {comm._is_server_operational()}")
    assert comm._is_server_operational()
    
    # Parar servidor
    print("\n4. Parando servidor...")
    comm.stop_server()
    print(f"   Estado: {comm.server_state}")
    assert comm.server_state == ServerState.STOPPED
    
    print("\n✅ Exemplo 1 finalizado\n")


# ===========================================================================
# EXEMPLO 2: Transições de SessionPhase
# ===========================================================================

def exemplo_2_transicoes():
    """Exemplo: Transições de fase da sessão"""
    print("=" * 70)
    print("EXEMPLO 2: Transições de SessionPhase")
    print("=" * 70)
    
    session = UnityCommunicator().session
    
    # Começar em IDLE
    print(f"\n1. Fase inicial: {session.phase}")
    assert session.phase == SessionPhase.IDLE
    
    # Tentar transição inválida
    print("\n2. Tentar IDLE → ACTIVE (deve falhar)...")
    resultado = session.transition_to(SessionPhase.ACTIVE)
    print(f"   Resultado: {resultado}")
    assert not resultado, "Deve ser inválido!"
    print(f"   Fase ainda: {session.phase}")
    
    # Transição válida
    print("\n3. IDLE → SETUP (válido)...")
    resultado = session.transition_to(SessionPhase.SETUP)
    print(f"   Resultado: {resultado}")
    assert resultado, "Deve ser válido!"
    print(f"   Fase agora: {session.phase}")
    
    # Continuar fluxo
    print("\n4. Seguir fluxo: SETUP → READY → ACTIVE → ENDING → IDLE...")
    fases = [SessionPhase.READY, SessionPhase.ACTIVE, SessionPhase.ENDING, SessionPhase.IDLE]
    for fase in fases:
        resultado = session.transition_to(fase)
        print(f"   → {fase.value}: {resultado}")
        assert resultado
    
    print("\n✅ Exemplo 2 finalizado\n")


# ===========================================================================
# EXEMPLO 3: Validação de PatientData
# ===========================================================================

def exemplo_3_paciente():
    """Exemplo: Criar e validar dados do paciente"""
    print("=" * 70)
    print("EXEMPLO 3: Validação de PatientData")
    print("=" * 70)
    
    # Dados válidos
    print("\n1. Criar paciente com dados válidos...")
    patient = PatientData(
        nome="João Silva",
        nivel=7,
        lado="Direito"
    )
    print(f"   Nome: {patient.nome}")
    print(f"   Nível: {patient.nivel}")
    print(f"   Lado: {patient.lado}")
    
    # Mensagem formatada
    print("\n2. Formatar mensagem do paciente...")
    msg = patient.format_message()
    print(msg)
    
    # Erro: nível inválido
    print("\n3. Tentar criar paciente com nível > 11...")
    try:
        bad_patient = PatientData(
            nome="Maria",
            nivel=15,  # Inválido!
            lado="Esquerdo"
        )
    except ValueError as e:
        print(f"   ✅ Erro capturado: {e}")
    
    # Erro: lado inválido
    print("\n4. Tentar criar paciente com lado inválido...")
    try:
        bad_patient = PatientData(
            nome="Carlos",
            nivel=5,
            lado="Centro"  # Inválido!
        )
    except ValueError as e:
        print(f"   ✅ Erro capturado: {e}")
    
    print("\n✅ Exemplo 3 finalizado\n")


# ===========================================================================
# EXEMPLO 4: Helpers de Query de Estado
# ===========================================================================

def exemplo_4_helpers():
    """Exemplo: Usar helpers para queries de estado"""
    print("=" * 70)
    print("EXEMPLO 4: Helpers de Query de Estado")
    print("=" * 70)
    
    comm = UnityCommunicator()
    
    # Servidor parado
    print("\n1. Servidor parado:")
    print(f"   _is_server_operational(): {comm._is_server_operational()}")
    assert not comm._is_server_operational()
    
    # Iniciar servidor
    print("\n2. Servidor rodando:")
    comm.start_server()
    print(f"   _is_server_operational(): {comm._is_server_operational()}")
    assert comm._is_server_operational()
    
    # Sessão em IDLE
    print("\n3. Sessão em IDLE:")
    print(f"   _is_session_waiting_trigger(): {comm._is_session_waiting_trigger()}")
    print(f"   _is_session_active_for_commands(): {comm._is_session_active_for_commands()}")
    assert not comm._is_session_waiting_trigger()
    assert not comm._is_session_active_for_commands()
    
    # Sessão em READY
    print("\n4. Sessão em READY:")
    comm.session.transition_to(SessionPhase.SETUP)
    comm.session.transition_to(SessionPhase.READY)
    print(f"   _is_session_waiting_trigger(): {comm._is_session_waiting_trigger()}")
    print(f"   _is_session_active_for_commands(): {comm._is_session_active_for_commands()}")
    assert comm._is_session_waiting_trigger()
    assert not comm._is_session_active_for_commands()
    
    # Sessão em ACTIVE
    print("\n5. Sessão em ACTIVE:")
    comm.session.transition_to(SessionPhase.ACTIVE)
    print(f"   _is_session_waiting_trigger(): {comm._is_session_waiting_trigger()}")
    print(f"   _is_session_active_for_commands(): {comm._is_session_active_for_commands()}")
    assert not comm._is_session_waiting_trigger()
    assert comm._is_session_active_for_commands()
    
    comm.stop_server()
    print("\n✅ Exemplo 4 finalizado\n")


# ===========================================================================
# EXEMPLO 5: Fluxo Completo de Sessão
# ===========================================================================

def exemplo_5_fluxo_completo():
    """Exemplo: Fluxo completo de uma sessão (sem VR real)"""
    print("=" * 70)
    print("EXEMPLO 5: Fluxo Completo de Sessão")
    print("=" * 70)
    
    comm = UnityCommunicator()
    
    # 1. Iniciar servidor
    print("\n1. Iniciando servidor...")
    comm.start_server()
    print(f"   ServerState: {comm.server_state}")
    
    # 2. Simular dados do paciente
    print("\n2. Preparando dados do paciente...")
    patient = PatientData(
        nome="João Silva",
        nivel=5,
        lado="Direito"
    )
    print(f"   Paciente: {patient.nome}")
    
    # 3. Simular confirmação de VR (sem TCP real)
    print("\n3. Simulando conexão VR...")
    comm.server_state = ServerState.CONNECTED
    comm.tcp_connected = True
    print(f"   ServerState: {comm.server_state}")
    
    # 4. Verificar pré-requisitos
    print("\n4. Verificar pré-requisitos para sessão...")
    print(f"   _is_server_ready_for_session(): {comm._is_server_ready_for_session()}")
    
    # 5. Iniciar fluxo de sessão
    print("\n5. Transição para SETUP...")
    comm.session.transition_to(SessionPhase.SETUP)
    print(f"   SessionPhase: {comm.session.phase}")
    print(f"   Ação: Enviar dados do paciente")
    
    # 6. Simular confirmação do VR
    print("\n6. Simular confirmação VR...")
    comm.session.transition_to(SessionPhase.READY)
    print(f"   SessionPhase: {comm.session.phase}")
    
    # 7. Aguardar trigger
    print("\n7. Aguardando trigger...")
    print(f"   _is_session_waiting_trigger(): {comm._is_session_waiting_trigger()}")
    
    # 8. Enviar trigger
    print("\n8. Enviar TRIGGER...")
    comm.session.transition_to(SessionPhase.ACTIVE)
    print(f"   SessionPhase: {comm.session.phase}")
    
    # 9. Enviar comandos
    print("\n9. Enviar comandos...")
    if comm._is_session_active_for_commands():
        print("   ✅ Pode enviar: send_hand_close('direita')")
        print("   ✅ Pode enviar: send_flower_action('esquerda')")
    
    # 10. Finalizar sessão
    print("\n10. Finalizar sessão...")
    comm.session.transition_to(SessionPhase.ENDING)
    print(f"   SessionPhase: {comm.session.phase}")
    
    # 11. Simular confirmação de finalização
    print("\n11. Simular confirmação VR...")
    comm.session.transition_to(SessionPhase.IDLE)
    print(f"   SessionPhase: {comm.session.phase}")
    
    # 12. Parar servidor
    print("\n12. Parando servidor...")
    comm.stop_server()
    print(f"   ServerState: {comm.server_state}")
    
    print("\n✅ Exemplo 5 finalizado\n")


# ===========================================================================
# EXEMPLO 6: Compatibilidade com Legado (UDP_sender)
# ===========================================================================

def exemplo_6_legado():
    """Exemplo: Como o código legado (UDP_sender) ainda funciona"""
    print("=" * 70)
    print("EXEMPLO 6: Compatibilidade com UDP_sender (Legado)")
    print("=" * 70)
    
    # UDP_sender ainda existe e funciona
    print("\n1. Inicializar com UDP_sender (compatibilidade)...")
    UDP_sender.init_zmq_socket()
    print(f"   UDP_sender.is_server_active(): {UDP_sender.is_server_active()}")
    
    # Verificar que internamente usa nova arquitetura
    print("\n2. Internamente usa ServerState...")
    comm = UDP_sender._communicator
    print(f"   Comunicador interno ServerState: {comm.server_state}")
    
    print("\n3. Métodos legados mapeiam para novos...")
    print("   UDP_sender.enviar_sinal() → comm.send_hand_close/etc")
    
    print("\n4. Parar...")
    UDP_sender.stop_zmq_socket()
    print(f"   UDP_sender.is_server_active(): {UDP_sender.is_server_active()}")
    
    print("\n✅ Exemplo 6 finalizado\n")


# ===========================================================================
# EXEMPLO 7: Fallback em Caso de Erro
# ===========================================================================

def exemplo_7_fallback():
    """Exemplo: Recuperação de erro com fallback"""
    print("=" * 70)
    print("EXEMPLO 7: Fallback e Recuperação de Erro")
    print("=" * 70)
    
    session = UnityCommunicator().session
    
    # Começar um fluxo
    print("\n1. Iniciar fluxo: IDLE → SETUP...")
    session.transition_to(SessionPhase.SETUP)
    print(f"   SessionPhase: {session.phase}")
    
    # Erro! Não conseguiu enviar dados
    print("\n2. ERRO: Não conseguiu enviar dados!")
    print("   Ação: Voltar para IDLE (fallback)")
    
    session.transition_to(SessionPhase.IDLE)
    print(f"   SessionPhase: {session.phase}")
    
    # Tentar novamente
    print("\n3. Tentar novamente...")
    session.transition_to(SessionPhase.SETUP)
    print(f"   SessionPhase: {session.phase}")
    
    # Desta vez sucesso
    print("\n4. Desta vez funcionou! Continuar...")
    session.transition_to(SessionPhase.READY)
    print(f"   SessionPhase: {session.phase}")
    
    # Cancelamento antes de iniciar
    print("\n5. Usuário clica em CANCELAR antes do trigger...")
    session.transition_to(SessionPhase.IDLE)
    print(f"   SessionPhase: {session.phase}")
    
    print("\n✅ Exemplo 7 finalizado\n")


# ===========================================================================
# MAIN: Executar todos os exemplos
# ===========================================================================

def main():
    """Executa todos os exemplos"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  EXEMPLOS: Usando a Máquina de Estados".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    try:
        exemplo_1_servidor()
        exemplo_2_transicoes()
        exemplo_3_paciente()
        exemplo_4_helpers()
        exemplo_5_fluxo_completo()
        exemplo_6_legado()
        exemplo_7_fallback()
        
        print("\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print("║" + "  ✅ TODOS OS EXEMPLOS FUNCIONARAM!".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
