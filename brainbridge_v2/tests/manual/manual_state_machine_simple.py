#!/usr/bin/env python
"""
Teste Simples da Máquina de Estados - Sem pytest
Apenas Python puro para demonstrar a arquitetura limpa
"""

import sys
import os

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import
from brainbridge_v2.infrastructure.communication.unity import (
    SessionPhase, ServerState, SessionState, PatientData,
    TaskType, UnityCommunicator
)


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def test(name, condition):
    """Testa uma condição e printa resultado"""
    if condition:
        print(f"{Colors.GREEN}✅{Colors.END} {name}")
        return True
    else:
        print(f"{Colors.RED}❌{Colors.END} {name}")
        return False


def section(title):
    """Printa seção de teste"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title.center(70)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}\n")


def main():
    """Executa todos os testes"""
    
    tests_passed = 0
    tests_total = 0
    
    # ====================================================================
    # TESTE 1: Transições Válidas
    # ====================================================================
    section("TESTE 1: Transições de Fase Válidas")
    
    session = SessionState()
    
    tests_total += 1
    tests_passed += test("Estado inicial é IDLE", session.phase == SessionPhase.IDLE)
    
    tests_total += 1
    tests_passed += test("IDLE -> SETUP", session.transition_to(SessionPhase.SETUP))
    
    tests_total += 1
    tests_passed += test("SETUP -> READY", session.transition_to(SessionPhase.READY))
    
    tests_total += 1
    tests_passed += test("READY -> ACTIVE", session.transition_to(SessionPhase.ACTIVE))
    
    tests_total += 1
    tests_passed += test("ACTIVE -> ENDING", session.transition_to(SessionPhase.ENDING))
    
    tests_total += 1
    tests_passed += test("ENDING -> IDLE", session.transition_to(SessionPhase.IDLE))
    
    # ====================================================================
    # TESTE 2: Transições Inválidas (Bloqueadas)
    # ====================================================================
    section("TESTE 2: Transições Inválidas (Devem Ser Bloqueadas)")
    
    session = SessionState()
    
    tests_total += 1
    tests_passed += test("IDLE -> READY (bloqueado)", not session.transition_to(SessionPhase.READY))
    
    tests_total += 1
    tests_passed += test("IDLE -> ACTIVE (bloqueado)", not session.transition_to(SessionPhase.ACTIVE))
    
    session.transition_to(SessionPhase.SETUP)
    tests_total += 1
    tests_passed += test("SETUP -> ACTIVE (bloqueado)", not session.transition_to(SessionPhase.ACTIVE))
    
    tests_total += 1
    tests_passed += test("SETUP -> ENDING (bloqueado)", not session.transition_to(SessionPhase.ENDING))
    
    session.transition_to(SessionPhase.READY)
    tests_total += 1
    tests_passed += test("READY -> ENDING (bloqueado)", not session.transition_to(SessionPhase.ENDING))
    
    # ====================================================================
    # TESTE 3: Reset de Estado
    # ====================================================================
    section("TESTE 3: Reset Limpa o Estado")
    
    session = SessionState()
    patient = PatientData(nome="João", nivel=5, lado="Direito")
    
    session.patient = patient
    session.task_type = TaskType.TREINO
    session.transition_to(SessionPhase.SETUP)
    
    tests_total += 1
    tests_passed += test("Estado preenchido (paciente)", session.patient is not None)
    
    tests_total += 1
    tests_passed += test("Estado preenchido (tarefa)", session.task_type == TaskType.TREINO)
    
    tests_total += 1
    tests_passed += test("Estado preenchido (fase)", session.phase == SessionPhase.SETUP)
    
    session.reset()
    
    tests_total += 1
    tests_passed += test("Reset: paciente = None", session.patient is None)
    
    tests_total += 1
    tests_passed += test("Reset: tarefa = None", session.task_type is None)
    
    tests_total += 1
    tests_passed += test("Reset: fase = IDLE", session.phase == SessionPhase.IDLE)
    
    # ====================================================================
    # TESTE 4: Validação de PatientData
    # ====================================================================
    section("TESTE 4: Validação de PatientData")
    
    tests_total += 1
    try:
        PatientData(nome="João", nivel=5, lado="Direito")
        tests_passed += test("PatientData válido", True)
    except:
        tests_passed += test("PatientData válido", False)
    
    tests_total += 1
    try:
        PatientData(nome="João", nivel=-1, lado="Direito")
        tests_passed += test("Rejeita nivel < 0", False)
    except ValueError:
        tests_passed += test("Rejeita nivel < 0", True)
    
    tests_total += 1
    try:
        PatientData(nome="João", nivel=12, lado="Direito")
        tests_passed += test("Rejeita nivel > 11", False)
    except ValueError:
        tests_passed += test("Rejeita nivel > 11", True)
    
    tests_total += 1
    try:
        PatientData(nome="João", nivel=5, lado="Centro")
        tests_passed += test("Rejeita lado inválido", False)
    except ValueError:
        tests_passed += test("Rejeita lado inválido", True)
    
    # ====================================================================
    # TESTE 5: ServerState do Communicator
    # ====================================================================
    section("TESTE 5: ServerState do Communicator")
    
    comm = UnityCommunicator()
    
    tests_total += 1
    tests_passed += test("Inicial: STOPPED", comm.server_state == ServerState.STOPPED)
    
    tests_total += 1
    comm.start_server()
    tests_passed += test("Após start: RUNNING", comm.server_state == ServerState.RUNNING)
    
    tests_total += 1
    comm.stop_server()
    tests_passed += test("Após stop: STOPPED", comm.server_state == ServerState.STOPPED)
    
    # ====================================================================
    # TESTE 6: Sem Interdependências
    # ====================================================================
    section("TESTE 6: Sem Interdependências Cruzadas")
    
    session = SessionState()
    
    tests_total += 1
    has_is_active = hasattr(session, 'is_active')
    tests_passed += test("SessionState NÃO tem is_active", not has_is_active)
    
    tests_total += 1
    has_waiting = hasattr(session, 'waiting_confirmation')
    tests_passed += test("SessionState NÃO tem waiting_confirmation", not has_waiting)
    
    tests_total += 1
    tests_passed += test("SessionState tem phase (enum)", isinstance(session.phase, SessionPhase))
    
    tests_total += 1
    tests_passed += test("UnityCommunicator tem server_state (enum)", isinstance(comm.server_state, ServerState))
    
    # ====================================================================
    # TESTE 7: Helpers de Query de Estado
    # ====================================================================
    section("TESTE 7: Helpers de Query de Estado")
    
    comm = UnityCommunicator()
    
    tests_total += 1
    tests_passed += test("_is_server_operational() False (parado)", not comm._is_server_operational())
    
    tests_total += 1
    comm.start_server()
    tests_passed += test("_is_server_operational() True (rodando)", comm._is_server_operational())
    
    tests_total += 1
    tests_passed += test("_is_session_waiting_trigger() False (IDLE)", not comm._is_session_waiting_trigger())
    
    tests_total += 1
    comm.session.transition_to(SessionPhase.SETUP)
    comm.session.transition_to(SessionPhase.READY)
    tests_passed += test("_is_session_waiting_trigger() True (READY)", comm._is_session_waiting_trigger())
    
    tests_total += 1
    tests_passed += test("_is_session_active_for_commands() False (READY)", not comm._is_session_active_for_commands())
    
    tests_total += 1
    comm.session.transition_to(SessionPhase.ACTIVE)
    tests_passed += test("_is_session_active_for_commands() True (ACTIVE)", comm._is_session_active_for_commands())
    
    comm.stop_server()
    
    # ====================================================================
    # TESTE 8: Fallback e Recuperação de Erro
    # ====================================================================
    section("TESTE 8: Fallback e Recuperação de Erro")
    
    session = SessionState()
    
    tests_total += 1
    session.transition_to(SessionPhase.SETUP)
    tests_passed += test("Pode transicionar para SETUP", session.phase == SessionPhase.SETUP)
    
    tests_total += 1
    session.transition_to(SessionPhase.IDLE)  # Voltar se falhar
    tests_passed += test("Pode voltar de SETUP para IDLE", session.phase == SessionPhase.IDLE)
    
    tests_total += 1
    session.transition_to(SessionPhase.SETUP)
    session.transition_to(SessionPhase.READY)
    session.transition_to(SessionPhase.IDLE)  # Cancelar antes de iniciar
    tests_passed += test("Pode cancelar de READY para IDLE", session.phase == SessionPhase.IDLE)
    
    # ====================================================================
    # RESUMO
    # ====================================================================
    section("RESUMO DOS TESTES")
    
    percentage = (tests_passed / tests_total) * 100 if tests_total > 0 else 0
    
    print(f"Total de testes: {tests_total}")
    print(f"Passaram: {Colors.GREEN}{tests_passed}{Colors.END}")
    print(f"Falharam: {Colors.RED}{tests_total - tests_passed}{Colors.END}")
    print(f"Taxa de sucesso: {percentage:.1f}%\n")
    
    if tests_passed == tests_total:
        print(Colors.GREEN + Colors.BOLD + "✅ TODOS OS TESTES PASSARAM!" + Colors.END)
        print("\n📊 Arquitetura da Máquina de Estados:")
        print("   • SessionPhase (enum): IDLE -> SETUP -> READY -> ACTIVE -> ENDING -> IDLE")
        print("   • ServerState (enum): STOPPED -> RUNNING <-> CONNECTED")
        print("   • Transições são mutuamente exclusivas e validadas")
        print("   • Sem interdependências cruzadas (nada de is_active + waiting_confirmation)")
        print("   • Helpers centralizam lógica de queries")
        print("   • Fallback e recuperação de erro funcionam corretamente\n")
        return True
    else:
        print(Colors.RED + Colors.BOLD + f"❌ {tests_total - tests_passed} TESTES FALHARAM!" + Colors.END + "\n")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
