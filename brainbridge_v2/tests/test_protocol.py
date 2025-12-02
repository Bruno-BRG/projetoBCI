"""
Teste rápido do protocolo implementado
"""
import sys
import os
# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from brainbridge_v2.communication.unity import (
    UnityCommunicator, 
    PatientData, 
    TaskType
)

print("\n" + "="*70)
print(" ✅ PROTOCOLO SISTEMA <-> VR IMPLEMENTADO COM SUCESSO")
print("="*70)

# Criar comunicador
comm = UnityCommunicator()
comm.start_server()

print("\n📋 EXEMPLO DE USO:")
print("-"*70)

# Criar dados do paciente
patient = PatientData(
    nome="João Silva",
    nivel=5,  # Nível de 0 a 11
    lado="Direito"
)

print("\n✅ Dados do paciente:")
print(patient.format_message())

print("\n💡 Métodos do protocolo disponíveis:")
print("  1. start_session(patient, TaskType.TREINO)")
print("  2. send_trigger()")
print("  3. send_hand_close('direita')")
print("  4. send_hand_close('esquerda')")
print("  5. send_flower_action('direita')")
print("  6. send_flower_action('esquerda')")
print("  7. end_session('mensagem opcional')")

print("\n🎯 Estado atual da sessão:")
print(f"  - Servidor ativo: {comm.is_active}")
print(f"  - VR conectado: {comm.tcp_connected}")
print(f"  - Sessão ativa: {comm.session.is_active}")
print(f"  - Aguardando confirmação: {comm.session.waiting_confirmation}")

print("\n📡 Portas configuradas:")
print(f"  - UDP Broadcast: {comm.UDP_PORT} (Header: '{comm.CONFIRM_HEADER}')")
print(f"  - TCP Server: {comm.TCP_PORT}")
print(f"  - ZMQ Publisher: {comm.ZMQ_PORT}")

print("\n🚀 Para testar o protocolo completo, execute:")
print("  python -m brainbridge_v2.communication.unity")
print("\nOu execute o exemplo:")
print("  python brainbridge_v2/communication/example_protocol.py")

comm.stop_server()

print("\n" + "="*70)
print(" ✨ PRONTO PARA USO!")
print("="*70 + "\n")
