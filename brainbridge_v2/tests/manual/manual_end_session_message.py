"""
Teste da mensagem padrão de finalização
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from brainbridge_v2.infrastructure.communication.unity import (
    UnityCommunicator,
    PatientData,
    TaskType
)

print("🧪 Testando mensagem de finalização\n")

comm = UnityCommunicator()
comm.start_server()

# Configurar uma sessão fictícia
patient = PatientData("João Silva", 5, "Direito")
comm.session.patient = patient
comm.session.task_type = TaskType.TREINO
comm.session.is_active = True

print("="*70)
print("Teste 1: Finalizar SEM mensagem (deve usar padrão)")
print("="*70)

# Capturar a mensagem que seria enviada
original_send = comm._send_protocol_message
messages_sent = []

def mock_send(msg):
    messages_sent.append(msg)
    return True

comm._send_protocol_message = mock_send
comm.end_session()  # Sem mensagem
comm._send_protocol_message = original_send

print("\n📨 Mensagem enviada:")
print(messages_sent[0])
print()

# Verificar se contém a mensagem padrão
if "Parabens voce esta mandando muito bem" in messages_sent[0]:
    print("✅ Mensagem padrão correta!")
else:
    print("❌ Mensagem padrão não encontrada")

print("\n" + "="*70)
print("Teste 2: Finalizar COM mensagem customizada")
print("="*70)

# Resetar sessão
comm.session.is_active = True
messages_sent.clear()

comm._send_protocol_message = mock_send
comm.end_session("Você superou todas as expectativas!")
comm._send_protocol_message = original_send

print("\n📨 Mensagem enviada:")
print(messages_sent[0])
print()

if "Você superou todas as expectativas!" in messages_sent[0]:
    print("✅ Mensagem customizada correta!")
else:
    print("❌ Mensagem customizada não encontrada")

comm.stop_server()

print("\n" + "="*70)
print("🎉 Testes de mensagem de finalização concluídos!")
print("="*70)
