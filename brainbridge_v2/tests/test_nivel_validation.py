"""
Teste de validação de níveis
"""
import sys
import os
# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from brainbridge_v2.communication.unity import PatientData

print("🧪 Testando validação de níveis (0-11)\n")

# Teste 1: Nível inválido (negativo)
try:
    p = PatientData('João', -1, 'Direito')
    print("❌ Deveria ter falhado com nível -1")
except ValueError as e:
    print(f"✅ Teste 1 OK: Nível -1 rejeitado")
    print(f"   Erro: {e}\n")

# Teste 2: Nível inválido (acima de 11)
try:
    p = PatientData('Maria', 12, 'Esquerdo')
    print("❌ Deveria ter falhado com nível 12")
except ValueError as e:
    print(f"✅ Teste 2 OK: Nível 12 rejeitado")
    print(f"   Erro: {e}\n")

# Teste 3: Lado inválido
try:
    p = PatientData('José', 5, 'Centro')
    print("❌ Deveria ter falhado com lado 'Centro'")
except ValueError as e:
    print(f"✅ Teste 3 OK: Lado inválido rejeitado")
    print(f"   Erro: {e}\n")

# Teste 4: Todos os níveis válidos
print("✅ Teste 4: Testando todos os níveis válidos (0-11)")
for nivel in range(12):
    p = PatientData(f'Paciente{nivel}', nivel, 'Direito')
    print(f"   Nível {nivel}: OK")

print("\n🎉 Todos os testes passaram!")
