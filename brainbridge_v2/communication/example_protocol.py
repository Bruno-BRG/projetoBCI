"""
Exemplo de uso do protocolo de comunicação Sistema <-> VR
Este script demonstra como usar o protocolo completo para uma sessão de treino

Protocolo implementado:
1. Sistema: Broadcast UDP com "Confirm"
2. VR: Responde com "Header: Confirm"
3. Sistema: Envia Dados Paciente (Nome, Nível, Lado)
4. Sistema: Envia Tarefa ("Treino" ou "Jogo")
5. Sistema: Envia Trigger + LEFT/RIGHT_HAND_CLOSE
6. VR: Responde com LEFT_FLOWER ou RIGHT_FLOWER
7. Sistema: Envia END_TASK com mensagem
8. VR: Confirma finalização
"""

import time
from brainbridge_v2.communication.unity import (
    UnityCommunicator,
    PatientData,
    TaskType,
    ActionCommand
)


def exemplo_sessao_completa():
    """
    Demonstra uma sessão completa seguindo o protocolo Sistema <-> VR
    """
    print("="*70)
    print(" EXEMPLO: SESSÃO COMPLETA DE TREINO VR")
    print("="*70)
    
    # 1. Criar comunicador
    communicator = UnityCommunicator()
    
    # 2. Configurar callbacks para monitorar eventos
    def on_vr_connected(connected):
        if connected:
            print("\n✅ VR conectado via TCP!")
        else:
            print("\n❌ VR desconectado")
    
    def on_vr_confirmation():
        print("\n✅ VR confirmou recebimento dos dados!")
        print("🎯 Pronto para enviar trigger e iniciar tarefa")
    
    def on_flower_action(action: ActionCommand):
        print(f"\n🌸 VR acionou: {action.value}")
    
    def on_message(msg):
        print(f"\n📨 Mensagem do VR: {msg}")
    
    communicator.set_connection_callback(on_vr_connected)
    communicator.set_confirmation_callback(on_vr_confirmation)
    communicator.set_flower_callback(on_flower_action)
    communicator.set_message_callback(on_message)
    
    # 3. Iniciar servidor
    print("\n🚀 Iniciando servidor de comunicação...")
    if not communicator.start_server():
        print("❌ Falha ao iniciar servidor")
        return
    
    print("✅ Servidor iniciado")
    print("📡 Aguardando VR conectar...")
    
    # 4. Aguardar VR conectar (max 10 segundos para teste)
    for i in range(10):
        if communicator.tcp_connected:
            break
        time.sleep(1)
        print(".", end="", flush=True)
    
    if not communicator.tcp_connected:
        print("\n⚠️  VR não conectou em 10 segundos")
        print("   (Em produção, aguarde manualmente)")
    
    print("\n")
    
    # 5. Configurar dados do paciente
    patient = PatientData(
        nome="João Silva",
        nivel=5,  # Nível de 0 a 11
        lado="Direito"
    )
    
    # 6. Iniciar sessão de treino
    print("\n📋 Iniciando sessão de treino...")
    if not communicator.start_session(patient, TaskType.TREINO):
        print("❌ Falha ao iniciar sessão")
        communicator.stop_server()
        return
    
    # 7. Aguardar confirmação do VR (max 5 segundos)
    print("\n⏳ Aguardando VR confirmar...")
    for i in range(5):
        if not communicator.session.waiting_confirmation:
            break
        time.sleep(1)
        print(".", end="", flush=True)
    
    # 8. Enviar trigger para iniciar
    print("\n\n🎯 Enviando trigger para iniciar tarefa...")
    time.sleep(1)  # Pequeno delay
    if not communicator.send_trigger():
        print("❌ Falha ao enviar trigger")
        communicator.stop_server()
        return
    
    print("\n✅ Tarefa iniciada no VR!")
    print("\n" + "="*70)
    print(" SESSÃO EM EXECUÇÃO - Simulando comandos durante treino")
    print("="*70)
    
    # 9. Simular comandos durante a sessão
    comandos_exemplo = [
        ("Fechar mão direita", lambda: communicator.send_hand_close("direita")),
        ("Ação flor direita", lambda: communicator.send_flower_action("direita")),
        ("Fechar mão esquerda", lambda: communicator.send_hand_close("esquerda")),
        ("Ação flor esquerda", lambda: communicator.send_flower_action("esquerda")),
    ]
    
    for descricao, comando in comandos_exemplo:
        time.sleep(2)
        print(f"\n📤 {descricao}...")
        comando()
    
    # 10. Finalizar sessão
    time.sleep(3)
    print("\n" + "="*70)
    print(" FINALIZANDO SESSÃO")
    print("="*70)
    
    if not communicator.end_task("Treino completado com sucesso"):
        print("❌ Falha ao finalizar tarefa")
    
    # 11. Aguardar confirmação de finalização
    print("\n⏳ Aguardando confirmação de finalização do VR...")
    time.sleep(3)
    
    # 12. Parar servidor
    print("\n🛑 Parando servidor...")
    communicator.stop_server()
    
    print("\n" + "="*70)
    print(" ✅ EXEMPLO CONCLUÍDO COM SUCESSO")
    print("="*70 + "\n")


def exemplo_sessao_interativa():
    """
    Demonstra uso interativo do protocolo com input do usuário
    """
    communicator = UnityCommunicator()
    
    # Configurar callbacks
    def on_connection(connected):
        status = "🟢 Conectado" if connected else "🔴 Desconectado"
        print(f"\n[Evento] VR {status}")
    
    def on_confirmation():
        print("\n[Evento] ✅ Confirmação recebida do VR")
    
    communicator.set_connection_callback(on_connection)
    communicator.set_confirmation_callback(on_confirmation)
    
    # Iniciar servidor
    print("\n🚀 Iniciando servidor...")
    if not communicator.start_server():
        print("❌ Falha ao iniciar servidor")
        return
    
    print("✅ Servidor ativo")
    print("📡 Aguardando VR conectar...\n")
    
    input("Pressione ENTER após o VR conectar...")
    
    # Coletar dados do paciente
    print("\n📋 Configuração da Sessão:")
    nome = input("Nome do paciente: ").strip() or "Paciente Teste"
    
    nivel_input = input("Nível (0-11): ").strip() or "5"
    try:
        nivel = int(nivel_input)
        if not (0 <= nivel <= 11):
            print("⚠️  Nível fora do range, usando 5")
            nivel = 5
    except ValueError:
        print("⚠️  Nível inválido, usando 5")
        nivel = 5
    
    lado = input("Lado afetado (Esquerdo/Direito): ").strip() or "Direito"
    tarefa = input("Tarefa (Treino/Jogo): ").strip() or "Treino"
    
    # Criar dados do paciente
    patient = PatientData(nome=nome, nivel=nivel, lado=lado.capitalize())
    task_type = TaskType.TREINO if tarefa.lower() == "treino" else TaskType.JOGO
    
    # Iniciar sessão
    print("\n🚀 Iniciando sessão...")
    if not communicator.start_session(patient, task_type):
        print("❌ Falha ao iniciar sessão")
        communicator.stop_server()
        return
    
    input("\nPressione ENTER após VR confirmar para enviar trigger...")
    
    # Enviar trigger
    if not communicator.send_trigger():
        print("❌ Falha ao enviar trigger")
        communicator.stop_server()
        return
    
    print("\n✅ Sessão iniciada!")
    print("\n📋 Comandos disponíveis durante a sessão:")
    print("  - fechar direita/esquerda")
    print("  - flor direita/esquerda")
    print("  - fim")
    
    # Loop de comandos
    while communicator.session.is_active:
        comando = input("\n> ").strip().lower()
        
        if not comando:
            continue
        
        if comando == "fim":
            msg = input("Mensagem de finalização (opcional): ").strip()
            communicator.end_task(msg if msg else "")
            break
        elif "fechar" in comando:
            lado = "direita" if "direita" in comando else "esquerda"
            communicator.send_hand_close(lado)
        elif "flor" in comando:
            lado = "direita" if "direita" in comando else "esquerda"
            communicator.send_flower_action(lado)
        else:
            print(f"⚠️  Comando não reconhecido: {comando}")
    
    print("\n🏁 Sessão finalizada")
    communicator.stop_server()
    print("✅ Programa encerrado\n")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print(" EXEMPLOS DO PROTOCOLO SISTEMA <-> VR")
    print("="*70)
    print("\nEscolha um exemplo:")
    print("  1 - Sessão completa automática (demonstração)")
    print("  2 - Sessão interativa (com inputs)")
    print("  3 - Sair")
    
    escolha = input("\nOpção: ").strip()
    
    if escolha == "1":
        print("\n⚠️  AVISO: Este exemplo requer VR conectado e respondendo!")
        input("Pressione ENTER para continuar ou Ctrl+C para cancelar...")
        exemplo_sessao_completa()
    elif escolha == "2":
        exemplo_sessao_interativa()
    else:
        print("\n👋 Até logo!\n")
        sys.exit(0)
