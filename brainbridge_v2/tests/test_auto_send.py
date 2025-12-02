# test_auto_send.py
# Execute a partir da raiz do repositório (onde fica a pasta brainbridge_v2)

import time
from brainbridge_v2.communication.unity import UnityCommunicator, PatientData, TaskType

def main():
    comm = UnityCommunicator()
    ok = comm.start_server()
    if not ok:
        print("Falha ao iniciar o servidor. Verifique logs.", flush=True)
        return

    # --- CONFIGURE AQUI os dados do paciente ANTES do Unity conectar ---
    comm.session.patient = PatientData("João Silva", 11, "Direito")
    comm.session.task_type = TaskType.TREINO

    print("Servidor iniciado e sessão configurada. Aguardando Unity conectar e enviar HEADER...", flush=True)

    try:
        # Mantém o processo ativo para receber conexões do Unity.
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário. Parando servidor...", flush=True)
    finally:
        comm.stop_server()
        print("Servidor parado.", flush=True)

if __name__ == "__main__":
    main()
