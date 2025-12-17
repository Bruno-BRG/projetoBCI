"""
Script diagnóstico para identificar desconexão entre TensorFlow e Unity

Executa verificações sequenciais:
1. TensorFlow e modelos disponíveis
2. Predictor funcionando
3. UnityCommunicator pronto
4. Fluxo completo de previsão
"""

import sys
import os
from pathlib import Path
import numpy as np
import traceback

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# Fix encoding para Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_tensorflow_available():
    """Verifica se TensorFlow está disponível"""
    print("\n" + "="*60)
    print("CHECK 1: TensorFlow disponível?")
    print("="*60)
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} encontrado")
        return True
    except ImportError as e:
        print(f"✗ TensorFlow NÃO disponível: {e}")
        return False


def check_keras_models_available():
    """Verifica se há modelos Keras disponíveis"""
    print("\n" + "="*60)
    print("CHECK 2: Modelos Keras disponíveis?")
    print("="*60)
    
    model_dirs = [
        BASE_DIR / 'brainbridge_v2' / 'models',
        BASE_DIR / 'models',
        BASE_DIR / 'files',
        BASE_DIR / 'HardThinking' / 'files',
    ]
    
    found_models = []
    for model_dir in model_dirs:
        if model_dir.exists():
            models = list(model_dir.glob('*.keras')) + list(model_dir.glob('*.h5'))
            if models:
                for model in models:
                    found_models.append(str(model))
                    print(f"  ✓ Encontrado: {model.name}")
    
    if not found_models:
        print("✗ Nenhum modelo Keras (.keras/.h5) encontrado")
        print(f"  Diretórios verificados:")
        for d in model_dirs:
            print(f"    - {d}")
        return False, []
    
    return True, found_models


def check_tensorflow_adapter():
    """Verifica se TensorFlowMLAdapter funciona"""
    print("\n" + "="*60)
    print("CHECK 3: TensorFlowMLAdapter funcionando?")
    print("="*60)
    
    try:
        from ml.tensorflow_adapter import TensorFlowMLAdapter
        adapter = TensorFlowMLAdapter()
        print(f"✓ TensorFlowMLAdapter inicializado: {adapter}")
        
        # Tentar carregar um modelo se disponível
        has_models, models = check_keras_models_available()
        if has_models and models:
            print(f"\n  Tentando carregar modelo: {Path(models[0]).name}")
            try:
                model = adapter.load_model(models[0])
                print(f"  ✓ Modelo carregado com sucesso!")
                
                # Tentar uma predição
                test_window = np.random.randn(250, 16).astype('float32')
                result = adapter.predict_on_window(test_window)
                print(f"  ✓ Predição bem-sucedida:")
                print(f"    - Label: {result['label']}")
                print(f"    - Probs: {result['probs']}")
                return True, models[0]
            except Exception as e:
                print(f"  ✗ Erro ao carregar/predizer: {e}")
                traceback.print_exc()
                return False, None
        else:
            print("  ⚠ Nenhum modelo disponível para testar")
            return True, None
            
    except Exception as e:
        print(f"✗ Erro ao inicializar TensorFlowMLAdapter: {e}")
        traceback.print_exc()
        return False, None


def check_predictor():
    """Verifica se Predictor funciona"""
    print("\n" + "="*60)
    print("CHECK 4: Predictor funcionando?")
    print("="*60)
    
    try:
        from ml.predictor import Predictor
        print("✓ Predictor import bem-sucedido")
        
        # Tentar usar com um modelo mock
        from unittest.mock import MagicMock, patch
        
        with patch('ml.models.load_keras_model') as mock_load:
            mock_model = MagicMock()
            mock_probs = np.array([[0.4, 0.6]])
            mock_model.predict.return_value = mock_probs
            mock_load.return_value = mock_model
            
            predictor = Predictor("dummy.keras")
            window = np.random.randn(250, 16)
            result = predictor.predict_window(window)
            
            print(f"✓ Predictor.predict_window() funcionando:")
            print(f"  - Result: {result}")
            
            return True
            
    except Exception as e:
        print(f"✗ Erro no Predictor: {e}")
        traceback.print_exc()
        return False


def check_unity_communicator():
    """Verifica se UnityCommunicator está pronto"""
    print("\n" + "="*60)
    print("CHECK 5: UnityCommunicator pronto?")
    print("="*60)
    
    try:
        from communication.unity import UnityCommunicator, ActionCommand, PatientData, TaskType
        
        # Verificar enums
        print("✓ ActionCommand enum:")
        for cmd in ActionCommand:
            print(f"  - {cmd.name}: {cmd.value}")
        
        print("\n✓ TaskType enum:")
        for task in TaskType:
            print(f"  - {task.name}: {task.value}")
        
        # Testar criação de UnityCommunicator
        comm = UnityCommunicator()
        print(f"\n✓ UnityCommunicator inicializado")
        print(f"  - is_active: {comm.is_active}")
        print(f"  - tcp_connected: {comm.tcp_connected}")
        
        # Testar PatientData
        patient = PatientData("Teste", 10, "Direito")
        print(f"\n✓ PatientData criado: {patient}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no UnityCommunicator: {e}")
        traceback.print_exc()
        return False


def check_prediction_to_command_flow():
    """Verifica fluxo completo: predição -> comando"""
    print("\n" + "="*60)
    print("CHECK 6: Fluxo predição -> comando?")
    print("="*60)
    
    try:
        from ml.predictor import Predictor
        from communication.unity import ActionCommand, UnityCommunicator
        from unittest.mock import MagicMock, patch
        
        # Setup predictor mock
        with patch('ml.models.load_keras_model') as mock_load:
            mock_model = MagicMock()
            
            # Simular duas previsões
            predictions = [
                np.array([[0.9, 0.1]]),  # Left
                np.array([[0.1, 0.9]]),  # Right
            ]
            
            for pred_prob in predictions:
                # Reset mock
                mock_model.predict.return_value = pred_prob
                mock_load.return_value = mock_model
                
                predictor = Predictor("dummy.keras")
                window = np.random.randn(250, 16)
                result = predictor.predict_window(window)
                
                # Mapear predição para comando
                if result['label'] == 'left':
                    command = ActionCommand.LEFT_HAND_CLOSE.value
                    print(f"\n✓ Predição LEFT -> comando '{command}'")
                else:
                    command = ActionCommand.RIGHT_HAND_CLOSE.value
                    print(f"\n✓ Predição RIGHT -> comando '{command}'")
                
                print(f"  - Probabilidades: {result['probs']}")
                print(f"  - Comando: {command}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no fluxo: {e}")
        traceback.print_exc()
        return False


def check_udp_sender_compatibility():
    """Verifica compatibilidade de UDP_sender"""
    print("\n" + "="*60)
    print("CHECK 7: UDP_sender compatível?")
    print("="*60)
    
    try:
        from communication.unity import UDP_sender, UnityCommunicator
        from unittest.mock import MagicMock, patch
        
        print("✓ UDP_sender import bem-sucedido")
        
        # Testar métodos principais
        print("\n✓ Métodos UDP_sender disponíveis:")
        print(f"  - enviar_sinal: {hasattr(UDP_sender, 'enviar_sinal')}")
        print(f"  - init_zmq_socket: {hasattr(UDP_sender, 'init_zmq_socket')}")
        print(f"  - stop_zmq_socket: {hasattr(UDP_sender, 'stop_zmq_socket')}")
        print(f"  - is_server_active: {hasattr(UDP_sender, 'is_server_active')}")
        
        # Testar mapeamento de ações
        with patch.object(UnityCommunicator, 'send_hand_command') as mock_send:
            mock_send.return_value = True
            
            print("\n✓ Testando mapeamento de ações:")
            UDP_sender.enviar_sinal('direita')
            print(f"  - 'direita' mapeado para: send_hand_command('direita')")
            
            UDP_sender.enviar_sinal('esquerda')
            print(f"  - 'esquerda' mapeado para: send_hand_command('esquerda')")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro no UDP_sender: {e}")
        traceback.print_exc()
        return False


def check_streaming_widget_integration():
    """Verifica integração com StreamingWidget"""
    print("\n" + "="*60)
    print("CHECK 8: StreamingWidget integração?")
    print("="*60)
    
    try:
        # Verificar imports necessários
        print("Verificando imports do StreamingWidget...")
        
        # Não vamos importar a GUI aqui (requer PyQt5 e display)
        # Mas verificaremos se os métodos existem
        
        from pathlib import Path
        streaming_file = BASE_DIR / 'gui' / 'widgets' / 'streaming.py'
        
        if not streaming_file.exists():
            print(f"✗ Arquivo não encontrado: {streaming_file}")
            return False
        
        content = streaming_file.read_text(encoding='utf-8', errors='ignore')
        
        # Verificar funções-chave
        checks = {
            'send_udp_signal': 'def send_udp_signal',
            'predict_movement': 'def predict_movement',
            'send_esp32_signal': 'def send_esp32_signal',
            'on_data_received': 'def on_data_received',
        }
        
        print("\n✓ Funções encontradas no StreamingWidget:")
        for name, pattern in checks.items():
            if pattern in content:
                print(f"  - {name}: ✓")
            else:
                print(f"  - {name}: ✗")
        
        # Verificar se predict_movement chama send_udp_signal
        if 'self.send_udp_signal' in content:
            print("\n✓ StreamingWidget.predict_movement chama send_udp_signal")
        else:
            print("\n✗ StreamingWidget.predict_movement NÃO chama send_udp_signal")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Erro ao verificar StreamingWidget: {e}")
        traceback.print_exc()
        return False


def main():
    """Executa todos os checks diagnósticos"""
    print("\n" + "#"*60)
    print("# DIAGNÓSTICO: Integração TensorFlow <-> Unity")
    print("#"*60)
    
    results = {}
    
    # Executar checks
    results['tensorflow'] = check_tensorflow_available()
    results['models'], models_list = check_keras_models_available()
    results['adapter'], model_path = check_tensorflow_adapter()
    results['predictor'] = check_predictor()
    results['unity'] = check_unity_communicator()
    results['flow'] = check_prediction_to_command_flow()
    results['udp_sender'] = check_udp_sender_compatibility()
    results['streaming_widget'] = check_streaming_widget_integration()
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DOS CHECKS")
    print("="*60)
    
    all_pass = True
    for check_name, result in results.items():
        if isinstance(result, tuple):
            result = result[0]
        
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not result:
            all_pass = False
    
    print("\n" + "="*60)
    
    if all_pass:
        print("✓ TODOS OS CHECKS PASSARAM!")
        print("\nPróximos passos:")
        print("1. Executar: python -m pytest tests/test_tensorflow_unity_integration.py -v")
        print("2. Verificar logs da GUI ao fazer predições")
        print("3. Testar comunicação Unity em time.sleep ou simular")
    else:
        print("✗ ALGUNS CHECKS FALHARAM")
        print("\nProblemas encontrados:")
        print("1. Verifique a instalação do TensorFlow se CHECK 1 falhou")
        print("2. Coloque modelos em brainbridge_v2/models/ se CHECK 2 falhou")
        print("3. Verifique imports e dependências se outros CHECKs falharam")
    
    print("\n" + "#"*60 + "\n")
    
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
