"""
Teste rápido: Capturar output real da IA (previsões)

Este script:
1. Carrega o modelo Keras
2. Gera dados EEG fake
3. Faz previsões
4. Mostra os resultados
"""

import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from brainbridge_v2.infrastructure.ml.tensorflow_adapter import TensorFlowMLAdapter
from brainbridge_v2.infrastructure.ml.predictor import Predictor

def main():
    print("\n" + "="*60)
    print("TESTE: Capturar Output Real da IA")
    print("="*60)
    
    # 1. Carregar modelo
    print("\n1️⃣  Carregando modelo...")
    model_path = BASE_DIR / 'models' / 'modelo_full.keras'
    
    if not model_path.exists():
        print(f"❌ Modelo não encontrado em {model_path}")
        return False
    
    adapter = TensorFlowMLAdapter()
    try:
        model = adapter.load_model(str(model_path))
        print(f"✓ Modelo carregado: {model_path.name}")
    except Exception as e:
        print(f"❌ Erro ao carregar: {e}")
        return False
    
    # 2. Gerar dados EEG simulados
    print("\n2️⃣  Gerando dados EEG simulados...")
    
    # Simular várias janelas de EEG
    test_windows = [
        np.random.randn(250, 16).astype('float32'),  # Janela aleatória
        np.random.randn(250, 16).astype('float32'),  # Outra janela
        np.random.randn(250, 16).astype('float32'),  # Mais uma
    ]
    
    print(f"✓ Geradas {len(test_windows)} janelas de EEG")
    print(f"  - Cada janela: shape (250, 16)")
    print(f"  - 250 amostras @ 125Hz = 2 segundos")
    print(f"  - 16 canais EEG")
    
    # 3. Fazer previsões
    print("\n3️⃣  Fazendo previsões...")
    
    predictions = []
    for i, window in enumerate(test_windows):
        try:
            # Usar Predictor para fazer previsão
            predictor = Predictor(str(model_path))
            result = predictor.predict_window(window)
            
            predictions.append(result)
            
            print(f"\n  📊 Previsão {i+1}:")
            print(f"     - Label: {result['label']}")
            print(f"     - Probabilidades: LEFT={result['probs'][0]:.2%}, RIGHT={result['probs'][1]:.2%}")
            print(f"     - Confiança: {max(result['probs']):.2%}")
            
        except Exception as e:
            print(f"  ❌ Erro na previsão {i+1}: {e}")
            return False
    
    # 4. Mapear para comandos Unity
    print("\n4️⃣  Mapeando previsões para comandos Unity...")
    
    from brainbridge_v2.infrastructure.communication.unity import ActionCommand
    
    for i, pred in enumerate(predictions):
        if pred['label'] == 'left':
            cmd = ActionCommand.LEFT_HAND_CLOSE.value
        else:
            cmd = ActionCommand.RIGHT_HAND_CLOSE.value
        
        print(f"\n  🎮 Previsão {i+1} → Comando:")
        print(f"     - Predição: {pred['label']}")
        print(f"     - Comando: {cmd}")
        print(f"     - Probs: {pred['probs']}")
    
    # 5. Resumo
    print("\n" + "="*60)
    print("RESUMO: OUTPUT DA IA CAPTURADO COM SUCESSO ✓")
    print("="*60)
    print(f"\n✓ {len(predictions)} previsões feitas")
    print(f"✓ Todos os outputs capturados")
    print(f"✓ Mapeamento para Unity funcionando")
    print(f"\nProximos passos:")
    print(f"1. Enviar comandos via UnityCommunicator.send_command()")
    print(f"2. Confirmar recebimento no VR")
    print(f"3. Validar ações no jogo\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
