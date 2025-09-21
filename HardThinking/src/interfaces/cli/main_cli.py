"""
Interface CLI Principal - Mantém compatibilidade com o sistema original
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Optional, Any
from pathlib import Path

# Adiciona path para importações
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

# Importações da arquitetura hexagonal
import numpy as np
from ...config import get_config
from ...application.use_cases.train_model_use_case import TrainModelUseCase, TrainModelRequest
from ...domain.value_objects.training_types import TrainingStrategy
from ...domain.entities.model import ModelArchitecture
from ...infrastructure.adapters.filesystem_adapter import LocalFileSystemAdapter
# Tenta importar o adapter TensorFlow; se não disponível, usa um stub que
# fornece mensagens claras ao ser usado. Isso evita que a importação do CLI
# quebre em ambientes sem TensorFlow (ex: falta de DLLs no Windows).
try:
    from ...infrastructure.adapters.tensorflow_ml_adapter import TensorFlowMLAdapter
except Exception:
    TensorFlowMLAdapter = None
    # stub simples que segue a interface esperada, mas falha com mensagem útil
    class TensorFlowMLAdapterStub:
        def __init__(self, config):
            self.available = False

        def create_model(self, *args, **kwargs):
            raise RuntimeError("TensorFlow não disponível no ambiente. Verifique a instalação do TensorFlow e dependências de sistema.")

        def train_model(self, *args, **kwargs):
            raise RuntimeError("TensorFlow não disponível no ambiente. Treinamento não pode ser executado.")

        def predict(self, *args, **kwargs):
            raise RuntimeError("TensorFlow não disponível no ambiente. Previsões não podem ser realizadas.")

        def evaluate_model(self, *args, **kwargs):
            raise RuntimeError("TensorFlow não disponível no ambiente. Avaliação não pode ser realizada.")

        def save_model(self, *args, **kwargs):
            raise RuntimeError("TensorFlow não disponível no ambiente. Salvamento de modelo indisponível.")

        def load_model(self, *args, **kwargs):
            raise RuntimeError("TensorFlow não disponível no ambiente. Carregamento de modelo indisponível.")

    TensorFlowMLAdapter = TensorFlowMLAdapterStub
from ...infrastructure.adapters.logging_adapter import PythonLoggingAdapter
from ...infrastructure.repositories.eeg_repository_impl import FileSystemEEGRepository, FileSystemSubjectRepository

# Importações de compatibilidade (wrapping do sistema antigo)
try:
    # Tenta importar do sistema antigo para compatibilidade
    sys.path.append(str(current_dir.parent.parent / "treino_modelo"))
    from analyze_eeg_data import analyze_all_data, generate_analysis_report
    from test_eeg_model import batch_test_model
    from eeg_classifier_integration import create_classifier_from_models_dir
except ImportError:
    # Se não conseguir importar, cria stubs
    def analyze_all_data(): return {}
    def generate_analysis_report(data): return "Análise não disponível"
    def batch_test_model(*args): return {}
    def create_classifier_from_models_dir(*args): return None

# Cores para terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class CLIInterface:
    """Interface CLI principal"""

    def __init__(self):
        self.config = get_config()
        self._setup_dependencies()

    def _setup_dependencies(self):
        """Configura dependências da arquitetura hexagonal"""
        # Adapters: use data directory as base for filesystem operations so paths are relative to the data root
        self.filesystem = LocalFileSystemAdapter(str(self.config.directories.data_dir))
        self.ml_adapter = TensorFlowMLAdapter(self.config)
        self.logger = PythonLoggingAdapter(str(self.config.directories.logs_dir))

        # Repositórios
        self.eeg_repository = FileSystemEEGRepository(
            str(self.config.directories.data_dir),
            self.filesystem
        )
        self.subject_repository = FileSystemSubjectRepository(
            str(self.config.directories.data_dir),
            self.filesystem
        )

        # Importa repositório de modelos
        from ...infrastructure.repositories.model_repository_impl import FileSystemModelRepository
        self.model_repository = FileSystemModelRepository(
            str(self.config.directories.models_dir),
            self.filesystem
        )

        # Notification stub (implementar se necessário)
        self.notification_adapter = NotificationStub()

        # Casos de uso
        self.train_model_use_case = TrainModelUseCase(
            self.eeg_repository,
            self.model_repository,  # Agora incluindo o repositório de modelos
            self.subject_repository,
            self.ml_adapter,
            self.logger,
            self.notification_adapter
        )

    def print_banner(self):
        """Exibe o banner do sistema"""
        banner = f"""
{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  {Colors.BOLD}██╗  ██╗ █████╗ ██████╗ ██████╗ ████████╗██╗  ██╗██╗███╗   ██╗██╗  ██╗██╗███╗   ██╗ ██████╗{Colors.ENDC}{Colors.CYAN} ║
║  {Colors.BOLD}██║  ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║  ██║██║████╗  ██║██║ ██╔╝██║████╗  ██║██╔════╝{Colors.ENDC}{Colors.CYAN} ║
║  {Colors.BOLD}███████║███████║██████╔╝██║  ██║   ██║   ███████║██║██╔██╗ ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗{Colors.ENDC}{Colors.CYAN}║
║  {Colors.BOLD}██╔══██║██╔══██║██╔══██╗██║  ██║   ██║   ██╔══██║██║██║╚██╗██║██╔═██╗ ██║██║╚██╗██║██║   ██║{Colors.ENDC}{Colors.CYAN}║
║  {Colors.BOLD}██║  ██║██║  ██║██║  ██║██████╔╝   ██║   ██║  ██║██║██║ ╚████║██║  ██╗██║██║ ╚████║╚██████╔╝{Colors.ENDC}{Colors.CYAN}║
║  {Colors.BOLD}╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝{Colors.ENDC}{Colors.CYAN} ║
║                                                                               ║
║  {Colors.BOLD}Sistema de Treinamento EEG Motor Imagery - Arquitetura Hexagonal{Colors.ENDC}{Colors.CYAN}            ║
║  {Colors.BOLD}Refatoração do BrainBridge com Clean Architecture{Colors.ENDC}{Colors.CYAN}                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
        print(banner)

    def show_main_menu(self):
        """Exibe menu principal"""
        menu = f"""
{Colors.BOLD}═══════════════════════════════════════════════════════════════════════════════
                              MENU PRINCIPAL
═══════════════════════════════════════════════════════════════════════════════{Colors.ENDC}

{Colors.CYAN}📊 TREINAMENTO DE MODELOS{Colors.ENDC}
  {Colors.GREEN}1.{Colors.ENDC} Treinar modelo para sujeito único
  {Colors.GREEN}2.{Colors.ENDC} Treinar com validação cruzada
  {Colors.GREEN}3.{Colors.ENDC} Treinar com Leave-One-Out validation
  {Colors.GREEN}4.{Colors.ENDC} Treinar modelo multi-sujeitos

{Colors.CYAN}📈 ANÁLISE DE DADOS{Colors.ENDC}
  {Colors.GREEN}5.{Colors.ENDC} Analisar dados de todos os sujeitos
  {Colors.GREEN}6.{Colors.ENDC} Analisar dados de sujeito específico
  {Colors.GREEN}7.{Colors.ENDC} Gerar relatório de análise

{Colors.CYAN}🧪 TESTE DE MODELOS{Colors.ENDC}
  {Colors.GREEN}8.{Colors.ENDC} Testar modelo existente
  {Colors.GREEN}9.{Colors.ENDC} Testar todos os modelos
  {Colors.GREEN}10.{Colors.ENDC} Comparar modelos

{Colors.CYAN}🔧 UTILIDADES{Colors.ENDC}
  {Colors.GREEN}11.{Colors.ENDC} Criar classificador integrado
  {Colors.GREEN}12.{Colors.ENDC} Informações do sistema
  {Colors.GREEN}13.{Colors.ENDC} Configurações

{Colors.RED}0.{Colors.ENDC} Sair

{Colors.BOLD}═══════════════════════════════════════════════════════════════════════════════{Colors.ENDC}
"""
        print(menu)

    def get_available_subjects(self) -> List[str]:
        """Obtém lista de sujeitos disponíveis"""
        try:
            return self.eeg_repository.get_all_subjects()
        except Exception as e:
            self.logger.log_error("Erro ao listar sujeitos", e)
            return []

    def select_subjects(self, multi_select: bool = False) -> List[str]:
        """Interface para seleção de sujeitos"""
        subjects = self.get_available_subjects()

        if not subjects:
            print(f"{Colors.RED}❌ Nenhum sujeito encontrado nos dados.{Colors.ENDC}")
            return []

        print(f"\n{Colors.CYAN}📋 Sujeitos disponíveis:{Colors.ENDC}")
        for i, subject in enumerate(subjects, 1):
            print(f"  {Colors.GREEN}{i}.{Colors.ENDC} {subject}")

        if multi_select:
            print(f"\n{Colors.YELLOW}💡 Digite os números dos sujeitos separados por vírgula (ex: 1,2,3):{Colors.ENDC}")
            choice = input("Seleção: ").strip()

            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected = [subjects[i] for i in indices if 0 <= i < len(subjects)]
                return selected
            except:
                print(f"{Colors.RED}❌ Seleção inválida.{Colors.ENDC}")
                return []
        else:
            choice = input("Selecione um sujeito (número): ").strip()
            try:
                index = int(choice) - 1
                if 0 <= index < len(subjects):
                    return [subjects[index]]
            except:
                pass

            print(f"{Colors.RED}❌ Seleção inválida.{Colors.ENDC}")
            return []

    def train_single_subject_model(self):
        """Treinar modelo para sujeito único"""
        print(f"\n{Colors.BOLD}🎯 Treinamento para Sujeito Único{Colors.ENDC}")

        subjects = self.select_subjects(multi_select=False)
        if not subjects:
            return

        subject_id = subjects[0]

        request = TrainModelRequest(
            subject_ids=[subject_id],
            strategy=TrainingStrategy.SINGLE_SUBJECT,
            model_architecture=ModelArchitecture.CNN_1D,
            hyperparameters={
                'input_shape': (250, 16),
                'num_classes': 2,
                'learning_rate': 0.001
            }
        )

        print(f"\n{Colors.CYAN}🚀 Iniciando treinamento para {subject_id}...{Colors.ENDC}")

        response = self.train_model_use_case.execute(request)

        if response.success:
            print(f"\n{Colors.GREEN}✅ Treinamento concluído com sucesso!{Colors.ENDC}")
            print(f"   Acurácia: {response.validation_result.validation_performance.accuracy:.3f}")
            print(f"   F1-Score: {response.validation_result.validation_performance.f1_score:.3f}")
        else:
            print(f"\n{Colors.RED}❌ Erro no treinamento: {response.error_message}{Colors.ENDC}")

    def train_cross_validation_model(self):
        """Treinar com validação cruzada"""
        print(f"\n{Colors.BOLD}🔄 Treinamento com Validação Cruzada{Colors.ENDC}")

        subjects = self.select_subjects(multi_select=True)
        if not subjects:
            return

        cv_folds = input("Número de folds (padrão 5): ").strip()
        try:
            cv_folds = int(cv_folds) if cv_folds else 5
        except:
            cv_folds = 5

        request = TrainModelRequest(
            subject_ids=subjects,
            strategy=TrainingStrategy.CROSS_VALIDATION,
            model_architecture=ModelArchitecture.CNN_1D,
            cv_folds=cv_folds,
            hyperparameters={
                'input_shape': (250, 16),
                'num_classes': 2,
                'learning_rate': 0.001
            }
        )

        print(f"\n{Colors.CYAN}🚀 Iniciando validação cruzada ({cv_folds} folds)...{Colors.ENDC}")

        response = self.train_model_use_case.execute(request)

        if response.success:
            print(f"\n{Colors.GREEN}✅ Validação cruzada concluída!{Colors.ENDC}")
            print(f"   Acurácia média: {response.validation_result.validation_performance.accuracy:.3f}")
            if response.validation_result.cross_validation_scores:
                scores = response.validation_result.cross_validation_scores
                print(f"   Scores CV: {[f'{s:.3f}' for s in scores]}")
                print(f"   Desvio padrão: {np.std(scores):.3f}")
        else:
            print(f"\n{Colors.RED}❌ Erro na validação: {response.error_message}{Colors.ENDC}")

    def show_system_info(self):
        """Mostra informações do sistema"""
        print(f"\n{Colors.BOLD}ℹ️  Informações do Sistema{Colors.ENDC}")

        info = self.config.get_system_info()

        for key, value in info.items():
            print(f"  {Colors.CYAN}{key.replace('_', ' ').title()}:{Colors.ENDC} {value}")

        # Informações adicionais
        subjects = self.get_available_subjects()
        print(f"  {Colors.CYAN}Sujeitos Disponíveis:{Colors.ENDC} {len(subjects)}")

        if subjects:
            print(f"  {Colors.CYAN}Lista de Sujeitos:{Colors.ENDC} {', '.join(subjects)}")

    def run(self):
        """Executa a interface CLI"""
        if self.config.cli.banner_enabled:
            self.print_banner()
        
        while True:
            self.show_main_menu()
            
            choice = input(f"{Colors.BOLD}Digite sua escolha: {Colors.ENDC}").strip()
            
            try:
                if choice == "0":
                    print(f"\n{Colors.CYAN}👋 Até logo!{Colors.ENDC}")
                    break
                elif choice == "1":
                    self.train_single_subject_model()
                elif choice == "2":
                    self.train_cross_validation_model()
                elif choice == "3":
                    print(f"{Colors.YELLOW}⚠️ Leave-One-Out em desenvolvimento...{Colors.ENDC}")
                elif choice == "4":
                    print(f"{Colors.YELLOW}⚠️ Multi-sujeitos em desenvolvimento...{Colors.ENDC}")
                elif choice == "5":
                    print(f"{Colors.YELLOW}⚠️ Análise geral em desenvolvimento...{Colors.ENDC}")
                elif choice == "6":
                    print(f"{Colors.YELLOW}⚠️ Análise de sujeito em desenvolvimento...{Colors.ENDC}")
                elif choice == "7":
                    print(f"{Colors.YELLOW}⚠️ Relatório em desenvolvimento...{Colors.ENDC}")
                elif choice == "8":
                    print(f"{Colors.YELLOW}⚠️ Teste de modelo em desenvolvimento...{Colors.ENDC}")
                elif choice == "9":
                    print(f"{Colors.YELLOW}⚠️ Teste de todos os modelos em desenvolvimento...{Colors.ENDC}")
                elif choice == "10":
                    print(f"{Colors.YELLOW}⚠️ Comparação de modelos em desenvolvimento...{Colors.ENDC}")
                elif choice == "11":
                    print(f"{Colors.YELLOW}⚠️ Classificador integrado em desenvolvimento...{Colors.ENDC}")
                elif choice == "12":
                    self.show_system_info()
                elif choice == "13":
                    print(f"{Colors.YELLOW}⚠️ Configurações em desenvolvimento...{Colors.ENDC}")
                else:
                    print(f"{Colors.RED}❌ Opção inválida. Tente novamente.{Colors.ENDC}")
                
                if choice != "0":
                    input(f"\n{Colors.CYAN}Pressione Enter para continuar...{Colors.ENDC}")
                    
            except KeyboardInterrupt:
                print(f"\n\n{Colors.CYAN}👋 Até logo!{Colors.ENDC}")
                break
            except Exception as e:
                self.logger.log_error("Erro na interface CLI", e)
                print(f"\n{Colors.RED}❌ Erro inesperado: {str(e)}{Colors.ENDC}")

class NotificationStub:
    """Stub para notificações"""
    def notify_training_started(self, model_id: str, subject_id: str):
        pass

    def notify_training_completed(self, model_id: str, metrics: Dict[str, float]):
        pass

    def notify_training_failed(self, model_id: str, error: str):
        pass

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="HardThinking - Sistema de Treinamento EEG")
    parser.add_argument("--data-dir", help="Diretório de dados")
    parser.add_argument("--no-banner", action="store_true", help="Desabilita banner")
    
    args = parser.parse_args()
    
    # Atualiza configuração se necessário
    config = get_config()
    if args.data_dir:
        config.directories.data_dir = Path(args.data_dir)
    if args.no_banner:
        config.cli.banner_enabled = False
    
    # Executa interface
    cli = CLIInterface()
    cli.run()

if __name__ == "__main__":
    main()
