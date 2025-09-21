from PyQt5.QtCore import QThread, pyqtSignal
import os
import sys
import shutil
from pathlib import Path
import traceback
import logging
from datetime import datetime


class ModelTrainerThread(QThread):
    # Emite mensagens de progresso (string)
    progress_signal = pyqtSignal(str)
    # Emite sinal quando terminar: (success: bool, message: str)
    finished_signal = pyqtSignal(bool, str)
    # Emite o caminho do modelo gerado quando disponível
    model_path_signal = pyqtSignal(str)

    def __init__(self, csv_file_path: str, patient_id: int, parent=None):
        super().__init__(parent)
        self.csv_file_path = csv_file_path
        self.patient_id = patient_id

    def run(self):
        try:
            self.progress_signal.emit("Iniciando preparação de dados para o HardThinking...")

            # configurar logger de arquivo específico para este treinamento
            try:
                logs_dir = Path(__file__).parent.parent / 'logs'
                logs_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = logs_dir / f"trainer_{ts}.log"
                logger = logging.getLogger(f"ModelTrainer-{ts}")
                logger.setLevel(logging.DEBUG)
                # evitar múltiplos handlers se rerun
                if not logger.handlers:
                    fh = logging.FileHandler(str(log_file), encoding='utf-8')
                    fh.setLevel(logging.DEBUG)
                    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                    fh.setFormatter(formatter)
                    logger.addHandler(fh)
                logger.info("Inicializando thread de treinamento")
            except Exception:
                logger = None
                log_file = None

            # Localizar diretório HardThinking/src subindo a árvore a partir do projeto bci
            current_dir = Path(__file__).parent.parent
            hardthinking_src = None

            # First try: climb a few levels from current_dir
            probe = current_dir
            for _ in range(6):
                candidate = probe / 'HardThinking' / 'src'
                if candidate.exists():
                    hardthinking_src = candidate
                    break
                probe = probe.parent

            # Second try: full parent chain
            if hardthinking_src is None:
                for parent in current_dir.resolve().parents:
                    candidate = parent / 'HardThinking' / 'src'
                    if candidate.exists():
                        hardthinking_src = candidate
                        break

            # Third try: current working directory hierarchy
            if hardthinking_src is None:
                cwd = Path.cwd()
                probe = cwd
                for _ in range(6):
                    candidate = probe / 'HardThinking' / 'src'
                    if candidate.exists():
                        hardthinking_src = candidate
                        break
                    probe = probe.parent

            if hardthinking_src is None:
                msg = "Não foi possível localizar 'HardThinking/src'. Verifique se a pasta HardThinking está ao lado do projeto ou ajuste PYTHONPATH."
                if logger:
                    logger.error(msg)
                    logger.error(f"Procurado em: {current_dir}, cwd={Path.cwd()}")
                self.progress_signal.emit(msg)
                self.finished_signal.emit(False, f"{msg} (ver detalhes no log: {log_file})" if log_file else msg)
                return

            hardthinking_root = hardthinking_src.parent
            # Adiciona HardThinking ao sys.path para importações locais
            if str(hardthinking_root) not in sys.path:
                sys.path.insert(0, str(hardthinking_root))

            # Importar componentes do HardThinking
            self.progress_signal.emit("Importando módulos do HardThinking...")
            try:
                from src.config import get_config
                from src.interfaces.cli.main_cli import CLIInterface
                from src.application.use_cases.train_model_use_case import TrainModelRequest, TrainModelUseCase
                from src.domain.value_objects.training_types import TrainingStrategy
                from src.domain.entities.model import ModelArchitecture
            except Exception as e:
                tb = traceback.format_exc()
                msg = f"Falha ao importar módulos do HardThinking: {e}\n{tb}"
                if logger:
                    logger.error(msg)
                self.progress_signal.emit("Erro ao importar módulos do HardThinking. Veja log para detalhes.")
                self.finished_signal.emit(False, f"Falha ao importar módulos do HardThinking. Ver log: {log_file}" if log_file else msg)
                return

            # Copiar CSV de gravação para pasta de dados do HardThinking
            config = get_config()
            data_dir = Path(config.directories.data_dir)
            subjects_dir = data_dir / f"S{self.patient_id:03d}"
            subjects_dir.mkdir(parents=True, exist_ok=True)

            dest_csv = subjects_dir / Path(self.csv_file_path).name
            self.progress_signal.emit(f"Copiando arquivo CSV para {dest_csv}...")
            try:
                shutil.copy2(self.csv_file_path, dest_csv)
                if logger:
                    logger.info(f"CSV copiado para {dest_csv}")
            except Exception as e:
                tb = traceback.format_exc()
                msg = f"Falha ao copiar CSV para HardThinking: {e}\n{tb}"
                if logger:
                    logger.error(msg)
                self.progress_signal.emit("Erro ao preparar dados para treinamento. Ver log para detalhes.")
                self.finished_signal.emit(False, f"Falha ao copiar CSV. Ver log: {log_file}" if log_file else msg)
                return

            # Preparar request simples para treinar com este sujeito (single_subject)
            request = TrainModelRequest(
                subject_ids=[f"S{self.patient_id:03d}"],
                strategy=TrainingStrategy.SINGLE_SUBJECT,
                model_architecture=ModelArchitecture.CNN_1D,
                hyperparameters={
                    'input_shape': (config.data.window_size, config.data.channels),
                    'num_classes': len(config.data.classes)
                }
            )

            self.progress_signal.emit("Construindo caso de uso de treinamento...")
            # Tentar executar o use-case em-processo; se falhar por ImportError (ex: TF DLL), usar subprocess
            try:
                # Iniciar componentes do CLIInterface para obter a instância do TrainModelUseCase
                cli = CLIInterface()
                trainer = cli.train_model_use_case

                # Se o adapter de ML do HardThinking existir mas declarar-se indisponível
                # (ex: TensorFlow não pôde ser carregado), forçar fallback para subprocess
                try:
                    ml_adapter = getattr(cli, 'ml_adapter', None)
                    if ml_adapter is not None and getattr(ml_adapter, 'available', True) is False:
                        raise ImportError("HardThinking ML adapter presente mas TensorFlow não disponível; solicitar subprocess.")
                except ImportError:
                    # relança para ser capturado pelo handler abaixo e acionar o subprocess
                    raise

                self.progress_signal.emit("Executando treinamento (pode demorar)...")
                if logger:
                    logger.info("Executando use case de treinamento do HardThinking (in-process)")
                try:
                    response = trainer.execute(request)
                except Exception as e:
                    tb = traceback.format_exc()
                    msg = f"Erro executando use case de treino (in-process): {e}\n{tb}"
                    if logger:
                        logger.error(msg)
                    self.progress_signal.emit("Erro durante o treino. Ver log para detalhes.")
                    self.finished_signal.emit(False, f"Erro no treinamento. Ver log: {log_file}" if log_file else msg)
                    return
            except Exception as import_err:
                # Erro ao importar CLIInterface ou dependências (ex: TensorFlow DLL). Tentar subprocess.
                tb = traceback.format_exc()
                if logger:
                    logger.warning(f"Import in-process failed, will attempt subprocess runner: {import_err}\n{tb}")
                self.progress_signal.emit("Ambiente local não suportado; executando treinamento em processo isolado...")

                # Construir script Python para executar o use-case no subprocess (retorna JSON)
                # Construir o código do subprocesso. O subprocess irá salvar o JSON de
                # resultado em um arquivo para que possamos abrir uma nova janela de
                # console e ainda pegar o resultado programaticamente.
                result_file_path = str((logs_dir / f"trainer_subproc_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").resolve())
                runner_code = (
                    "import json\n"
                    "import sys\n"
                    "from pathlib import Path\n"
                    f"sys.path.insert(0, r'{str(hardthinking_root)}')\n"
                    "try:\n"
                    "    from src.interfaces.cli.main_cli import CLIInterface\n"
                    "    from src.application.use_cases.train_model_use_case import TrainModelRequest\n"
                    "    from src.domain.value_objects.training_types import TrainingStrategy\n"
                    "    from src.domain.entities.model import ModelArchitecture\n"
                    "    cli = CLIInterface()\n"
                    +f"    request = TrainModelRequest(subject_ids={request.subject_ids!r}, strategy=TrainingStrategy.{request.strategy.name}, model_architecture=ModelArchitecture.{request.model_architecture.name}, hyperparameters={request.hyperparameters!r})\n"
                    "    resp = cli.train_model_use_case.execute(request)\n"
                    "    out = {'success': bool(getattr(resp, 'success', False)), 'error': getattr(resp, 'error_message', None), 'model_path': getattr(resp.model, 'file_path', getattr(resp.model, 'path', None) if resp and getattr(resp, 'model', None) else None)}\n"
                    f"    with open(r'{result_file_path}', 'w', encoding='utf-8') as rf:\n"
                    "        rf.write(json.dumps(out))\n"
                    "    print(json.dumps(out))\n"
                    "except Exception as e:\n"
                    "    import traceback\n"
                    f"    with open(r'{result_file_path}', 'w', encoding='utf-8') as rf:\n"
                    "        rf.write(json.dumps({'success': False, 'error': str(e), 'trace': traceback.format_exc()}))\n"
                    "    print(json.dumps({'success': False, 'error': str(e), 'trace': traceback.format_exc()}))\n"
                )

                try:
                    import subprocess

                    # Preparar arquivos de stdout/stderr para salvar logs
                    sub_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    sub_out_file = logs_dir / f"trainer_subproc_{sub_ts}.stdout"
                    sub_err_file = logs_dir / f"trainer_subproc_{sub_ts}.stderr"

                    # Abrir arquivos e lançar subprocess em nova console no Windows
                    f_out = open(sub_out_file, 'w', encoding='utf-8')
                    f_err = open(sub_err_file, 'w', encoding='utf-8')

                    env = {**os.environ, 'PYTHONPATH': str(hardthinking_root)}
                    # Use console separado no Windows para que o usuário veja os prints ao vivo
                    if os.name == 'nt':
                        proc = subprocess.Popen([sys.executable, '-u', '-c', runner_code], stdout=f_out, stderr=f_err, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
                    else:
                        proc = subprocess.Popen([sys.executable, '-u', '-c', runner_code], stdout=f_out, stderr=f_err, env=env)

                    # Esperar término
                    returncode = proc.wait()

                    # fechar arquivos para garantir flush
                    try:
                        f_out.close()
                        f_err.close()
                    except Exception:
                        pass

                    if logger:
                        logger.info(f"Subprocess finished with returncode: {returncode}")
                        logger.info(f"Subprocess stdout saved to: {sub_out_file}")
                        logger.info(f"Subprocess stderr saved to: {sub_err_file}")

                    # Ler o arquivo de resultado escrito pelo subprocess
                    result_file = Path(result_file_path)
                    try:
                        if result_file.exists():
                            import json as _json
                            parsed = _json.loads(result_file.read_text(encoding='utf-8').strip())
                        else:
                            parsed = None
                    except Exception:
                        parsed = None

                    if parsed and parsed.get('success'):
                        model_path = parsed.get('model_path')
                        response = type('R', (), {'success': True, 'model': type('M', (), {'file_path': model_path})})()
                    else:
                        err = None
                        if parsed:
                            err = parsed.get('error')
                        if not err:
                            # Ler stderr para mensagem
                            try:
                                err_txt = sub_err_file.read_text(encoding='utf-8')
                                err = err_txt.strip() if err_txt.strip() else f"Subprocess failed with returncode {returncode}"
                            except Exception:
                                err = f"Subprocess failed with returncode {returncode}"
                        response = type('R', (), {'success': False, 'error_message': err})()

                except Exception as e:
                    if logger:
                        logger.error(f"Falha ao executar subprocess runner: {e}\n{traceback.format_exc()}")
                    self.progress_signal.emit("Falha ao executar treinamento em processo isolado. Ver log para detalhes.")
                    self.finished_signal.emit(False, f"Falha no treinamento (subprocess). Ver log: {log_file}" if log_file else str(e))
                    return

            if response and getattr(response, 'success', False):
                model = response.model
                # tentar obter caminho do modelo retornado pelo use case
                model_path = None
                if hasattr(model, 'file_path') and model.file_path:
                    model_path = str(model.file_path)
                elif hasattr(model, 'path') and model.path:
                    model_path = str(model.path)

                # Normalizar caminho: se for relativo, tentar múltiplas bases onde o subprocess possa ter salvo
                try:
                    model_abs = None
                    if model_path:
                        p = Path(model_path)
                        if p.is_absolute():
                            model_abs = p
                        else:
                            # candidatar bases onde 'files/' pode ter sido criado
                            candidate_bases = [
                                hardthinking_root,                # HardThinking/files
                                hardthinking_root.parent,         # workspace_root/files (quando subprocess rodou no repo root)
                                Path.cwd(),                       # current working dir of launcher
                            ]
                            for base in candidate_bases:
                                candidate = (base / model_path).resolve()
                                if candidate.exists():
                                    model_abs = candidate
                                    if logger:
                                        logger.info(f"Modelo encontrado em base {base}: {model_abs}")
                                    break
                            # se não encontrado, fallback para resolver relativo ao HardThinking
                            if model_abs is None:
                                model_abs = (hardthinking_root / model_path).resolve()
                    else:
                        model_abs = None
                except Exception:
                    model_abs = None

                # Copiar modelo para bci/models para facilitar o carregamento pelo BCI
                copied_model_path = None
                try:
                    if model_abs and model_abs.exists():
                        bci_models_dir = Path(__file__).parent.parent / 'models'
                        bci_models_dir.mkdir(parents=True, exist_ok=True)
                        dest_model = bci_models_dir / model_abs.name
                        shutil.copy2(str(model_abs), str(dest_model))
                        copied_model_path = str(dest_model)
                        if logger:
                            logger.info(f"Modelo copiado para {copied_model_path}")
                    else:
                        if logger:
                            logger.warning(f"Caminho absoluto do modelo não existe: {model_abs}")
                except Exception as e:
                    if logger:
                        logger.error(f"Falha ao copiar modelo: {e}\n{traceback.format_exc()}")

                msg = f"Treinamento concluído com sucesso. Modelo salvo em: {copied_model_path if copied_model_path else (model_path if model_path else 'files/*.keras')}"
                self.progress_signal.emit(msg)
                # emitir caminho se disponível (priorizar o copiado dentro de bci/models)
                emit_path = copied_model_path or (str(model_abs) if model_abs else model_path)
                if emit_path:
                    try:
                        self.model_path_signal.emit(str(emit_path))
                    except Exception:
                        pass
                self.finished_signal.emit(True, f"{msg} (log: {log_file})" if log_file else msg)
            else:
                err = response.error_message if response is not None else 'Resposta inválida do use case'
                msg = f"Falha no treinamento: {err}"
                self.progress_signal.emit(msg)
                self.finished_signal.emit(False, msg)

        except Exception as e:
            tb = traceback.format_exc()
            msg = f"Erro inesperado no thread de treinamento: {e}\n{tb}"
            self.progress_signal.emit(msg)
            self.finished_signal.emit(False, msg)
