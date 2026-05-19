"""
Pipeline de treinamento de modelos para BrainBridge v2

Fluxo (alinhado ao HardThinking):
 - Lê CSVs no formato OpenBCI (com coluna Annotations: T1/T2/T0)
 - Extrai segmentos entre marcadores: T1..T0 (classe 0), T2..T0 (classe 1)
 - Janela deslizante (window=250, overlap=125) sobre cada segmento
 - Pré-processamento por janela: Butterworth 8–30 Hz e normalização z-score por canal
 - Treina um modelo CNN 1D Keras com callbacks (EarlyStopping/ReduceLROnPlateau)
 - Split estratificado explícito para validação e métricas
 - Salva o modelo em data/models
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import csv
import time
import re

from .models import build_cnn_1d, load_keras_model
from .physionet_eegmmidb_protocol import is_left_right_training_file
from ..config.settings import MODELS_DIR
from ..signal_processing.butter_filter import ButterworthFilter

try:
    # Métricas e split estratificado
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
except Exception:
    # sklearn é opcional em runtime; treinar ainda funciona sem métricas extras
    train_test_split = None  # type: ignore
    accuracy_score = None  # type: ignore
    f1_score = None  # type: ignore


@dataclass
class TrainResult:
    model_path: str
    final_accuracy: Optional[float]
    final_loss: Optional[float]
    history: Dict[str, List[float]]
    training_time: float
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None


@dataclass
class SubjectWindowSummary:
    group_id: str
    csv_files: List[str]
    windows: int
    class_counts: Dict[int, int]


@dataclass
class GeneralizedTrainResult(TrainResult):
    heldout_groups: List[str] = None
    train_groups: List[str] = None
    group_summaries: List[SubjectWindowSummary] = None
    group_metrics: Dict[str, Dict[str, float]] = None


def _load_openbci_csv(csv_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Carrega CSV OpenBCI e retorna (data, markers).

    data: np.ndarray shape (n_samples, 16)
    markers: lista de strings (ex: '', 'T1', 'T2', 'T0')
    """
    rows = []
    markers = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        # pular headers do OpenBCI começando com '%'
        for row in reader:
            if not row:
                continue
            if row[0].startswith('%'):
                continue
            # Header real tem "Sample Index"; vamos detectar e pular a linha de header
            if row[0] == 'Sample Index':
                continue
            rows.append(row)

    # Cada linha: [Sample Index, EXG0..EXG15, Accel0..3, Other..., Analog..., Timestamp..., Annotations]
    # Precisamos extrair EXG0..EXG15 (colunas 1..16) e a última coluna (Annotations)
    data = []
    for r in rows:
        # cuidar de linhas incompletas
        if len(r) < 34:
            # tentar continuar se houver ao menos 17 colunas (índice + 16 canais)
            if len(r) < 17:
                continue
        try:
            channels = [float(x) for x in r[1:17]]
            data.append(channels)
            markers.append(r[-1] if len(r) > 0 else '')
        except Exception:
            continue

    return np.array(data, dtype=np.float32), markers


def _find_marker_indices(markers: List[str]) -> Dict[str, List[int]]:
    idxs = {'T0': [], 'T1': [], 'T2': []}
    for i, m in enumerate(markers):
        if m in idxs:
            idxs[m].append(i)
    return idxs


def _extract_segments_between_markers(data: np.ndarray,
                                      start_indices: List[int],
                                      end_indices: List[int]) -> List[np.ndarray]:
    segments: List[np.ndarray] = []
    for s in start_indices:
        e = next((e for e in end_indices if e > s), None)
        if e is None:
            continue
        seg = data[s:e]
        if len(seg) > 0:
            segments.append(seg)
    return segments


def _create_windows_ht(data: np.ndarray,
                       markers: List[str],
                       window_size: int = 250,
                       step: int = 125,
                       fs: float = 125.0,
                       apply_filter: bool = True,
                       band: Tuple[float, float] = (8.0, 30.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Extrai janelas rotuladas no estilo HardThinking.

    - Constrói segmentos T1..T0 (label 0) e T2..T0 (label 1)
    - Desliza janela com 'window_size' e 'step' dentro de cada segmento
    - Aplica filtro 8–30 Hz e normalização z-score por canal por janela
    """
    idxs = _find_marker_indices(markers)
    seg_T1 = _extract_segments_between_markers(data, idxs.get('T1', []), idxs.get('T0', []))
    seg_T2 = _extract_segments_between_markers(data, idxs.get('T2', []), idxs.get('T0', []))

    filt = None
    if apply_filter:
        low, high = band
        filt = ButterworthFilter(lowcut=low, highcut=high, fs=fs, order=6)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    def process_segment(seg: np.ndarray, label: int):
        nonlocal X_list, y_list
        n = len(seg)
        if n < window_size:
            return
        for start in range(0, n - window_size + 1, step):
            w = seg[start:start + window_size]
            # filtro por janela para evitar vazamento entre segmentos
            if filt is not None:
                # espera shape (channels, samples) no nosso ButterworthFilter; ajustar
                w_f = filt.apply_filter(w.T).T  # (win, ch)
            else:
                w_f = w
            # normalização z-score por canal
            mu = np.mean(w_f, axis=0, keepdims=True)
            sigma = np.std(w_f, axis=0, keepdims=True) + 1e-6
            w_n = (w_f - mu) / sigma
            X_list.append(w_n.astype(np.float32))
            y_list.append(label)

    for seg in seg_T1:
        process_segment(seg, 0)
    for seg in seg_T2:
        process_segment(seg, 1)

    if not X_list:
        return (
            np.zeros((0, window_size, data.shape[1] if data.ndim == 2 else 16), dtype=np.float32),
            np.zeros((0,), dtype=np.int32)
        )
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def _infer_group_id_from_path(csv_path: str | Path) -> str:
    path = Path(csv_path)
    candidates = [path.parent.name, path.stem]
    for candidate in candidates:
        match = re.search(r"([PS]\d{2,4})", candidate, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    if path.parent.name:
        return path.parent.name
    return path.stem


def _build_training_callbacks():
    callbacks = []
    try:
        import importlib  # lazy import to avoid hard dependency at import time
        tf_cb = importlib.import_module('tensorflow.keras.callbacks')
        EarlyStopping = getattr(tf_cb, 'EarlyStopping')
        ReduceLROnPlateau = getattr(tf_cb, 'ReduceLROnPlateau')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-4)
        ]
    except Exception:
        callbacks = []
    return callbacks


def _collect_windowed_dataset(
    csv_files: List[str],
    *,
    window_size: int = 250,
    step: int = 125,
    fs: float = 125.0,
    apply_filter: bool = True,
    band: Tuple[float, float] = (8.0, 30.0),
    group_resolver: Optional[Callable[[str], str]] = None,
    left_right_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[SubjectWindowSummary]]:
    all_X = []
    all_y = []
    all_groups = []
    summaries_by_group: Dict[str, SubjectWindowSummary] = {}
    resolver = group_resolver or _infer_group_id_from_path

    for path in csv_files:
        if left_right_only and not is_left_right_training_file(path):
            continue
        group_id = str(resolver(path))
        data, markers = _load_openbci_csv(Path(path))
        X, y = _create_windows_ht(
            data,
            markers,
            window_size=window_size,
            step=step,
            fs=fs,
            apply_filter=apply_filter,
            band=band,
        )
        if len(X) == 0:
            continue

        all_X.append(X)
        all_y.append(y)
        all_groups.append(np.array([group_id] * len(y), dtype=object))

        summary = summaries_by_group.get(group_id)
        if summary is None:
            summary = SubjectWindowSummary(
                group_id=group_id,
                csv_files=[],
                windows=0,
                class_counts={0: 0, 1: 0},
            )
            summaries_by_group[group_id] = summary
        summary.csv_files.append(str(path))
        summary.windows += int(len(y))
        labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            summary.class_counts[int(label)] = summary.class_counts.get(int(label), 0) + int(count)

    if not all_X:
        channels = 16
        return (
            np.zeros((0, window_size, channels), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=object),
            [],
        )

    return (
        np.concatenate(all_X, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_groups, axis=0),
        list(summaries_by_group.values()),
    )


def load_generalized_windowed_dataset(
    csv_files: List[str],
    *,
    window_size: int = 250,
    step: int = 125,
    group_resolver: Optional[Callable[[str], str]] = None,
    left_right_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[SubjectWindowSummary]]:
    """Carrega janelas rotuladas preservando o grupo/paciente de cada CSV.

    API de desenvolvimento para validar datasets generalizados sem treinar.
    """
    return _collect_windowed_dataset(
        csv_files,
        window_size=window_size,
        step=step,
        group_resolver=group_resolver,
        left_right_only=left_right_only,
    )


def train_from_csvs(csv_files: List[str],
                    window_size: int = 250,
                    step: int = 125,
                    epochs: int = 30,
                    batch_size: int = 32,
                    model_name: Optional[str] = None,
                    base_model_path: Optional[str] = None,
                    model_builder: Optional[Callable[[Tuple[int, int], int], Any]] = None,
                    model_loader: Optional[Callable[[str], Any]] = None) -> TrainResult:
    """Treina a partir de uma lista de CSVs OpenBCI.

    Salva o modelo em data/models e retorna métricas básicas.
    """
    import tensorflow as tf  # lança ImportError cedo se faltar

    # Carregar e empilhar dados (segmentação T1/T2 -> T0)
    all_X = []
    all_y = []
    for path in csv_files:
        data, markers = _load_openbci_csv(Path(path))
        X, y = _create_windows_ht(
            data, markers,
            window_size=window_size, step=step,
            fs=125.0, apply_filter=True, band=(8.0, 30.0)
        )
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        raise ValueError("Nenhuma janela válida encontrada nos CSVs fornecidos.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Construir modelo novo ou continuar fine-tuning a partir de um checkpoint.
    expected_input_shape = (window_size, X.shape[-1])
    if base_model_path:
        base_path = Path(base_model_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Modelo base nao encontrado: {base_model_path}")
        loader = model_loader or load_keras_model
        model = loader(str(base_path))
        model_input_shape = getattr(model, "input_shape", None)
        if model_input_shape is not None and len(model_input_shape) >= 3:
            loaded_shape = tuple(model_input_shape[-2:])
            if loaded_shape != expected_input_shape:
                raise ValueError(
                    "Modelo base incompativel com os dados de treino: "
                    f"esperado {expected_input_shape}, recebido {loaded_shape}."
                )
    else:
        builder = model_builder or (lambda input_shape, num_classes: build_cnn_1d(input_shape, num_classes))
        model = builder(expected_input_shape, 2)

    # Split explícito para validação estratificada (se sklearn disponível)
    if train_test_split is not None and len(np.unique(y)) > 1 and len(y) > 10:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        # fallback para usar todo X como treino e validação via validation_split
        X_train, y_train = X, y
        X_val, y_val = None, None

    # Callbacks estilo HardThinking
    callbacks = _build_training_callbacks()

    # Treinar
    t0 = time.time()
    if X_val is None:
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        val_loss = float(history.history.get('val_loss', [np.nan])[-1]) if 'val_loss' in history.history else None
        val_acc = float(history.history.get('val_accuracy', [np.nan])[-1]) if 'val_accuracy' in history.history else None
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        # Avaliar no conjunto de validação
        try:
            eval_loss, eval_acc = model.evaluate(X_val, y_val, verbose=0)
            val_loss = float(eval_loss)
            val_acc = float(eval_acc)
        except Exception:
            val_loss, val_acc = None, None
    t1 = time.time()

    # Salvar
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = model_name or f"cnn1d_{int(t0)}"
    out_path = MODELS_DIR / f"{model_name}.keras"
    model.save(out_path)

    # Resultados
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    final_acc = float(history.history.get('accuracy', [None])[-1]) if 'accuracy' in history.history else None
    final_loss = float(history.history.get('loss', [None])[-1]) if 'loss' in history.history else None

    return TrainResult(
        model_path=str(out_path),
        final_accuracy=final_acc,
        final_loss=final_loss,
        history=hist,
        training_time=(t1 - t0),
        val_accuracy=val_acc,
        val_loss=val_loss
    )


def train_generalized_from_csvs(
    csv_files: List[str],
    window_size: int = 250,
    step: int = 125,
    epochs: int = 30,
    batch_size: int = 32,
    model_name: Optional[str] = None,
    group_resolver: Optional[Callable[[str], str]] = None,
    model_builder: Optional[Callable[[Tuple[int, int], int], Any]] = None,
    validation_size: float = 0.2,
    left_right_only: bool = True,
) -> GeneralizedTrainResult:
    """Treina modelo dev/generalizado com validação por grupo/paciente.

    Diferente de `train_from_csvs`, este fluxo nunca mistura janelas do mesmo
    grupo entre treino e validação. O grupo padrão é inferido do caminho do CSV
    (ex: pasta/arquivo contendo P001, P002, S001, etc.).
    """
    import tensorflow as tf  # noqa: F401  # lança ImportError cedo se faltar

    if not csv_files:
        raise ValueError("Lista de CSVs nao pode ser vazia.")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size deve ficar entre 0 e 1.")

    X, y, groups, summaries = _collect_windowed_dataset(
        csv_files,
        window_size=window_size,
        step=step,
        group_resolver=group_resolver,
        left_right_only=left_right_only,
    )
    if len(X) == 0:
        raise ValueError("Nenhuma janela válida encontrada nos CSVs fornecidos.")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            "Treino generalizado exige ao menos dois grupos/pacientes distintos."
        )

    if len(np.unique(y)) < 2:
        raise ValueError("Treino generalizado exige janelas das classes T1 e T2.")

    try:
        from sklearn.model_selection import GroupShuffleSplit
    except Exception as e:
        raise ImportError("scikit-learn nao esta instalado. 'pip install scikit-learn'") from e

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=42,
    )
    train_idx, val_idx = next(splitter.split(X, y, groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    train_groups = sorted(str(group) for group in np.unique(groups[train_idx]))
    heldout_groups = sorted(str(group) for group in np.unique(groups[val_idx]))

    builder = model_builder or (lambda input_shape, num_classes: build_cnn_1d(input_shape, num_classes))
    model = builder((window_size, X.shape[-1]), 2)
    callbacks = _build_training_callbacks()

    t0 = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    try:
        eval_loss, eval_acc = model.evaluate(X_val, y_val, verbose=0)
        val_loss = float(eval_loss)
        val_acc = float(eval_acc)
    except Exception:
        val_loss, val_acc = None, None

    group_metrics: Dict[str, Dict[str, float]] = {}
    try:
        probabilities = model.predict(X_val, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)
        for group_id in heldout_groups:
            mask = groups[val_idx] == group_id
            if np.any(mask):
                group_metrics[group_id] = {
                    "accuracy": float(np.mean(y_pred[mask] == y_val[mask])),
                    "windows": float(np.sum(mask)),
                }
    except Exception:
        group_metrics = {}

    t1 = time.time()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = model_name or f"generalized_cnn1d_{int(t0)}"
    out_path = MODELS_DIR / f"{model_name}.keras"
    model.save(out_path)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    final_acc = float(history.history.get('accuracy', [None])[-1]) if 'accuracy' in history.history else None
    final_loss = float(history.history.get('loss', [None])[-1]) if 'loss' in history.history else None

    return GeneralizedTrainResult(
        model_path=str(out_path),
        final_accuracy=final_acc,
        final_loss=final_loss,
        history=hist,
        training_time=(t1 - t0),
        val_accuracy=val_acc,
        val_loss=val_loss,
        heldout_groups=heldout_groups,
        train_groups=train_groups,
        group_summaries=summaries,
        group_metrics=group_metrics,
    )


class ModelTrainer:
    """API simples de treinamento no estilo HardThinking, encapsulada em uma classe.

    Exemplos de uso:
        trainer = ModelTrainer()
        result = trainer.train_from_csvs(["S001/session1.csv"])  
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_from_csvs(self,
                        csv_files: List[str],
                        window_size: int = 250,
                        step: int = 125,
                        epochs: int = 30,
                        batch_size: int = 32,
                        model_name: Optional[str] = None,
                        base_model_path: Optional[str] = None) -> TrainResult:
        """Encapsula a função de treinamento de lista de CSVs.

        Args:
            csv_files: lista de caminhos para CSVs no formato OpenBCI
            window_size: tamanho da janela (amostras)
            step: passo entre janelas (amostras)
            epochs: épocas de treino
            batch_size: tamanho do batch
            model_name: nome opcional para o arquivo do modelo
            base_model_path: checkpoint opcional para continuar fine-tuning
        """
        # Delegamos para a função já implementada acima para reuso
        return train_from_csvs(
            csv_files=csv_files,
            window_size=window_size,
            step=step,
            epochs=epochs,
            batch_size=batch_size,
            model_name=model_name,
            base_model_path=base_model_path,
        )

    def train_from_directory(self,
                             directory: str,
                             pattern: str = "*.csv",
                             window_size: int = 250,
                             step: int = 125,
                             epochs: int = 30,
                             batch_size: int = 32,
                             model_name: Optional[str] = None,
                             base_model_path: Optional[str] = None) -> TrainResult:
        """Treina buscando todos os CSVs em um diretório (não recursivo).

        Útil para treinar rapidamente a partir de uma pasta de sujeito.
        """
        dir_path = Path(directory)
        csvs = [str(p) for p in dir_path.glob(pattern) if p.is_file()]
        if not csvs:
            raise ValueError(f"Nenhum CSV encontrado em {directory} com padrão {pattern}")
        return self.train_from_csvs(
            csv_files=csvs,
            window_size=window_size,
            step=step,
            epochs=epochs,
            batch_size=batch_size,
            model_name=model_name,
            base_model_path=base_model_path,
        )

    def train_generalized_from_csvs(self,
                                    csv_files: List[str],
                                    window_size: int = 250,
                                    step: int = 125,
                                    epochs: int = 30,
                                    batch_size: int = 32,
                                    model_name: Optional[str] = None,
                                    group_resolver: Optional[Callable[[str], str]] = None,
                                    validation_size: float = 0.2,
                                    left_right_only: bool = True) -> GeneralizedTrainResult:
        """Treino dev/generalizado com validação por grupo/paciente."""
        return train_generalized_from_csvs(
            csv_files=csv_files,
            window_size=window_size,
            step=step,
            epochs=epochs,
            batch_size=batch_size,
            model_name=model_name,
            group_resolver=group_resolver,
            validation_size=validation_size,
            left_right_only=left_right_only,
        )

    def train_generalized_from_directory(self,
                                         directory: str,
                                         pattern: str = "*.csv",
                                         window_size: int = 250,
                                         step: int = 125,
                                         epochs: int = 30,
                                         batch_size: int = 32,
                                         model_name: Optional[str] = None,
                                         validation_size: float = 0.2,
                                         left_right_only: bool = True) -> GeneralizedTrainResult:
        """Treino dev/generalizado buscando CSVs recursivamente por diretório."""
        dir_path = Path(directory)
        csvs = [str(p) for p in dir_path.rglob(pattern) if p.is_file()]
        if not csvs:
            raise ValueError(f"Nenhum CSV encontrado em {directory} com padrão {pattern}")
        return self.train_generalized_from_csvs(
            csv_files=csvs,
            window_size=window_size,
            step=step,
            epochs=epochs,
            batch_size=batch_size,
            model_name=model_name,
            validation_size=validation_size,
            left_right_only=left_right_only,
        )
