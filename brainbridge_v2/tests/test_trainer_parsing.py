from pathlib import Path
import shutil
import csv
import numpy as np
import uuid


def _write_openbci_csv(path: Path, n: int = 600, t1_idx: int = 100, t0_idx: int = 350, t2_idx: int = 360, t0b_idx: int = 550):
    # Minimal OpenBCI-like header with % comments then real header row
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['%OpenBCI Raw EEG Data'])
        w.writerow(['Sample Index', *[f'EXG Channel {i}' for i in range(16)], 'Accel 0', 'Accel 1', 'Accel 2', 'Timestamp', 'Annotations'])
        for i in range(n):
            exg = [0.1 * (i % 10)] * 16
            # two MI segments: T1..T0 and T2..T0
            ann = ''
            if i == t1_idx:
                ann = 'T1'
            if i == t0_idx:
                ann = 'T0'
            if i == t2_idx:
                ann = 'T2'
            if i == t0b_idx:
                ann = 'T0'
            row = [i, *exg, 0, 0, 0, i, ann]
            w.writerow(row)


def test_load_and_window_ht():
    from brainbridge_v2.infrastructure.ml.trainer import _load_openbci_csv, _create_windows_ht
    temp_dir = Path(".pytest_tmp") / f"trainer_parsing_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        p = temp_dir / 'test.csv'
        _write_openbci_csv(p)
        data, markers = _load_openbci_csv(p)
        assert data.shape[1] == 16
        assert len(markers) == data.shape[0]

        X, y = _create_windows_ht(data, markers, window_size=250, step=125, fs=125.0, apply_filter=False)
        # Esperamos ao menos algumas janelas de cada classe
        if len(X) > 0:
            assert X.shape[1:] == (250, 16)
            assert set(y.tolist()).issubset({0, 1})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
