# BrainBridge – AI coding assistant instructions

These rules make AI agents immediately productive in this repo. Keep it short, specific, and aligned with current code (bci legacy + brainbridge_v2 refactor + HardThinking legacy).

## Big picture
- Two legacy generations being integrated into brainbridge_v2:
  - `bci/` (legacy PyQt5 GUI): patient DB (SQLite), UDP EEG streaming, OpenBCI CSV logging, real-time plotting, markers (T1/T2, auto T0), baseline timer.
  - `HardThinking/` (legacy training CLI): hexagonal architecture; reads OpenBCI CSV format and outputs `.keras` models.
  - `brainbridge_v2/` (active refactor): clean modular layout per `NOVA_ESTRUTURA.md`; integrates GUI/ML/DB/comm from both legacy codebases with single `main.py` entry point. Much already implemented.
- Primary data flow: UDP -> filter (Butterworth 0.5–50Hz) -> OpenBCI-compatible CSV -> training (brainbridge_v2 integrating HardThinking) -> models in `models/`.

## Entry points and workflows
- GUI (legacy): run `python -m bci` or `python bci/main.py`. PyQt5 window class: `bci/ui/BCI_main_window.py` which wires tabs:
  - Patients: `DatabaseManager` (SQLite at `bci/data/database/bci_patients.db`).
  - Streaming/Recording: uses `UDPReceiver`/`RealTimeUDPConverter` + `OpenBCICSVLogger`.
- Training (legacy): HardThinking CLI in `HardThinking/main.py`; consumes CSV with exact OpenBCI headers.
- v2 refactor (active): `brainbridge_v2/main.py` supports flags like `--cli`, `--simulate`, `--train`, `--check-env` (see `brainbridge_v2/README.md`). Integrates functionality from both legacy systems.

## Key conventions and patterns
- Paths and folders: use helpers in `bci/configs/config.py` (PROJECT_ROOT, ensure_folders_exist, get_recording_path/get_database_path). Don't hardcode paths.
- CSV format: must match OpenBCI GUI exactly. Header comment lines plus columns: Sample Index, EXG Channel 0..15, Accel 0..2, Other/Analog placeholders, Timestamp fields, Annotations. See `bci/network/openbci_csv_logger.py` and `realtime_udp_converter.py` for canonical structure.
- Markers: UI triggers T1/T2; system auto-inserts T0 after ~250 samples at 125 Hz. Baseline blocks markers for 5 minutes and shows countdown (see logger state fields).
- UDP: default host `localhost`, port `12345`. Class `UDPReceiver_BCI` is exported as `UDPReceiver` for legacy imports. Handle string/JSON payloads with flexible decoding.
- Filtering: `ButterworthFilter` applied before saving when converting UDP buffers; default fs=125 Hz, band 0.5–50 Hz, order 6. Reset filter state per session.
- TensorFlow: main GUI preloads TF to avoid Windows DLL clashes. Code must degrade gracefully if TF not installed.
- Database: SQLite via `DatabaseManager` with tables `patients` and `recordings` and convenience methods `add_patient`, `add_recording`, `update_recording_end_time`, etc.

## Cross-component boundaries
- Keep UI responsive: long work in threads (see `RealTimeUDPConverter`, streaming thread widgets). Don't block the Qt event loop.
- Maintain compatibility: new implementations in `brainbridge_v2/` should preserve public behaviors and file formats used by both legacy systems.
- Do not import infrastructure into domain when touching HardThinking patterns; dependencies flow inward (ports/adapters pattern).

## Typical tasks and how to do them here
- Add a new EEG source: implement a receiver that feeds the same dict/array shapes accepted by `RealTimeUDPConverter._convert_to_openbci_format`, keep 16 channels, 125 Hz, and reuse the filter.
- Extend markers: modify `OpenBCICSVLogger.log_sample` and UI buttons; ensure auto-T0 logic still fires and headers unchanged.
- Change storage locations: update `bci/configs/config.py` helpers only; other code should call these functions.
- Integrate a trained model: load `.keras` from `models/` in a background thread; preload TF at import and surface failures via QMessageBox + stderr logging (see `bci/main.py` setup_qt_error_logging).

## Commands developers actually run (Windows PowerShell)
- Install: `pip install -r requirements.txt`; optional: `pip install -r brainbridge_v2/requirements.txt` or `HardThinking/requirements.txt`.
- GUI legacy: `python -m bci` from repo root.
- Training legacy: `cd HardThinking; python main.py`.
- v2 system: `cd brainbridge_v2; python main.py [--cli|--simulate|--train|--check-env]`.
- UDP->CSV converter smoke: run `python -m bci.network.realtime_udp_converter` and send JSON to UDP 12345.

## Examples from code
- UDP receiver export pattern: `bci/network/udp_receiver_BCI.py` defines `UDPReceiver_BCI` and sets `UDPReceiver = UDPReceiver_BCI` for legacy imports.
- OpenBCI header writing: see `_write_openbci_header()` in `realtime_udp_converter.py` for the exact preamble and column names.
- GUI error logging: `setup_qt_error_logging()` in `bci/main.py` patches `QMessageBox` methods to also log to stderr.

## When unsure
- Prefer mirroring the legacy behaviors from both `bci/` and `HardThinking/` when implementing in `brainbridge_v2/`; tests and tools expect compatibility.
- If touching training or data format, verify round-trip: converter -> CSV -> training reads successfully in both legacy and v2 systems.
