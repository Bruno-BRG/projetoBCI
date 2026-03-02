"""
Constantes do sistema BrainBridge
"""

# Marcadores de eventos
MARKER_LEFT_HAND = "T1"
MARKER_RIGHT_HAND = "T2"
MARKER_END = "T0"

# Estados do sistema
STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATE_STREAMING = "streaming"
STATE_TRAINING = "training"

# Filtros padrão
FILTER_LOW_PASS = 30.0
FILTER_HIGH_PASS = 8.0
FILTER_NOTCH = 50.0

# Canais EEG padrão
EEG_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4"
]

# Formato de arquivo
CSV_DELIMITER = ","
CSV_ENCODING = "utf-8"

# Timeouts
NETWORK_TIMEOUT = 5.0
DATABASE_TIMEOUT = 10.0