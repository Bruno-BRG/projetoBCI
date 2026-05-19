"""
Protocol map for PhysioNet EEGMMIDB runs.

The same event codes (T1/T2) mean different things depending on the run.
For BrainBridge left-vs-right training, only unilateral fist runs are valid.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re


@dataclass(frozen=True)
class PhysioNetRunProtocol:
    run: int
    family: str
    task: str
    t1_label: Optional[str]
    t2_label: Optional[str]
    usable_for_left_right: bool
    notes: str


RUN_PROTOCOLS = {
    1: PhysioNetRunProtocol(1, "baseline", "eyes_open", None, None, False, "Baseline; no motor labels."),
    2: PhysioNetRunProtocol(2, "baseline", "eyes_closed", None, None, False, "Baseline; no motor labels."),
    3: PhysioNetRunProtocol(3, "executed", "left_right_fist", "left", "right", True, "Executed left/right fist."),
    4: PhysioNetRunProtocol(4, "imagined", "left_right_fist", "left", "right", True, "Imagined left/right fist."),
    5: PhysioNetRunProtocol(5, "executed", "both_fists_vs_feet", "both_fists", "both_feet", False, "Not left/right."),
    6: PhysioNetRunProtocol(6, "imagined", "both_fists_vs_feet", "both_fists", "both_feet", False, "Not left/right."),
    7: PhysioNetRunProtocol(7, "executed", "left_right_fist", "left", "right", True, "Executed left/right fist."),
    8: PhysioNetRunProtocol(8, "imagined", "left_right_fist", "left", "right", True, "Imagined left/right fist."),
    9: PhysioNetRunProtocol(9, "executed", "both_fists_vs_feet", "both_fists", "both_feet", False, "Not left/right."),
    10: PhysioNetRunProtocol(10, "imagined", "both_fists_vs_feet", "both_fists", "both_feet", False, "Not left/right."),
    11: PhysioNetRunProtocol(11, "executed", "left_right_fist", "left", "right", True, "Executed left/right fist."),
    12: PhysioNetRunProtocol(12, "imagined", "left_right_fist", "left", "right", True, "Imagined left/right fist."),
    13: PhysioNetRunProtocol(13, "executed", "both_fists_vs_feet", "both_fists", "both_feet", False, "Not left/right."),
    14: PhysioNetRunProtocol(14, "imagined", "both_fists_vs_feet", "both_fists", "both_feet", False, "Not left/right."),
}

LEFT_RIGHT_RUNS = frozenset(
    run for run, protocol in RUN_PROTOCOLS.items() if protocol.usable_for_left_right
)
UNSUPPORTED_LEFT_RIGHT_RUNS = frozenset(
    run for run, protocol in RUN_PROTOCOLS.items() if not protocol.usable_for_left_right
)


def infer_physionet_run(path: str | Path) -> Optional[int]:
    """Return the EEGMMIDB run number from names like S001R04_csv_openbci.csv."""
    match = re.search(r"S\d{3}R(\d{2})", Path(path).stem, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def get_run_protocol(path_or_run: str | Path | int) -> Optional[PhysioNetRunProtocol]:
    run = path_or_run if isinstance(path_or_run, int) else infer_physionet_run(path_or_run)
    if run is None:
        return None
    return RUN_PROTOCOLS.get(int(run))


def is_left_right_training_file(path: str | Path) -> bool:
    """True for EEGMMIDB unilateral fist runs: R03/R04/R07/R08/R11/R12."""
    protocol = get_run_protocol(path)
    if protocol is None:
        return True
    return protocol.usable_for_left_right


def describe_run(path_or_run: str | Path | int) -> str:
    protocol = get_run_protocol(path_or_run)
    if protocol is None:
        return "unknown/non-PhysioNet run"
    t1 = protocol.t1_label or "none"
    t2 = protocol.t2_label or "none"
    usable = "left/right usable" if protocol.usable_for_left_right else "excluded from left/right"
    return f"R{protocol.run:02d}: {protocol.family} {protocol.task}; T1={t1}; T2={t2}; {usable}"
