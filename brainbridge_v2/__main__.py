"""Package entrypoint for BrainBridge v2.

Supports both:
- `python -m brainbridge_v2`
- `python brainbridge_v2/__main__.py`
"""

from pathlib import Path
import sys


# When executed as a script path, ensure repo root is importable.
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from brainbridge_v2.presentation.main import main


if __name__ == "__main__":
    raise SystemExit(main())
