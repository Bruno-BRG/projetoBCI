import sys
from pathlib import Path


# Allow importing project modules with package-qualified paths (`brainbridge_v2.*`).
PROJECT_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PACKAGE_ROOT))
