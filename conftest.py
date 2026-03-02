"""
Global pytest bootstrap for workspace package layout.

Ensures imports work for both:
- `brainbridge_v2.*`
- package-qualified paths that may resolve through `brainbridge.*`
"""

import sys
import os
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
REPO_PARENT = REPO_ROOT.parent

for candidate in (REPO_PARENT, REPO_ROOT, REPO_ROOT / "brainbridge_v2"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

# Use a writable temp directory inside the current workspace to avoid sandbox permission issues.
WORKSPACE_ROOT = Path(os.getcwd())
TEST_TMP_DIR = WORKSPACE_ROOT / ".pytest_tmp"
TEST_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TEST_TMP_DIR)
os.environ["TMP"] = str(TEST_TMP_DIR)
os.environ["TEMP"] = str(TEST_TMP_DIR)
tempfile.tempdir = str(TEST_TMP_DIR)

# Prevent matplotlib from creating ad-hoc temp config folders at process shutdown.
MPL_CONFIG_DIR = WORKSPACE_ROOT / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
