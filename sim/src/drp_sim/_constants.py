"""Shared P3020 robot filesystem constants and URDF helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

# sim/ package root: three parents up from sim/src/drp_sim/_constants.py
_SIM_ROOT = Path(__file__).resolve().parent.parent.parent

# Robot description bundled at sim/robot/ (URDF + STL meshes).
_ROBOT_DIR = _SIM_ROOT / "robot"
_URDF_PATH = str(_ROBOT_DIR / "urdf" / "p3020.urdf")
_PACKAGE_ROOT = str(_ROBOT_DIR)
_PACKAGE_PREFIX = "package://dsr_description2/"
_PROCESSED_URDF = str(Path(os.environ.get("TMPDIR", "/tmp")) / "p3020_processed.urdf")

# cuRobo robot configuration (USD assets directory).
_CUROBO_CFG_PATH = str(_SIM_ROOT / "usd" / "assets" / "robots" / "p3020_curobo.yaml")
N_JOINTS = 5


def preprocess_urdf(src: str = _URDF_PATH, dst: str = _PROCESSED_URDF) -> None:
    """Replace ``package://`` URIs with absolute paths in *src* and write to *dst*."""
    with open(src) as f:
        content = f.read()
    pattern = re.compile(re.escape(_PACKAGE_PREFIX) + r"(.+?)(?=[\"'])")
    processed = pattern.sub(lambda m: os.path.join(_PACKAGE_ROOT, m.group(1)), content)
    with open(dst, "w") as f:
        f.write(processed)
