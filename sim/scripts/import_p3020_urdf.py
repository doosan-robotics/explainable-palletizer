"""Isaac Sim 5.1 standalone script: import p3020 URDF and save as USD.

Run in the Isaac Sim Python environment (NOT in the uv workspace):
    source ~/venvs/isaacsim-5/bin/activate
    python sim/scripts/import_p3020_urdf.py [--headless] [--usd-out PATH]

The script pre-processes the URDF to resolve ``package://dsr_description2/``
references to absolute filesystem paths before passing it to the Isaac Sim
URDF importer.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

# ── Isaac Sim 5.1 bootstrap (must happen before any omni.* imports) ────────
from isaacsim import SimulationApp

# Robot description bundled at sim/robot/ (URDF + STL meshes)
_ROBOT_DIR = Path(__file__).resolve().parent.parent / "robot"
_DEFAULT_URDF = str(_ROBOT_DIR / "urdf" / "p3020.urdf")
_DEFAULT_USD_OUT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "usd",
    "assets",
    "robots",
    "p3020.usd",
)
_PACKAGE_ROOT = str(_ROBOT_DIR)
_PACKAGE_PREFIX = "package://dsr_description2/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import p3020 URDF into Isaac Sim USD")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument(
        "--urdf",
        type=str,
        default=_DEFAULT_URDF,
        help="Path to p3020 URDF",
    )
    parser.add_argument(
        "--usd-out",
        type=str,
        default=_DEFAULT_USD_OUT,
        help="Output USD file path",
    )
    return parser.parse_args()


def preprocess_urdf(urdf_path: str) -> str:
    """Replace package:// URIs with absolute paths; return path to temp file."""
    with open(urdf_path) as f:
        content = f.read()

    def replace_pkg(match: re.Match) -> str:
        rel = match.group(1)
        return os.path.join(_PACKAGE_ROOT, rel)

    pattern = re.compile(re.escape(_PACKAGE_PREFIX) + r"(.+?)(?=[\"'])")
    processed = pattern.sub(replace_pkg, content)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".urdf",
        delete=False,
        prefix="p3020_processed_",
    ) as tmp:
        tmp.write(processed)
        return tmp.name


def main() -> None:
    args = parse_args()

    simulation_app = SimulationApp({"headless": args.headless})

    # ── Post-bootstrap imports ─────────────────────────────────────────────
    import omni.kit.commands
    import omni.usd

    urdf_path = os.path.abspath(args.urdf)
    usd_out = os.path.abspath(args.usd_out)

    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    print(f"[import_p3020] Preprocessing URDF: {urdf_path}")
    processed_urdf = preprocess_urdf(urdf_path)

    try:
        print("[import_p3020] Importing URDF into Isaac Sim...")

        # Isaac Sim 5.1: use object-based ImportConfig (NOT dict)
        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False  # mandatory in Isaac Sim 5.1
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.distance_scale = 1.0
        import_config.default_position_drive_damping = 1e3

        # dest_path="" → import into current stage (prim path comes from robot name)
        _, robot_prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=processed_urdf,
            import_config=import_config,
            dest_path="",
            get_articulation_root=True,
        )
        print(f"[import_p3020] Robot imported at prim path: {robot_prim_path}")
    finally:
        os.unlink(processed_urdf)

    os.makedirs(os.path.dirname(usd_out), exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().Export(usd_out)
    print(f"[import_p3020] USD saved to: {usd_out}")

    simulation_app.close()
    print("[import_p3020] Done.")


if __name__ == "__main__":
    main()
