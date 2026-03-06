"""Standalone test: Isaac Sim URDF load + cuRobo MotionGen for P3020.

Run with Isaac Sim Python:
    uv run python sim/scripts/test_isaac_curobo.py [--headless]

Stages:
  1. URDF import → robot into World stage (current stage, no dest_path)
  2. Robot articulation joint get/set
  3. cuRobo MotionGen joint-space planning  (plan_single_js)
  4. cuRobo MotionGen Cartesian planning    (plan_single)
  5. Export robot-only USD for env.py use
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import yaml

# ── Bootstrap (must be first) ──────────────────────────────────────────────
from isaacsim import SimulationApp

_ROBOT_DIR = Path(__file__).resolve().parent.parent / "robot"
URDF_PATH = str(_ROBOT_DIR / "urdf" / "p3020.urdf")
PACKAGE_ROOT = str(_ROBOT_DIR)
PACKAGE_PREFIX = "package://dsr_description2/"
USD_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "usd", "assets", "robots", "p3020.usd")
)
CUROBO_CFG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "usd", "assets", "robots", "p3020_curobo.yaml")
)
N_JOINTS = 5
ACTIVE_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_5", "joint_6"]
PROCESSED_URDF = "/tmp/p3020_processed.urdf"
RESULTS_FILE = "/tmp/test_isaac_curobo_results.txt"

_results: list[str] = []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-curobo", action="store_true", help="Skip cuRobo tests")
    return p.parse_args()


def _log(msg: str) -> None:
    """Write to file log (stdout may be buffered/captured by Isaac Sim)."""
    _results.append(msg)
    with open(RESULTS_FILE, "a") as f:
        f.write(msg + "\n")
        f.flush()


def status(msg: str) -> None:
    line = f"\n{'=' * 60}\n  {msg}\n{'=' * 60}"
    print(line, flush=True)
    _log(line)


def ok(msg: str) -> None:
    line = f"  [PASS] {msg}"
    print(line, flush=True)
    _log(line)


def err(msg: str) -> None:
    line = f"  [FAIL] {msg}"
    print(line, flush=True)
    _log(line)


# ── URDF pre-processing ────────────────────────────────────────────────────


def preprocess_urdf(src: str, dst: str) -> None:
    """Replace package:// URIs with absolute paths and write to dst."""
    with open(src) as f:
        content = f.read()
    pattern = re.compile(re.escape(PACKAGE_PREFIX) + r"(.+?)(?=[\"'])")
    processed = pattern.sub(lambda m: os.path.join(PACKAGE_ROOT, m.group(1)), content)
    with open(dst, "w") as f:
        f.write(processed)


# ── Stage 1: URDF import into current World stage ─────────────────────────


def import_urdf_to_stage() -> str:
    """Import p3020 URDF directly into the current stage. Returns prim path."""
    import omni.kit.commands

    status("Stage 1: Importing p3020 URDF into World stage")

    preprocess_urdf(URDF_PATH, PROCESSED_URDF)
    ok(f"Preprocessed URDF → {PROCESSED_URDF}")

    # Isaac Sim 5.1: object-based ImportConfig (NOT dict)
    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False  # Must be False in Isaac Sim 5.1
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.distance_scale = 1.0
    import_config.default_position_drive_damping = 1e3

    # dest_path="" → import into current stage (World stage)
    result, robot_prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=PROCESSED_URDF,
        import_config=import_config,
        dest_path="",
        get_articulation_root=True,
    )
    ok(f"URDF import result={result}, articulation root='{robot_prim_path}'")
    return robot_prim_path


# ── Stage 2: articulation test ────────────────────────────────────────────


def test_articulation(world, robot_prim_path: str):
    import numpy as np
    from isaacsim.core.api.robots import Robot

    status("Stage 2: Robot articulation + joint control")

    robot = world.scene.add(Robot(prim_path=robot_prim_path, name="p3020"))
    world.reset()
    ok(f"World reset OK, num_dof={robot.num_dof}")

    pos = robot.get_joint_positions()
    ok(f"Initial joint positions ({pos.shape}): {np.round(pos, 4).tolist()}")

    # Teleport to a realistic "ready" pose (arm slightly forward and down)
    # joint_2 ≈ -30°, joint_3 ≈ 60° → arm tilted forward, EE around z=1.6m
    n = robot.num_dof
    target = np.zeros(n)
    target[:N_JOINTS] = [0.0, -0.524, 1.047, 0.0, 0.0]  # [0, -30°, 60°, 0, 0]
    robot.set_joint_positions(target)
    for _ in range(30):
        world.step(render=False)

    pos_after = robot.get_joint_positions()
    ok(f"After set_joint_positions: {np.round(pos_after, 4).tolist()}")
    return robot


# ── cuRobo helpers ────────────────────────────────────────────────────────


def build_motion_gen():
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    status("Building cuRobo MotionGen")

    with open(CUROBO_CFG) as f:
        cfg_dict = yaml.safe_load(f)

    # Override with pre-processed URDF (no package:// URIs)
    cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = PROCESSED_URDF
    cfg_dict["robot_cfg"]["kinematics"]["asset_root_path"] = PACKAGE_ROOT

    robot_cfg = RobotConfig.from_dict(cfg_dict)
    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        interpolation_dt=1.0 / 60.0,
    )
    mg = MotionGen(mg_cfg)
    mg.warmup(enable_graph=True)
    ok("MotionGen warmed up")
    return mg


def _current_js(robot):
    import torch
    from curobo.types.robot import JointState

    pos = robot.get_joint_positions()
    # cuRobo requires (batch, dof) shape → (1, N_JOINTS)
    t = torch.tensor(pos[:N_JOINTS].tolist(), dtype=torch.float32).cuda().unsqueeze(0)
    return JointState.from_position(t)


def test_joint_planning(mg, robot, world):
    import numpy as np
    import torch
    from curobo.types.robot import JointState
    from isaacsim.core.utils.types import ArticulationAction

    status("Stage 3: cuRobo joint-space planning (plan_single_js)")

    start = _current_js(robot)
    # Realistic joint-space goal: arm forward and slightly rotated
    # [joint_1=30°, joint_2=-45°, joint_3=90°, joint_5=0°, joint_6=0°]
    goal_t = torch.tensor([[0.524, -0.785, 1.571, 0.0, 0.0]], dtype=torch.float32).cuda()
    goal = JointState.from_position(goal_t)

    t0 = time.perf_counter()
    result = mg.plan_single_js(start, goal)
    elapsed = time.perf_counter() - t0

    if not result.success.item():
        err(f"Joint planning failed ({elapsed:.3f}s)")
        return False

    traj = result.get_interpolated_plan().position.tolist()
    ok(f"Plan: {len(traj)} waypoints in {elapsed:.3f}s")

    n = robot.num_dof
    for i, wp in enumerate(traj):
        full = np.zeros(n)
        full[:N_JOINTS] = wp[:N_JOINTS]
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=full))
        world.step(render=False)

        # Spot-check: compare commanded vs actual at a few waypoints
        if i in (0, len(traj) // 2, len(traj) - 1):
            actual = robot.get_joint_positions()
            cmd = np.array(wp[:N_JOINTS])
            err_norm = np.linalg.norm(actual[:N_JOINTS] - cmd)
            c = np.round(cmd, 3).tolist()
            a = np.round(actual[:N_JOINTS], 3).tolist()
            ok(f"  wp[{i:3d}] cmd={c}  actual={a}  |err|={err_norm:.4f} rad")

    ok("Joint trajectory executed in Isaac Sim")
    return True


def test_pose_planning(mg, robot, world):
    import numpy as np
    import torch
    from curobo.types.math import Pose
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    from isaacsim.core.utils.types import ArticulationAction

    status("Stage 4: cuRobo Cartesian planning (plan_single)")

    start = _current_js(robot)
    # Realistic Cartesian target: EE in front of robot at palletizer working height
    # x=0.8m forward, y=0.2m side, z=1.4m (above base, arm folded forward)
    goal_pose = Pose(
        position=torch.tensor([[0.8, 0.2, 1.4]], dtype=torch.float32),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )

    t0 = time.perf_counter()
    result = mg.plan_single(start, goal_pose, MotionGenPlanConfig(max_attempts=3))
    elapsed = time.perf_counter() - t0

    if not result.success.item():
        err(f"Pose planning failed ({elapsed:.3f}s) — target may be out of reach")
        return False

    traj = result.get_interpolated_plan().position.tolist()
    ok(f"Plan: {len(traj)} waypoints in {elapsed:.3f}s")

    n = robot.num_dof
    for i, wp in enumerate(traj):
        full = np.zeros(n)
        full[:N_JOINTS] = wp[:N_JOINTS]
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=full))
        world.step(render=False)

        if i in (0, len(traj) // 2, len(traj) - 1):
            actual = robot.get_joint_positions()
            cmd = np.array(wp[:N_JOINTS])
            err_norm = np.linalg.norm(actual[:N_JOINTS] - cmd)
            c = np.round(cmd, 3).tolist()
            a = np.round(actual[:N_JOINTS], 3).tolist()
            ok(f"  wp[{i:3d}] cmd={c}  actual={a}  |err|={err_norm:.4f} rad")

    ok("Cartesian trajectory executed in Isaac Sim")
    return True


# ── Stage 5: export clean robot USD ──────────────────────────────────────


def export_robot_usd(robot_prim_path: str) -> None:
    """Export the robot prim as a standalone USD for env.py use."""
    import omni.usd

    status("Stage 5: Exporting robot USD")

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    if not robot_prim.IsValid():
        err(f"Prim not found at {robot_prim_path}")
        return

    # Export the whole stage; record the articulation prim path for env.py
    os.makedirs(os.path.dirname(USD_PATH), exist_ok=True)
    stage.GetRootLayer().Export(USD_PATH)
    ok(f"Stage exported to {USD_PATH}")
    ok(f"Robot articulation prim path in USD: {robot_prim_path}")
    print(f"\n  NOTE: Update env.py PRIM_PATH to '{robot_prim_path}' if needed")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    app = SimulationApp({"headless": args.headless})

    from isaacsim.core.api import World

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
    world.scene.add_default_ground_plane()

    robot_prim_path = import_urdf_to_stage()
    robot = test_articulation(world, robot_prim_path)

    if not args.no_curobo:
        try:
            mg = build_motion_gen()
            test_joint_planning(mg, robot, world)
            test_pose_planning(mg, robot, world)
        except Exception as exc:
            err(f"cuRobo error: {exc}")
            import traceback

            traceback.print_exc()
            if args.headless:
                raise  # propagate failure in CI / headless runs
    else:
        print("  [SKIP] cuRobo tests (--no-curobo)")

    if not os.path.exists(USD_PATH):
        export_robot_usd(robot_prim_path)

    status("All stages complete")

    if not args.headless:
        print("  GUI mode: simulation running — close the window to exit.\n", flush=True)
        while app.is_running():
            world.step(render=True)

    app.close()


if __name__ == "__main__":
    main()
