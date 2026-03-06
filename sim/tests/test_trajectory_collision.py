"""Trajectory collision tests: planned paths must not hit conveyor or pallet stands.

Each waypoint's robot collision spheres (world frame) are checked against the
scene obstacle AABBs.  If any sphere penetrates an obstacle the test fails and
reports the offending waypoints with their joint states.

Requires cuRobo + CUDA.  Tests are skipped when unavailable.
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

CUROBO_AVAILABLE = importlib.util.find_spec("curobo") is not None
_CUDA_OK = False
if CUROBO_AVAILABLE:
    try:
        import torch

        _CUDA_OK = torch.cuda.is_available()
    except ImportError:
        pass

_ROBOT_DIR = Path(__file__).resolve().parent.parent / "robot"
_URDF_SRC = str(_ROBOT_DIR / "urdf" / "p3020.urdf")
_URDF_DST = "/tmp/p3020_collision_test.urdf"
_CFG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "usd", "assets", "robots", "p3020_curobo.yaml")
)

_SKIP = not (CUROBO_AVAILABLE and _CUDA_OK and os.path.exists(_URDF_SRC))
_SKIP_REASON = "cuRobo, CUDA, or URDF source not available"

# ---------------------------------------------------------------------------
# Obstacle AABBs matching palletizing_demo.py scene layout
# Format: (x_min, x_max, y_min, y_max, z_min, z_max)
# ---------------------------------------------------------------------------

_CONV_AABB = (0.05, 3.05, -1.075, -0.525, 0.0, 1.25)  # conveyor full body

_SHELF_W = 2 * 0.35 + 0.10  # PALLET_COLS * BOX_SIZE[0] + 0.10
_SHELF_D = 1 * 0.25 + 0.10  # PALLET_ROWS * BOX_SIZE[1] + 0.10
_P1 = (-0.45, 0.75, 1.00)
_P2 = (0.10, 0.75, 1.00)
_p1x_min = _P1[0] - 0.05
_p1x_max = _P1[0] + _SHELF_W + 0.05
_p2x_min = _P2[0] - 0.05
_p2x_max = _P2[0] + _SHELF_W + 0.05
_P_Y_HALF = _SHELF_D / 2 + 0.05
_PALLET1_AABB = (_p1x_min, _p1x_max, _P1[1] - _P_Y_HALF, _P1[1] + _P_Y_HALF, 0.0, _P1[2] + 0.05)
_PALLET2_AABB = (_p2x_min, _p2x_max, _P2[1] - _P_Y_HALF, _P2[1] + _P_Y_HALF, 0.0, _P2[2] + 0.05)

# Pick approach pose (same as demo)
_PICK_POS = [0.35, -0.80, 1.58]
_PICK_QUAT = [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sphere_aabb_overlap(cx: float, cy: float, cz: float, r: float, aabb: tuple) -> float:
    """Penetration depth of sphere into AABB.  >0 means collision."""
    x1, x2, y1, y2, z1, z2 = aabb
    dx = max(0.0, max(x1 - cx, cx - x2))
    dy = max(0.0, max(y1 - cy, cy - y2))
    dz = max(0.0, max(z1 - cz, cz - z2))
    return r - (dx * dx + dy * dy + dz * dz) ** 0.5


def _preprocess_urdf() -> str:
    pkg_root = os.path.dirname(_URDF_SRC)
    with open(_URDF_SRC) as f:
        content = f.read()
    pattern = re.compile(re.escape("package://dsr_description2/") + r"(.+?)(?=[\"'])")
    processed = pattern.sub(lambda m: os.path.join(pkg_root, m.group(1)), content)
    with open(_URDF_DST, "w") as f:
        f.write(processed)
    return _URDF_DST


def _collect_violations(
    spheres,  # torch.Tensor [N, n_spheres, 4]  (x, y, z, r — world frame)
    traj_pos,  # torch.Tensor [N, dof]
    aabb: tuple,
    label: str,
) -> list[str]:
    """Return human-readable lines for each waypoint that collides with aabb."""
    lines = []
    for wp in range(spheres.shape[0]):
        joints = [f"{v:.3f}" for v in traj_pos[wp].tolist()]
        for si in range(spheres.shape[1]):
            cx, cy, cz, r = spheres[wp, si].tolist()
            overlap = _sphere_aabb_overlap(cx, cy, cz, r, aabb)
            if overlap > 0:
                lines.append(
                    f"  wp[{wp:3d}] sphere[{si}] overlaps {label} "
                    f"by {overlap * 100:.1f} cm  joints={joints}"
                )
                break  # one report per waypoint is enough
    return lines


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def traj_fixture():
    """Build MotionGen with scene obstacles, plan pick approach, return data."""
    if _SKIP:
        pytest.skip(_SKIP_REASON)

    import torch
    import yaml
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.math import Pose
    from curobo.types.robot import JointState, RobotConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    _preprocess_urdf()

    with open(_CFG) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = _URDF_DST
    cfg_dict["robot_cfg"]["kinematics"]["asset_root_path"] = os.path.dirname(_URDF_SRC)

    robot_cfg = RobotConfig.from_dict(cfg_dict)
    kin = CudaRobotModel(robot_cfg.kinematics)

    conv_h = 1.10 + 0.15  # CONVEYOR_HEIGHT + margin
    conv_cx = (3.0 + 0.10) / 2.0
    world_cfg = WorldConfig(
        cuboid=[
            Cuboid(
                name="conveyor",
                pose=[conv_cx, -0.80, conv_h / 2, 1.0, 0.0, 0.0, 0.0],
                dims=[3.0 - 0.10 + 0.10, 0.55 + 0.15, conv_h],
            ),
        ]
    )
    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg, world_model=world_cfg, interpolation_dt=1.0 / 60.0
    )
    mg = MotionGen(mg_cfg)
    mg.warmup()

    start = JointState.from_position(torch.zeros(1, 5, dtype=torch.float32).cuda())
    goal = Pose(
        position=torch.tensor([_PICK_POS], dtype=torch.float32),
        quaternion=torch.tensor([_PICK_QUAT], dtype=torch.float32),
    )
    result = mg.plan_single(start, goal, MotionGenPlanConfig(max_attempts=15))
    traj_pos = result.get_interpolated_plan().position  # [N, 5]
    return traj_pos, kin, result.success.item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestPickApproachNoCollision:
    def test_plan_succeeds(self, traj_fixture):
        """cuRobo must find a valid pick approach with scene obstacles active."""
        _, _, success = traj_fixture
        assert success, "MotionGen failed to plan pick approach — check joint limits / IK config"

    def test_clears_conveyor(self, traj_fixture):
        """All trajectory sphere positions must clear the conveyor AABB."""
        traj_pos, kin, success = traj_fixture
        if not success:
            pytest.skip("Plan failed; cannot check collision")

        state = kin.get_state(traj_pos)
        spheres = state.link_spheres_tensor  # [N, n_spheres, 4] — world-frame x,y,z,r
        assert spheres is not None, "link_spheres_tensor not computed — check robot config"

        violations = _collect_violations(spheres, traj_pos, _CONV_AABB, "conveyor")
        if violations:
            msg = f"Conveyor collision at {len(violations)} waypoints:\n" + "\n".join(
                violations[:5]
            )
            pytest.fail(msg)

    def test_clears_pallet_stands(self, traj_fixture):
        """All trajectory sphere positions must clear both pallet stand regions."""
        traj_pos, kin, success = traj_fixture
        if not success:
            pytest.skip("Plan failed; cannot check collision")

        state = kin.get_state(traj_pos)
        spheres = state.link_spheres_tensor
        assert spheres is not None, "link_spheres_tensor not computed — check robot config"

        violations: list[str] = []
        for label, aabb in [("Pallet1", _PALLET1_AABB), ("Pallet2", _PALLET2_AABB)]:
            violations.extend(_collect_violations(spheres, traj_pos, aabb, label))

        if violations:
            msg = f"Pallet collision at {len(violations)} waypoints:\n" + "\n".join(violations[:5])
            pytest.fail(msg)
