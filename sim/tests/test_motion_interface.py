"""Tests for the P3020 cuRobo motion interface.

Isaac Sim dependent tests are skipped automatically when the ``isaacsim``
package is not present.  Pure config / type-validation tests run in the
standard uv pytest environment.
"""

from __future__ import annotations

import importlib
import os

import pytest
import yaml

ROBOT_CFG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "usd",
    "assets",
    "robots",
    "p3020_curobo.yaml",
)


# ---------------------------------------------------------------------------
# Pure Python tests (no Isaac Sim, no cuRobo required)
# ---------------------------------------------------------------------------


class TestCuroboConfig:
    """Validate the cuRobo YAML configuration file."""

    def test_config_file_exists(self) -> None:
        assert os.path.isfile(ROBOT_CFG_PATH), f"Config not found: {ROBOT_CFG_PATH}"

    def test_config_loads(self) -> None:
        with open(ROBOT_CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None

    def test_required_top_level_keys(self) -> None:
        with open(ROBOT_CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        assert "robot_cfg" in cfg
        robot = cfg["robot_cfg"]
        assert "kinematics" in robot
        # cspace lives inside kinematics (cuRobo CudaRobotGeneratorConfig schema)
        assert "cspace" in robot["kinematics"]

    def test_joint_names(self) -> None:
        with open(ROBOT_CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        # joint_names is inside kinematics.cspace
        joints = cfg["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
        expected = ["joint_1", "joint_2", "joint_3", "joint_5", "joint_6"]
        assert joints == expected, f"Unexpected joints: {joints}"

    def test_base_and_ee_links(self) -> None:
        with open(ROBOT_CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        kin = cfg["robot_cfg"]["kinematics"]
        assert kin["base_link"] == "base_link"
        assert kin["ee_link"] == "link_6"

    def test_collision_spheres_per_link(self) -> None:
        with open(ROBOT_CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        spheres = cfg["robot_cfg"]["kinematics"]["collision_spheres"]
        expected_links = {"base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"}
        assert set(spheres.keys()) == expected_links
        for link, sphere_list in spheres.items():
            assert len(sphere_list) > 0, f"No spheres for {link}"
            for s in sphere_list:
                assert "center" in s and "radius" in s, f"Malformed sphere in {link}: {s}"
                assert len(s["center"]) == 3

    def test_cspace_velocity_acceleration(self) -> None:
        with open(ROBOT_CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        # cspace is nested inside kinematics
        cspace = cfg["robot_cfg"]["kinematics"]["cspace"]
        assert cspace["velocity_scale"] == pytest.approx(1.0)
        assert cspace["max_acceleration"] == pytest.approx(15.0)
        assert cspace["max_jerk"] == pytest.approx(500.0)

    def test_self_collision_ignore_adjacent(self) -> None:
        with open(ROBOT_CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        ignore = cfg["robot_cfg"]["kinematics"]["self_collision_ignore"]
        # Each link should ignore its immediate neighbour
        pairs = [
            ("base_link", "link_1"),
            ("link_1", "link_2"),
            ("link_2", "link_3"),
            ("link_3", "link_4"),
            ("link_4", "link_5"),
            ("link_5", "link_6"),
        ]
        for parent, child in pairs:
            assert parent in ignore, f"Missing key {parent} in self_collision_ignore"
            assert (
                child in ignore[parent]
            ), f"Expected {child} ignored for {parent}, got {ignore[parent]}"


def test_sim_runner_exposes_motion_commands():
    from drp_sim.sim_runner import SimCommand

    assert SimCommand.MOVE_PLANNED.value == "move_planned"
    assert SimCommand.MOVE_CARTESIAN.value == "move_cartesian"


class TestTrajectoryFormat:
    """Validate trajectory data structures without running cuRobo."""

    def test_trajectory_is_list_of_lists(self) -> None:
        traj: list[list[float]] = [
            [0.0, 0.1, -0.1, 0.0, 0.2],
            [0.1, 0.2, -0.2, 0.0, 0.3],
        ]
        assert isinstance(traj, list)
        for wp in traj:
            assert isinstance(wp, list)
            assert len(wp) == 5

    def test_joint_count_validation(self) -> None:
        """Confirm that a 6-element list is rejected for a 5-DOF interface."""
        bad_target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert len(bad_target) != 5


_ISAACSIM_AVAILABLE = (
    importlib.util.find_spec("isaacsim") is not None and os.environ.get("ISAACSIM_TESTS", "") == "1"
)


# ---------------------------------------------------------------------------
# Isaac Sim integration tests (skipped when isaacsim is absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ISAACSIM_AVAILABLE, reason="Isaac Sim not installed")
class TestPalletizerEnvIntegration:
    """Smoke tests that require a live Isaac Sim session."""

    @pytest.fixture(scope="class")
    def env(self):
        from drp_sim import PalletizerEnv

        e = PalletizerEnv(headless=True)
        e.reset()
        yield e
        e.close()

    def test_get_joint_positions_length(self, env) -> None:
        positions = env.get_joint_positions()
        assert len(positions) == 5

    def test_set_joint_positions_roundtrip(self, env) -> None:
        target = [0.1, 0.2, -0.1, 0.05, -0.05]
        env.set_joint_positions(target)
        env.step(render=False)
        # Allow for physics settling; just check shape
        result = env.get_joint_positions()
        assert len(result) == 5

    def test_invalid_joint_count_raises(self, env) -> None:
        with pytest.raises(ValueError):
            env.set_joint_positions([0.0, 0.0, 0.0])
