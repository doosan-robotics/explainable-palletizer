"""cuRobo MotionGen interface for the P3020 palletizer environment.

Wraps cuRobo ``MotionGen`` to plan joint-space and Cartesian trajectories and
optionally execute them inside the Isaac Sim world via ``PalletizerEnv``.

cuRobo must be installed in the same Python environment as Isaac Sim.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import yaml

from drp_sim._constants import _CUROBO_CFG_PATH, _PACKAGE_ROOT, _PROCESSED_URDF

if TYPE_CHECKING:
    from drp_sim.env import PalletizerEnv

logger = logging.getLogger(__name__)

# VGC10 gripper geometry (world space, mixed_palletizing_scene.usd)
_VGC_LEN: float = 0.23  # gripper length (m)
_BOX_HALF_H: float = 0.103  # box half-height (m)
_BOX_ATTACH_Z: float = _VGC_LEN + _BOX_HALF_H  # EE-to-box-centre offset (0.333 m)


_DEFAULT_CFG = _CUROBO_CFG_PATH
# cuRobo requires CUDA. CUROBO_DEVICE allows running on a separate GPU to
# avoid CUDA context conflicts with Isaac Sim on datacenter GPUs.
_DEVICE = os.environ.get("CUROBO_DEVICE", "cuda:0")


class MotionInterface:
    """cuRobo MotionGen wrapper for the P3020 robot.

    Parameters
    ----------
    env:
        Initialised ``PalletizerEnv`` instance (``reset()`` already called).
    robot_cfg_path:
        Path to the cuRobo robot YAML configuration.
    """

    def __init__(
        self,
        env: PalletizerEnv,
        robot_cfg_path: str = _DEFAULT_CFG,
        motion_gen: object | None = None,
    ) -> None:
        self._env = env
        self._cfg_path = os.path.abspath(robot_cfg_path)
        self._motion_gen = motion_gen if motion_gen is not None else self._build_motion_gen()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def move_to_joints(
        self,
        target: list[float],
        execute: bool = True,
        speed: float = 1.0,
        step_callback: Callable[[], None] | None = None,
        pre_step_callback: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        """Plan (and optionally execute) motion to a joint-space target.

        Parameters
        ----------
        target:
            Goal joint positions in radians for the 5 active joints.
        execute:
            When ``True``, replay the trajectory in Isaac Sim step-by-step.
        speed:
            Velocity scale as a fraction of the robot's maximum speed (0.0-1.0).

        Returns
        -------
        list[list[float]]
            Trajectory as a list of waypoints, each a 5-element position list.
        """
        if len(target) != 5:
            raise ValueError(f"Expected 5 joint targets, got {len(target)}")
        if not (0.0 < speed <= 1.0):
            raise ValueError(f"speed must be in (0, 1], got {speed}")

        import torch
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        current = self._env.get_joint_positions()
        start_state = JointState.from_position(
            torch.tensor([current], dtype=torch.float32).to(_DEVICE)
        )
        goal_state = JointState.from_position(
            torch.tensor([target], dtype=torch.float32).to(_DEVICE)
        )

        result = self._motion_gen.plan_single_js(
            start_state, goal_state, MotionGenPlanConfig(max_attempts=3)
        )
        traj = self._extract_trajectory(result)

        if execute:
            self._execute_trajectory(traj, step_callback, pre_step_callback)

        return traj

    def move_to_pose(
        self,
        position: list[float],
        quaternion: list[float],
        execute: bool = True,
        speed: float = 1.0,
        max_attempts: int = 3,
        orientation_constraint: bool = False,
        step_callback: Callable[[], None] | None = None,
        pre_step_callback: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        """Plan (and optionally execute) motion to a Cartesian end-effector pose.

        Parameters
        ----------
        position:
            [x, y, z] in metres.
        quaternion:
            [w, x, y, z] unit quaternion.
        execute:
            When ``True``, replay the trajectory in Isaac Sim step-by-step.
        max_attempts:
            Number of planning attempts before giving up.
        orientation_constraint:
            When ``True``, hold the goal orientation fixed throughout the
            entire trajectory (not just at the goal).  Prevents end-effector
            tilting during carry motions.  Uses cuRobo ``PoseCostMetric``
            with rotation-only hold weights ``[0, 0, 0, 1, 1, 1]``.

        Returns
        -------
        list[list[float]]
            Trajectory as a list of waypoints, each a 5-element position list.
        """
        import torch
        from curobo.types.math import Pose
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig, PoseCostMetric

        current = self._env.get_joint_positions()
        start_state = JointState.from_position(
            torch.tensor([current], dtype=torch.float32).to(_DEVICE)
        )

        goal_pose = Pose(
            position=torch.tensor([position], dtype=torch.float32).to(_DEVICE),
            quaternion=torch.tensor([quaternion], dtype=torch.float32).to(_DEVICE),
        )

        # time_dilation_factor > 1.0 slows the trajectory; speed=1.0 → no dilation.
        dilation = 1.0 / speed if speed > 0 else 1.0
        plan_cfg = MotionGenPlanConfig(
            max_attempts=max_attempts,
            time_dilation_factor=dilation if dilation != 1.0 else None,
        )
        if orientation_constraint:
            # hold_partial_pose=True: apply constraint at every trajectory waypoint,
            # not only at the goal.  hold_vec_weight=[rx, ry, rz, px, py, pz]:
            # unit rotation weights (hold EE orientation) + zero position weights (free path).
            # cuRobo convention: indices 0-2 = rotation, indices 3-5 = position.
            plan_cfg.pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            )

        logger.debug(
            "move_to_pose: pos=%s quat=%s orient_constraint=%s",
            position,
            quaternion,
            orientation_constraint,
        )
        result = self._motion_gen.plan_single(start_state, goal_pose, plan_cfg)
        traj = self._extract_trajectory(result)

        if execute:
            self._execute_trajectory(traj, step_callback, pre_step_callback)

        return traj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_motion_gen(self):
        return self._build_motion_gen_static(self._cfg_path)

    @staticmethod
    def _build_motion_gen_static(cfg_path: str = _DEFAULT_CFG):
        """Instantiate and return a configured ``MotionGen`` object.

        Can be called without a MotionInterface instance (for pre-building
        before Isaac Sim loads the scene and corrupts the CUDA context).
        """
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import Cuboid, WorldConfig
        from curobo.types.robot import RobotConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        with open(os.path.abspath(cfg_path)) as f:
            cfg_dict = yaml.safe_load(f)

        # Replace package:// URIs so cuRobo can find meshes
        cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = _PROCESSED_URDF
        cfg_dict["robot_cfg"]["kinematics"]["asset_root_path"] = _PACKAGE_ROOT

        # Seed with a ground-plane cuboid so the PRIMITIVE collision checker
        # never starts empty (cuRobo raises ValueError if zero obstacles).
        ground = Cuboid(
            name="ground",
            pose=[0.0, 0.0, -0.55, 1.0, 0.0, 0.0, 0.0],
            dims=[2.0, 2.0, 0.01],
        )
        world_model = WorldConfig(cuboid=[ground])

        import torch
        from curobo.types.base import TensorDeviceType

        device = torch.device(_DEVICE)
        tensor_args = TensorDeviceType(device=device)

        # cuRobo uses both PyTorch and Isaac Sim's bundled warp for CUDA
        # kernels.  Both maintain separate default-device state.  Temporarily
        # switch both defaults so all allocations land on the target GPU,
        # then restore for Isaac Sim.
        prev_torch_device = torch.cuda.current_device()
        prev_warp_device = None
        if device.type == "cuda":
            torch.cuda.set_device(device)
            try:
                import warp as wp

                logger.info(
                    "warp: ver=%s file=%s devs=%d cur=%s",
                    getattr(wp, "__version__", "?"),
                    getattr(wp, "__file__", "?"),
                    wp.get_cuda_device_count(),
                    wp.get_device(),
                )
                prev_warp_device = wp.get_device()
                wp.set_device(str(device))
                logger.info("warp: after set_device -> %s", wp.get_device())
            except Exception as e:
                logger.warning("warp set_device failed: %s", e)

        try:
            robot_cfg = RobotConfig.from_dict(cfg_dict, tensor_args=tensor_args)
            mg_cfg = MotionGenConfig.load_from_robot_config(
                robot_cfg,
                world_model=world_model,
                tensor_args=tensor_args,
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
                interpolation_dt=1.0 / 60.0,
                use_cuda_graph=False,
            )
            mg = MotionGen(mg_cfg)
            MotionInterface._fix_tensor_args(mg, tensor_args)
            MotionInterface._disable_lbfgs_cuda_kernel(mg)
            mg.warmup(enable_graph=False)
        finally:
            torch.cuda.set_device(prev_torch_device)
            if prev_warp_device is not None:
                with contextlib.suppress(Exception):
                    wp.set_device(str(prev_warp_device))

        return mg

    @staticmethod
    def _fix_tensor_args(mg, tensor_args) -> None:
        """Recursively fix tensor_args and move tensors to target device.

        cuRobo doesn't propagate tensor_args to all sub-components, so
        tensors end up on cuda:0.  This walks the object graph, patches
        tensor_args, and moves existing tensors to the correct device.
        """
        import torch
        from curobo.types.base import TensorDeviceType

        target = tensor_args.device
        seen: set[int] = set()
        moved = 0

        def _walk(obj):
            nonlocal moved
            oid = id(obj)
            if oid in seen:
                return
            seen.add(oid)
            try:
                if (
                    hasattr(obj, "tensor_args")
                    and isinstance(obj.tensor_args, TensorDeviceType)
                    and obj.tensor_args.device != target
                ):
                    object.__setattr__(obj, "tensor_args", tensor_args)
            except (RecursionError, Exception):
                return
            for name, val in list(vars(obj).items()):
                if isinstance(val, torch.Tensor) and val.device != target:
                    object.__setattr__(obj, name, val.to(target))
                    moved += 1
                elif hasattr(val, "__dict__"):
                    _walk(val)

        _walk(mg)
        logger.info(
            "_fix_tensor_args: visited %d objects, moved %d tensors to %s", len(seen), moved, target
        )

    @staticmethod
    def _disable_lbfgs_cuda_kernel(mg):
        """Disable LBFGS CUDA kernel — buggy on sm_90 datacenter GPUs."""
        seen: set[int] = set()
        patched = 0

        def _walk(obj):
            nonlocal patched
            oid = id(obj)
            if oid in seen:
                return
            seen.add(oid)
            try:
                if getattr(obj, "use_cuda_kernel", False):
                    object.__setattr__(obj, "use_cuda_kernel", False)
                    patched += 1
            except (RecursionError, Exception):
                return
            try:
                for val in vars(obj).values():
                    if isinstance(val, list | tuple):
                        for item in val:
                            if hasattr(item, "__dict__"):
                                _walk(item)
                    elif isinstance(val, dict):
                        for item in val.values():
                            if hasattr(item, "__dict__"):
                                _walk(item)
                    elif hasattr(val, "__dict__"):
                        _walk(val)
            except (RecursionError, Exception):
                pass

        _walk(mg)
        logger.info("_disable_lbfgs_cuda_kernel: patched %d objects", patched)

    def update_pallet_obstacles(
        self,
        pallet_y_max: float = 0.0,
        default_box_dims: tuple[float, float, float] = (0.30, 0.25, 0.22),
        placed_boxes: (
            list[tuple[object, list[float], tuple[float, float, float], float]] | None
        ) = None,
    ) -> int:
        """Update cuRobo world model with boxes currently placed on the pallet.

        Only boxes with world Y < *pallet_y_max* are treated as obstacles;
        boxes at higher Y are still on the conveyor and are excluded.
        Call once before each pick-place sequence so newly stacked boxes are
        reflected in collision avoidance.

        Parameters
        ----------
        pallet_y_max:
            Boxes with world y >= this value are on the conveyor and skipped.
        default_box_dims:
            (width, depth, height) fallback bounding box for spawner boxes
            that lack per-box dimension data.
        placed_boxes:
            Manually tracked placed boxes as (rigid, [x,y,z], (dx,dy,dz)).

        Returns
        -------
        int
            Number of cuboid obstacles pushed to cuRobo.
        """
        from curobo.geom.types import Cuboid, WorldConfig

        spawner = getattr(self._env, "_spawner", None)
        if spawner is None:
            self._motion_gen.update_world_obstacles(WorldConfig())
            return 0

        margin = 0.05  # collision margin per axis

        # Always include static obstacles (ground + conveyor)
        cuboids: list[Cuboid] = [
            Cuboid(
                name="ground",
                pose=[0.0, 0.0, -0.55, 1.0, 0.0, 0.0, 0.0],
                dims=[2.0, 2.0, 0.01],
            ),
            Cuboid(
                name="conveyor",
                pose=[1.55, 0.40, -0.275, 1.0, 0.0, 0.0, 0.0],
                dims=[2.90, 0.70, 1.25],
            ),
        ]
        for i, (box_prim, _path, _step) in enumerate(spawner._boxes):
            try:
                pos, _ = box_prim.get_world_poses()
                if float(pos[0, 1]) >= pallet_y_max:
                    continue
                cuboids.append(
                    Cuboid(
                        name=f"pallet_box_{i}",
                        pose=[
                            float(pos[0, 0]),
                            float(pos[0, 1]),
                            float(pos[0, 2]),
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        dims=list(default_box_dims),
                    )
                )
            except Exception:
                continue

        # Include manually tracked placed boxes with per-box dimensions.
        # pos is the box geometric centre; cuRobo Cuboid.pose expects the
        # centre, so pos is used directly (no origin offset needed here).
        if placed_boxes:
            for j, (_box_rigid, pos, dims, _offset) in enumerate(placed_boxes):
                cuboids.append(
                    Cuboid(
                        name=f"placed_box_{j}",
                        pose=[pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0],
                        dims=[d + margin for d in dims],
                    )
                )

        world_cfg = WorldConfig(cuboid=cuboids) if cuboids else WorldConfig()
        self._motion_gen.update_world(world_cfg)
        return len(cuboids)

    # ------------------------------------------------------------------
    # Box tracking utilities
    # ------------------------------------------------------------------

    @staticmethod
    def snap_rigid(box_prim: object, position: tuple[float, float, float]) -> None:
        """Teleport a RigidPrim to *position* and zero all velocities."""
        import numpy as np

        box_prim.set_world_poses(positions=np.array([position]))
        box_prim.set_linear_velocities(np.array([[0.0, 0.0, 0.0]]))
        box_prim.set_angular_velocities(np.array([[0.0, 0.0, 0.0]]))

    def get_ee_position(self) -> tuple[float, float, float]:
        """Return the world-space position of link_6 (end-effector) from the USD stage."""
        import omni.usd
        from pxr import UsdGeom

        prim_path = self._env.prim_path
        if prim_path is None:
            raise RuntimeError("env.prim_path not set (load_robot=True required)")

        stage = omni.usd.get_context().get_stage()
        base = prim_path.rsplit("/", 1)[0]  # "/p3020/root_joint" -> "/p3020"
        for cand in (f"{prim_path}/link_6", f"{base}/link_6"):
            p = stage.GetPrimAtPath(cand)
            if p.IsValid():
                t = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0).ExtractTranslation()
                return float(t[0]), float(t[1]), float(t[2])
        raise RuntimeError("link_6 prim not found in stage")

    def make_track_callback(
        self,
        box_prim: object,
        attach_z: float = _BOX_ATTACH_Z,
    ) -> Callable[[], None]:
        """Return a step callback that teleports *box_prim* to follow the EE.

        Intended for carry motions (lift → swing → pallet). Pass the returned
        callable as ``step_callback`` to :meth:`move_to_pose`.

        Parameters
        ----------
        box_prim:
            ``RigidPrim`` of the box being carried.
        attach_z:
            EE-to-box-centre vertical offset (metres).  Defaults to VGC10
            gripper length + box half-height (0.333 m).
        """

        def _track() -> None:
            ex, ey, ez = self.get_ee_position()
            self.snap_rigid(box_prim, (ex, ey, ez - attach_z))

        return _track

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_trajectory(self, result) -> list[list[float]]:
        """Convert a cuRobo plan result into a list of joint-position waypoints."""
        success = result.success.item()
        if not success:
            logger.error(
                "cuRobo planning FAILED: optimized_dt=%s, status=%s",
                getattr(result, "optimized_dt", "?"),
                getattr(result, "status", "?"),
            )
            raise RuntimeError("cuRobo MotionGen planning failed")
        positions = result.get_interpolated_plan().position
        logger.debug("cuRobo plan: %d waypoints", len(positions))
        return positions.tolist()

    def _execute_trajectory(
        self,
        traj: list[list[float]],
        step_callback: Callable[[], None] | None = None,
        pre_step_callback: Callable[[], None] | None = None,
    ) -> None:
        """Step the Isaac Sim world through each waypoint in the trajectory.

        Uses ``step_physics_only`` to avoid spawner/buffer interference
        during motion execution.  Callers that need buffer pinning should
        pass it via ``pre_step_callback``.
        """
        step_fn = getattr(self._env, "step_physics_only", self._env.step)
        for waypoint in traj:
            self._env.set_joint_positions(waypoint)
            if pre_step_callback is not None:
                pre_step_callback()
            step_fn(render=True)
            if step_callback is not None:
                step_callback()
