"""P3020 robot with optional ghost overlay and cuRobo motion planning.

Provides two reusable classes:

    P3020GhostRobot   -- transparent blue overlay that snaps to a commanded pose
    P3020Robot   -- real robot + optional ghost + optional cuRobo, given an
                    already-created Isaac Sim World

Typical usage in a demo script::

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": False})

    from isaacsim.core.api import World
    from drp_sim.robot import P3020Robot

    world = World(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/60)
    world.scene.add_default_ground_plane()

    robot = P3020Robot(world, ghost=True, curobo=True)
    robot.setup()

    robot.move_to_joints([0.0, -0.524, 1.047, 0.0, 0.0])
    robot.move_to_pose([0.8, 0.2, 1.4], quaternion=[1.0, 0.0, 0.0, 0.0])
    app.close()
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable

import yaml

from drp_sim._constants import _PACKAGE_ROOT, _PROCESSED_URDF, _URDF_PATH, N_JOINTS, preprocess_urdf

_GHOST_URDF = "/tmp/p3020_ghost_nocollision.urdf"
_CUROBO_CFG = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "usd",
        "assets",
        "robots",
        "p3020_curobo.yaml",
    )
)

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_5", "joint_6"]
HOME_JOINTS: list[float] = [0.0, 0.0, math.radians(90), math.radians(90), 0.0]

# Collision spheres for a 0.35 x 0.25 x 0.20 m box held by the VGC10 gripper.
# Box bottom face is at z = -(VGC10_LENGTH + BOX_HEIGHT) = -(0.23 + 0.20) = -0.43 m
# from link_6. Three overlapping spheres cover the box volume.
_BOX_PAYLOAD_SPHERES = [
    {"center": [0.0, 0.0, -0.30], "radius": 0.18},
    {"center": [0.0, 0.0, -0.38], "radius": 0.16},
    {"center": [0.0, 0.0, -0.45], "radius": 0.14},
]
_EE_QUAT_DEFAULT = [0.0, 1.0, 0.0, 0.0]  # 180° around X → joint_5 flipped, cup faces down


# ---------------------------------------------------------------------------
# URDF helpers
# ---------------------------------------------------------------------------


def make_ghost_urdf(
    src: str = _PROCESSED_URDF,
    dst: str = _GHOST_URDF,
) -> None:
    """Write a collision-free version of *src* for the ghost robot.

    Strips all ``<collision>`` blocks so PhysX creates no collision shapes;
    the ghost is display-only but can still be driven via joint positions.
    """
    with open(src) as f:
        content = f.read()
    content = re.sub(r"<collision\b[^>]*>.*?</collision>", "", content, flags=re.DOTALL)
    with open(dst, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# P3020GhostRobot
# ---------------------------------------------------------------------------


class P3020GhostRobot:
    """Transparent blue robot that snaps to the commanded target each step.

    Physics collision is fully disabled — the ghost is display-only.

    Parameters
    ----------
    world:
        The Isaac Sim ``World`` instance (``reset()`` must be called after
        both robots are added to the scene).
    prim_path:
        Articulation root path returned by the URDF importer for the ghost,
        e.g. ``'/P3020GhostRobot/root_joint'``.
    """

    def __init__(self, world, prim_path: str) -> None:
        import omni.usd
        from isaacsim.core.api.robots import Robot
        from pxr import Gf, Sdf, Usd, UsdShade

        self._robot = world.scene.add(Robot(prim_path=prim_path, name="p3020_ghost"))

        ghost_base = "/".join(prim_path.split("/")[:-1])

        stage = omni.usd.get_context().get_stage()
        mat_path = "/World/GhostMaterial"
        material = UsdShade.Material.Define(stage, mat_path)

        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.10, 0.50, 1.00)
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.02, 0.12, 0.40)
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.40)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        root_prim = stage.GetPrimAtPath(ghost_base)
        if root_prim.IsValid():
            binding_api = UsdShade.MaterialBindingAPI
            binding_api.Apply(root_prim)
            binding_api(root_prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        else:
            bound = 0
            for prim in Usd.PrimRange(stage.GetPseudoRoot()):
                if ghost_base in str(prim.GetPath()) and prim.GetTypeName() == "Mesh":
                    UsdShade.MaterialBindingAPI.Apply(prim)
                    UsdShade.MaterialBindingAPI(prim).Bind(
                        material, UsdShade.Tokens.strongerThanDescendants
                    )
                    bound += 1
            print(f"[ghost] fallback: material bound to {bound} mesh prims", flush=True)

    def update(self, cmd: list[float]) -> None:
        """Teleport ghost joints directly to *cmd* (radians, first ``N_JOINTS`` used)."""
        import numpy as np

        n = self._robot.num_dof
        full = np.zeros(n)
        full[:N_JOINTS] = cmd[:N_JOINTS]
        self._robot.set_joint_positions(full)


# ---------------------------------------------------------------------------
# P3020Robot
# ---------------------------------------------------------------------------


class P3020Robot:
    """Real robot + optional ghost overlay + optional cuRobo, given a World.

    The caller is responsible for creating ``SimulationApp`` and ``World``
    before constructing this object.  Call :meth:`setup` once after
    construction to import URDFs and initialise motion planning.

    Parameters
    ----------
    world:
        Isaac Sim ``World`` instance.
    ghost:
        If ``True``, a transparent ghost robot is added to the scene.
    curobo:
        If ``True``, cuRobo ``MotionGen`` is built and warmed up.
    world_obstacles:
        Optional list of cuRobo ``Cuboid`` objects added to the motion
        planner's world model.
    urdf_path:
        Override the default URDF source path.
    curobo_cfg:
        Override the default cuRobo YAML config path.
    """

    def __init__(
        self,
        world,
        *,
        ghost: bool = False,
        curobo: bool = True,
        world_obstacles: list | None = None,
        urdf_path: str = _URDF_PATH,
        curobo_cfg: str = _CUROBO_CFG,
    ) -> None:
        self._world = world
        self._ghost_enabled = ghost
        self._curobo_enabled = curobo
        self._obstacles = world_obstacles or []
        self._urdf_path = urdf_path
        self._curobo_cfg = os.path.abspath(curobo_cfg)

        self._robot = None
        self._prim_path: str | None = None
        self._ghost: P3020GhostRobot | None = None
        self._mg = None
        self._mg_payload = None  # payload-aware MotionGen (lazy-built on first attach)
        self._payload_attached: bool = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Preprocess URDF, import robot(s), reset world, and init cuRobo.

        Must be called once before any motion methods.
        """
        preprocess_urdf(self._urdf_path, _PROCESSED_URDF)
        self._prim_path = self._import_real_robot()

        if self._ghost_enabled:
            make_ghost_urdf(_PROCESSED_URDF, _GHOST_URDF)
            ghost_prim = self._import_ghost_robot()
            self._world.reset()
            self._ghost = P3020GhostRobot(self._world, ghost_prim)
        else:
            self._world.reset()

        # Position drive gains chosen for stability at 60 Hz with PhysX.
        # stiffness = 2e3 N*m/rad:
        #   wrist joint (I ~ 0.1 kg*m^2): wn = sqrt(2e3/0.1) = 141 rad/s < 188 (Nyquist) -> stable
        #   shoulder (I ~ 3 kg*m^2):      wn = 26 rad/s -> good, sag < 9 deg at 300 N*m
        # damping = 1e3 N*m*s/rad -> all joints over-damped, no oscillation
        self._set_drive_gains(self._prim_path, stiffness=2e3, damping=1e3)

        if self._curobo_enabled:
            self._mg = self._build_motion_gen()

    @classmethod
    def from_existing(
        cls,
        world,
        robot,
        prim_path: str,
        world_obstacles: list | None = None,
        curobo_cfg: str = _CUROBO_CFG,
    ) -> P3020Robot:
        """Wrap an already-imported robot articulation with cuRobo MotionGen.

        Use when PalletizerEnv has already loaded the robot into the stage so
        re-importing the URDF is not needed.  ``_PROCESSED_URDF`` must already
        exist — PalletizerEnv._import_urdf calls preprocess_urdf() before URDF
        import so the file is always available when load_robot=True.

        Parameters
        ----------
        world:
            Isaac Sim World instance (already reset via env.reset()).
        robot:
            Existing Robot articulation object from the scene.
        prim_path:
            Articulation root prim path (e.g. ``'/p3020/root_joint'``).
        world_obstacles:
            Optional cuRobo Cuboid obstacles for collision avoidance.
        curobo_cfg:
            Override the cuRobo YAML config path.
        """
        obj = cls.__new__(cls)
        obj._world = world
        obj._ghost_enabled = False
        obj._curobo_enabled = True
        obj._obstacles = world_obstacles or []
        obj._urdf_path = _URDF_PATH
        obj._curobo_cfg = os.path.abspath(curobo_cfg)
        obj._robot = robot
        obj._prim_path = prim_path
        obj._ghost = None
        obj._mg = None
        obj._mg_payload = None
        obj._payload_attached = False
        obj._mg = obj._build_motion_gen()
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def robot(self):
        """The underlying Isaac Sim ``Robot`` articulation object."""
        return self._robot

    @property
    def ghost(self) -> P3020GhostRobot | None:
        """The ``P3020GhostRobot`` instance, or ``None`` if ghost was disabled."""
        return self._ghost

    @property
    def motion_gen(self):
        """The cuRobo ``MotionGen`` instance, or ``None`` if cuRobo was disabled."""
        return self._mg

    @property
    def prim_path(self) -> str | None:
        """Articulation root prim path of the real robot."""
        return self._prim_path

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_to_joints(
        self,
        goal: list[float],
        *,
        steps_per_wp: int = 12,
        render: bool = True,
        step_callback: Callable[[], None] | None = None,
        pre_step_callback: Callable[[], None] | None = None,
    ) -> bool:
        """Move to a joint-space goal using cuRobo (falls back to linear interpolation).

        The ghost, if present, snaps to *goal* immediately before motion starts.

        Parameters
        ----------
        goal:
            Target joint positions in radians for the ``N_JOINTS`` active joints.
        steps_per_wp:
            Physics steps rendered per trajectory waypoint.
        render:
            Whether to call ``world.step(render=True)``.

        Returns
        -------
        bool
            ``True`` if cuRobo planning succeeded (or no cuRobo); ``False`` if
            planning failed and linear interpolation was used instead.
        """
        if self._ghost:
            self._ghost.update(goal)
            if pre_step_callback is not None:
                pre_step_callback()
            self._world.step(render=render)
            if step_callback is not None:
                step_callback()

        mg = self._active_mg
        if mg is not None:
            import torch
            from curobo.types.robot import JointState
            from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

            current = self._robot.get_joint_positions()[:N_JOINTS].tolist()
            result = mg.plan_single_js(
                JointState.from_position(torch.tensor([current], dtype=torch.float32).cuda()),
                JointState.from_position(torch.tensor([goal], dtype=torch.float32).cuda()),
                MotionGenPlanConfig(max_attempts=2),
            )
            if result.success.item():
                traj = result.get_interpolated_plan().position.tolist()
                self._execute_traj(
                    traj,
                    goal,
                    steps_per_wp,
                    render,
                    step_callback,
                    pre_step_callback,
                )
                return True

            # cuRobo failed — fall back
            print("[P3020Robot] cuRobo plan failed, falling back to interpolation", flush=True)

        self._interpolate(goal, render=render)
        return False

    def move_to_pose(
        self,
        position: list[float],
        *,
        quaternion: list[float] | None = None,
        orientation_constraint: bool = False,
        max_attempts: int = 15,
        steps_per_wp: int = 12,
        render: bool = True,
        step_callback: Callable[[], None] | None = None,
        pre_step_callback: Callable[[], None] | None = None,
    ) -> bool:
        """Move end-effector to a Cartesian pose using cuRobo.

        Parameters
        ----------
        position:
            ``[x, y, z]`` in metres.
        quaternion:
            ``[w, x, y, z]`` unit quaternion.  Defaults to identity.
        orientation_constraint:
            When ``True``, hold the goal orientation fixed throughout the
            entire trajectory (not just at the goal).  Prevents end-effector
            tilting during carry motions.  Uses cuRobo ``PoseCostMetric``
            with rotation-only hold weights ``[0, 0, 0, 1, 1, 1]``.
        max_attempts:
            cuRobo planning attempts before giving up.
        steps_per_wp:
            Physics steps rendered per trajectory waypoint.
        render:
            Whether to call ``world.step(render=True)``.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if planning failed.
        """
        mg = self._active_mg
        if mg is None:
            raise RuntimeError("cuRobo is disabled — cannot call move_to_pose()")

        quat = quaternion if quaternion is not None else _EE_QUAT_DEFAULT

        import torch
        from curobo.types.math import Pose
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig, PoseCostMetric

        current = self._robot.get_joint_positions()[:N_JOINTS].tolist()
        start = JointState.from_position(torch.tensor([current], dtype=torch.float32).cuda())
        goal = Pose(
            position=torch.tensor([position], dtype=torch.float32),
            quaternion=torch.tensor([quat], dtype=torch.float32),
        )

        plan_cfg = MotionGenPlanConfig(max_attempts=max_attempts)
        if orientation_constraint:
            # hold_partial_pose=True: apply constraint at every trajectory waypoint,
            # not only at the goal.  hold_vec_weight=[px, py, pz, rx, ry, rz]:
            # zero position weights (free path) + unit rotation weights (hold orientation).
            plan_cfg.pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            )

        result = mg.plan_single(start, goal, plan_cfg)
        if not result.success.item():
            print(f"[P3020Robot] IK/plan failed for position {position}", flush=True)
            return False

        traj = result.get_interpolated_plan().position.tolist()
        ghost_target = traj[-1][:N_JOINTS]
        if self._ghost:
            self._ghost.update(ghost_target)
            if pre_step_callback is not None:
                pre_step_callback()
            self._world.step(render=render)
            if step_callback is not None:
                step_callback()
        self._execute_traj(
            traj,
            ghost_target,
            steps_per_wp,
            render,
            step_callback,
            pre_step_callback,
        )
        return True

    def go_home(
        self,
        *,
        render: bool = True,
        step_callback: Callable[[], None] | None = None,
    ) -> bool:
        """Move to HOME_JOINTS using linear interpolation.

        Bypasses cuRobo plan_single_js to avoid post-plan CUDA state issues.
        The home pose [0, 0, 90, 90, 0] deg is always safely reachable by interpolation.
        """
        if self._ghost:
            self._ghost.update(HOME_JOINTS)
            self._world.step(render=render)
        self._interpolate(HOME_JOINTS, render=render, step_callback=step_callback)
        return True

    def attach_payload(self) -> None:
        """Mark box as attached; lazy-build payload-aware MotionGen on first call.

        The payload MotionGen adds collision spheres for the held cardboard box
        (0.35 x 0.25 x 0.20 m) attached below the VGC10 cup so subsequent
        motion planning keeps the box clear of obstacles.
        """
        self._payload_attached = True
        if self._mg is not None and self._mg_payload is None:
            print("[P3020Robot] Building payload MotionGen...", flush=True)
            self._mg_payload = self._build_motion_gen(payload_spheres=True)
            print("[P3020Robot] Payload MotionGen ready", flush=True)

    def detach_payload(self) -> None:
        """Revert to base MotionGen (no held box)."""
        self._payload_attached = False

    @property
    def _active_mg(self):
        """Return payload-aware MotionGen when holding a box, base otherwise."""
        if self._payload_attached and self._mg_payload is not None:
            return self._mg_payload
        return self._mg

    def step(
        self,
        n: int = 1,
        *,
        render: bool = True,
        step_callback: Callable[[], None] | None = None,
        pre_step_callback: Callable[[], None] | None = None,
    ) -> None:
        """Advance physics by *n* steps."""
        for _ in range(n):
            if pre_step_callback is not None:
                pre_step_callback()
            self._world.step(render=render)
            if step_callback is not None:
                step_callback()

    def hold(self, goal: list[float], steps: int = 120, *, render: bool = True) -> None:
        """Hold *goal* for *steps* physics steps, keeping ghost pinned to *goal*.

        Re-applies the position command every step so the drive actively
        resists any residual physics drift.
        """
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        n = self._robot.num_dof
        full = np.zeros(n)
        full[:N_JOINTS] = goal[:N_JOINTS]
        action = ArticulationAction(joint_positions=full, joint_velocities=np.zeros(n))
        for _ in range(steps):
            self._robot.get_articulation_controller().apply_action(action)
            if self._ghost:
                self._ghost.update(goal)
            self._world.step(render=render)

    def get_joint_positions(self) -> list[float]:
        """Return current joint positions (radians) for the ``N_JOINTS`` active joints."""
        return self._robot.get_joint_positions()[:N_JOINTS].tolist()

    def _teleport_to_home(self) -> None:
        """Instantly set joints to HOME_JOINTS without interpolation or stepping."""
        import numpy as np

        n = self._robot.num_dof
        full = np.zeros(n)
        full[:N_JOINTS] = HOME_JOINTS[:N_JOINTS]
        self._robot.set_joint_positions(full)
        if self._ghost:
            self._ghost.update(HOME_JOINTS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_drive_gains(self, prim_path: str, stiffness: float, damping: float) -> None:
        """Set angular drive stiffness and damping on all real-robot joints.

        Only modifies joints under the real robot root; the ghost robot (nested
        under the same base path) is excluded to keep it purely kinematic.
        """
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        robot_root = prim_path.rsplit("/", 1)[0]  # e.g., /p3020
        # Ghost robot lives at /p3020/p3020/...; skip it entirely.
        ghost_prefix = robot_root + "/" + robot_root.lstrip("/")
        count = 0
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if not path.startswith(robot_root + "/"):
                continue
            if path.startswith(ghost_prefix):
                continue
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                drive.GetStiffnessAttr().Set(stiffness)
                drive.GetDampingAttr().Set(damping)
                count += 1
        print(
            f"[P3020Robot] drive gains set on {count} joints (k={stiffness:.0e}, d={damping:.0e})",
            flush=True,
        )

    def _urdf_import_config(self):
        import omni.kit.commands

        _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
        cfg.merge_fixed_joints = False
        cfg.convex_decomp = False
        cfg.import_inertia_tensor = True
        cfg.fix_base = True
        cfg.distance_scale = 1.0
        cfg.default_position_drive_damping = 1e3
        return cfg

    def _import_real_robot(self) -> str:
        import omni.kit.commands
        from isaacsim.core.api.robots import Robot

        _, prim = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=_PROCESSED_URDF,
            import_config=self._urdf_import_config(),
            dest_path="",
            get_articulation_root=True,
        )
        self._robot = self._world.scene.add(Robot(prim_path=prim, name="p3020"))
        print(f"[P3020Robot] real robot → {prim}", flush=True)
        return prim

    def _import_ghost_robot(self) -> str:
        import omni.kit.commands

        _, prim = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=_GHOST_URDF,
            import_config=self._urdf_import_config(),
            dest_path="",
            get_articulation_root=True,
        )
        if prim is None:
            raise RuntimeError("Ghost URDF import returned None prim path")
        print(f"[P3020Robot] ghost robot → {prim}", flush=True)
        return prim

    def _build_motion_gen(self, payload_spheres: bool = False):
        from curobo.types.robot import RobotConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        with open(self._curobo_cfg) as f:
            cfg_dict = yaml.safe_load(f)
        cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = _PROCESSED_URDF
        cfg_dict["robot_cfg"]["kinematics"]["asset_root_path"] = _PACKAGE_ROOT

        if payload_spheres:
            # Append box collision spheres to link_6 so planner keeps the held
            # box clear of obstacles (conveyor, pallet stands).
            link6 = cfg_dict["robot_cfg"]["kinematics"]["collision_spheres"]["link_6"]
            link6.extend(_BOX_PAYLOAD_SPHERES)

            # The box extends ~0.43 m below link_6.  During carry motions the
            # box volume can legitimately be close to the robot's upper-arm and
            # forearm links.  Without these ignores cuRobo reports false
            # self-collisions and either fails to plan or produces wild paths.
            sci = cfg_dict["robot_cfg"]["kinematics"]["self_collision_ignore"]
            existing = set(sci.get("link_6", []))
            sci["link_6"] = sorted(existing | {"link_5", "link_4", "link_3", "link_2"})

            # Reduce dynamics for 30 kg payload: slower, smoother motion.
            # Heavily penalise joint_1 (base rotation) to prevent spinning
            # during pick-and-place carry phase.
            cspace = cfg_dict["robot_cfg"]["kinematics"]["cspace"]
            cspace["max_acceleration"] = 5.0  # default 15.0
            cspace["velocity_scale"] = 0.5  # default 1.0
            cspace["max_jerk"] = 100.0  # default 500.0
            # joint order: joint_1, joint_2, joint_3, joint_5, joint_6
            cspace["null_space_weight"] = [4.0, 4.0, 0.5, 1.0, 0.5]
            cspace["cspace_distance_weight"] = [4.0, 4.0, 0.5, 1.0, 0.5]

        robot_cfg = RobotConfig.from_dict(cfg_dict)

        # Use PRIMITIVE collision checker (cuboids only) to avoid the
        # warp-dependent WorldMeshCollision.  WorldMeshCollision requires
        # Isaac Sim's bundled warp (1.8.x) to be CUDA-initialized, but that
        # initialization is skipped when curobo/warp (1.11.x) is imported
        # before SimulationApp boots.  PRIMITIVE supports all cuboid obstacles
        # used here and is warp-free.
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig

        # Payload MotionGen plans the carry phase (robot has already lifted clear
        # of obstacles).  Keeping scene obstacles in the payload planner causes IK
        # failure: the lift start pose sits exactly on the conveyor obstacle
        # boundary once the payload spheres extend below link_6.  The base
        # MotionGen (pick / lift approach) correctly enforces obstacle avoidance.
        obs = [] if payload_spheres else self._obstacles
        world_cfg = WorldConfig(cuboid=obs) if obs else WorldConfig()
        mg_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_model=world_cfg,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            interpolation_dt=1.0 / 60.0,
        )

        mg = MotionGen(mg_cfg)
        # Payload MotionGen skips the CUDA graph to avoid corrupting shared
        # GPU state between the two planners; planning is ~100 ms vs 45 ms.
        mg.warmup(enable_graph=not payload_spheres)
        print("[P3020Robot] cuRobo ready", flush=True)
        return mg

    def _execute_traj(
        self,
        traj: list[list[float]],
        ghost_target: list[float],
        steps_per_wp: int,
        render: bool,
        step_callback: Callable[[], None] | None = None,
        pre_step_callback: Callable[[], None] | None = None,
        traj_vel: list[list[float]] | None = None,
    ) -> None:
        """Execute a cuRobo interpolated trajectory waypoint-by-waypoint.

        Joints are teleported to each waypoint via ``set_joint_positions()``,
        bypassing PhysX drive dynamics entirely.  This gives exact trajectory
        tracking regardless of joint inertia -- no oscillation, no spin-out,
        no accumulated lag.

        The root cause of the previous trembling: with k=2e3, d=150 the base
        joint (I~8 kg m^2) is underdamped (zeta=0.59) and spins out; no single
        damping value satisfies zeta>=1 for both the base (I~8) and wrist
        (I~0.1) simultaneously.  Teleporting bypasses this constraint entirely.

        The drive setpoint is kept in sync so PD hold gains lock the robot
        at the final pose after the trajectory ends.

        ``steps_per_wp`` and ``traj_vel`` are accepted for API compatibility
        but not used.  cuRobo trajectories are already interpolated at the
        simulation rate (60 Hz) so one physics step per waypoint is correct.
        """
        import csv
        import os

        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        n = self._robot.num_dof
        full_pos = np.zeros(n)

        # Joint-state monitor: written when JOINT_MONITOR env var is set.
        monitor_path = os.environ.get("JOINT_MONITOR")
        mf = None
        if monitor_path:
            mf = open(monitor_path, "a", newline="")  # noqa: SIM115
            wr = csv.writer(mf)
            if os.path.getsize(monitor_path) == 0:
                wr.writerow(
                    [
                        "wp",
                        "j1c",
                        "j2c",
                        "j3c",
                        "j5c",
                        "j6c",
                        "j1a",
                        "j2a",
                        "j3a",
                        "j5a",
                        "j6a",
                        "j1v",
                        "j2v",
                        "j3v",
                        "j5v",
                        "j6v",
                        "j1e",
                        "j2e",
                        "j3e",
                        "j5e",
                        "j6e",
                    ]
                )

        max_err = 0.0
        max_vel_obs = 0.0

        try:
            for wp_idx, wp in enumerate(traj):
                full_pos[:N_JOINTS] = wp[:N_JOINTS]
                # Teleport joints to exact trajectory position.
                self._robot.set_joint_positions(full_pos)
                # Keep drive reference in sync for stable hold after motion.
                self._robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=full_pos.copy())
                )
                if self._ghost:
                    self._ghost.update(ghost_target)
                if pre_step_callback is not None:
                    pre_step_callback()
                self._world.step(render=render)
                if step_callback is not None:
                    step_callback()
                if mf is not None:
                    act = self._robot.get_joint_positions()[:N_JOINTS].tolist()
                    vel = self._robot.get_joint_velocities()[:N_JOINTS].tolist()
                    err = [act[j] - wp[j] for j in range(N_JOINTS)]
                    wr.writerow([wp_idx, *wp[:N_JOINTS], *act, *vel, *err])
                    max_err = max(max_err, max(abs(e) for e in err))
                    max_vel_obs = max(max_vel_obs, max(abs(v) for v in vel))
        finally:
            if mf is not None:
                mf.close()
                print(
                    f"[monitor] max_track_err={max_err:.4f} rad  max_vel={max_vel_obs:.3f} rad/s",
                    flush=True,
                )

    def _interpolate(
        self,
        goal: list[float],
        steps: int = 180,
        render: bool = True,
        step_callback: Callable[[], None] | None = None,
    ) -> None:
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        start = self._robot.get_joint_positions()[:N_JOINTS].tolist()
        n = self._robot.num_dof
        for i in range(1, steps + 1):
            t = i / steps
            ts = t * t * (3 - 2 * t)
            cmd = (np.array(start) + ts * (np.array(goal) - np.array(start))).tolist()
            full = np.zeros(n)
            full[:N_JOINTS] = cmd
            self._robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=full)
            )
            if self._ghost:
                self._ghost.update(goal)
            self._world.step(render=render)
            if step_callback is not None:
                step_callback()
