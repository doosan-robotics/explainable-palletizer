"""Isaac Sim palletizer environment loading a pre-built USD scene.

Must be run inside the Isaac Sim Python environment, not the uv workspace:
    uv run python -c "from drp_sim import PalletizerEnv; ..."

The SimulationApp must be created before any ``omni.*`` or ``isaacsim.core``
imports (Isaac Sim 5.1 bootstrap requirement).  Callers are responsible for
creating the app first or using the ``headless`` parameter in ``__init__``.
"""

from __future__ import annotations

import contextlib
import random
from pathlib import Path
from typing import TYPE_CHECKING

from drp_sim._constants import _PROCESSED_URDF, _URDF_PATH, N_JOINTS, preprocess_urdf
from drp_sim.box_image_capture import BoxImageCapture
from drp_sim.box_spawn_config import BufferParams, load_box_spawn_config
from drp_sim.box_spawner import BoxSpawner, BoxTypeConfig
from drp_sim.conveyor_buffer import ConveyorBuffer
from drp_sim.sticker_attacher import StickerAttacher

if TYPE_CHECKING:
    from isaacsim.core.prims import RigidPrim

# USD scene bundled with the sim package: sim/usd/mixed_palletizing_scene.usd
_SIM_ROOT = Path(__file__).resolve().parent.parent.parent
_USD_SCENE = str(_SIM_ROOT / "usd" / "mixed_palletizing_scene.usd")

_ROBOT_NAME = "p3020"


def _suppress_semantics_warning() -> None:
    """Silence the 'SemanticsAPI is deprecated' warning from Isaac Sim internals."""
    import carb

    settings = carb.settings.get_settings()
    settings.set("/log/semantics.schema.property/level", "error")


def _patch_rigid_prim_del() -> None:
    """Patch Isaac Sim 5.1 RigidPrim.__del__ to avoid AttributeError on shutdown.

    The bug: RigidPrim.__del__ calls XFormPrim.destroy() which iterates
    self._callbacks, but _callbacks is never initialised when the prim wrapper
    is garbage-collected after the USD stage has already been torn down.
    """
    from isaacsim.core.prims import RigidPrim

    if getattr(RigidPrim, "_patched_del", False):
        return

    _original_del = RigidPrim.__del__

    def _safe_del(self: object) -> None:
        with contextlib.suppress(AttributeError):
            _original_del(self)

    RigidPrim.__del__ = _safe_del
    RigidPrim._patched_del = True  # type: ignore[attr-defined]


class PalletizerEnv:
    """Isaac Sim environment that loads the pre-built mixed palletizing USD scene.

    Parameters
    ----------
    headless:
        Start Isaac Sim without a GUI window.
    usd_path:
        Absolute path to the scene USD file.
    load_robot:
        Import the p3020 URDF and add it to the scene. Requires the URDF and
        mesh files to be present at the paths defined by ``_URDF_PATH`` and
        ``_PACKAGE_ROOT``.
    urdf_path:
        Override the default URDF path (only used when load_robot=True).
    spawn_boxes:
        Enable automatic box spawning on the conveyor belt.
    spawn_position:
        (x, y, z) world position where new boxes appear.
    box_scale:
        (sx, sy, sz) scale applied to each spawned box.
    spawn_interval:
        Time in seconds between automatic box spawns.
    seed:
        Random seed for reproducibility.  Seeds both ``random`` and
        ``numpy.random`` at the start of ``reset()``.  ``None`` disables
        seeding.
    generate_box_images:
        When set, capture a top-down RGB image of each spawned box and save
        to this directory.  Uses Omni Replicator ``BasicWriter``.
    box_num:
        Stop spawning and exit after this many boxes (and images, if
        ``generate_box_images`` is set).  ``None`` means run indefinitely.
    type_weights:
        Mapping of box type name to sampling weight passed to
        ``BoxSpawner``.  ``None`` uses the spawner's built-in defaults.
    config_path:
        Path to a YAML box spawn config file.  When provided, box type
        definitions and spawn parameters are loaded from the file.
        CLI arguments (type_weights, spawn_interval) override YAML values.
    """

    def __init__(
        self,
        headless: bool = True,
        usd_path: str = _USD_SCENE,
        load_robot: bool = False,
        urdf_path: str = _URDF_PATH,
        spawn_boxes: bool = True,
        spawn_interval: float | None = None,
        robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        seed: int | None = 42,
        generate_box_images: str | None = None,
        box_num: int | None = None,
        type_weights: dict[str, float] | None = None,
        config_path: str | None = None,
    ) -> None:
        from isaacsim import SimulationApp

        self._app = SimulationApp({"headless": headless})
        self._init_fields(
            usd_path=usd_path,
            load_robot=load_robot,
            urdf_path=urdf_path,
            spawn_boxes=spawn_boxes,
            spawn_interval=spawn_interval,
            robot_position=robot_position,
            seed=seed,
            generate_box_images=generate_box_images,
            box_num=box_num,
            type_weights=type_weights,
            config_path=config_path,
        )

    @classmethod
    def from_app(
        cls,
        sim_app: object,
        *,
        usd_path: str = _USD_SCENE,
        load_robot: bool = False,
        urdf_path: str = _URDF_PATH,
        spawn_boxes: bool = True,
        spawn_interval: float | None = None,
        robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        seed: int | None = 42,
        generate_box_images: str | None = None,
        box_num: int | None = None,
        type_weights: dict[str, float] | None = None,
        config_path: str | None = None,
    ) -> PalletizerEnv:
        """Create a PalletizerEnv using an already-running SimulationApp.

        Use this when the SimulationApp was created externally (e.g. by the
        sim server entrypoint) and must not be re-created.

        Parameters
        ----------
        sim_app:
            An already-created ``SimulationApp`` instance.
        """
        env = cls.__new__(cls)
        env._app = sim_app
        env._init_fields(
            usd_path=usd_path,
            load_robot=load_robot,
            urdf_path=urdf_path,
            spawn_boxes=spawn_boxes,
            spawn_interval=spawn_interval,
            robot_position=robot_position,
            seed=seed,
            generate_box_images=generate_box_images,
            box_num=box_num,
            type_weights=type_weights,
            config_path=config_path,
        )
        return env

    def _init_fields(
        self,
        *,
        usd_path: str,
        load_robot: bool,
        urdf_path: str,
        spawn_boxes: bool,
        spawn_interval: float | None,
        robot_position: tuple[float, float, float],
        seed: int | None,
        generate_box_images: str | None,
        box_num: int | None,
        type_weights: dict[str, float] | None,
        config_path: str | None,
    ) -> None:
        """Shared initialisation used by both __init__ and from_app."""
        _patch_rigid_prim_del()
        _suppress_semantics_warning()
        self._usd_path = usd_path
        self._load_robot = load_robot
        self._urdf_path = urdf_path
        self._spawn_boxes = spawn_boxes
        self._robot_position = robot_position
        self._seed = seed
        self._generate_box_images = generate_box_images
        self._box_num = box_num

        # Layered config: code defaults -> YAML -> CLI args
        configs, weights, spawn, buffer = load_box_spawn_config(config_path)
        self._type_configs: dict[str, BoxTypeConfig] | None = configs
        self._spawn_params = spawn
        self._buffer_params: BufferParams = buffer
        self._type_weights = type_weights or weights
        self._spawn_interval = spawn_interval if spawn_interval is not None else spawn.interval
        self._prim_path: str | None = None
        self._world = None
        self._robot = None
        self._spawner: BoxSpawner | None = None
        self._image_capture: BoxImageCapture | None = None
        self._last_box_count: int = 0
        self._spawner_paused: bool = False
        self._random_light_path: str = ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Initialise (or re-initialise) the World from the USD scene."""
        import numpy as np
        import omni.usd
        from isaacsim.core.api import World

        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)

        # Open the pre-built palletizing scene
        omni.usd.get_context().open_stage(self._usd_path)

        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )

        if self._load_robot:
            from isaacsim.core.api.robots import Robot

            self._prim_path = self._import_urdf()
            if any(self._robot_position):
                self._place_robot(self._prim_path, self._robot_position)
            self._robot = self._world.scene.add(Robot(prim_path=self._prim_path, name=_ROBOT_NAME))

        self._world.reset()
        self._setup_random_light()
        self._last_box_count = 0

        if self._spawn_boxes:
            sticker_attacher = self._resolve_sticker_attacher()

            self._spawner = BoxSpawner(
                world=self._world,
                type_configs=self._type_configs,
                type_weights=self._type_weights,
                spawn_position=self._spawn_params.position,
                velocity=self._spawn_params.velocity,
                spawn_interval=self._spawn_interval,
                sticker_attacher=sticker_attacher,
                box_ttl=self._spawn_interval,
            )
            # Pre-compute native bboxes so Usd.Stage.Open() is never
            # called during active simulation — opening a secondary USD
            # stage while Fabric is running triggers prototype cascade
            # invalidation that crashes PhysX.
            self._warm_spawner_bboxes()
            self._buffer = ConveyorBuffer(
                spawner=self._spawner,
                endpoint=self._buffer_params.endpoint,
                length=self._buffer_params.length,
                conveyor_velocity=self._spawn_params.velocity,
            )
        else:
            self._spawner = None
            self._buffer = None

        if self._spawner is not None:
            if self._generate_box_images:
                self._image_capture = BoxImageCapture(output_dir=self._generate_box_images)
            else:
                self._image_capture = BoxImageCapture(output_dir=None)
            self._image_capture.setup()
        else:
            self._image_capture = None

    def step(self, render: bool = True) -> None:
        """Advance the simulation by one physics step."""
        if self._world is None:
            raise RuntimeError("Call reset() before step()")
        if self._spawner is not None and self._world.is_playing() and not self._spawner_paused:
            if self._box_num is None or self._spawner.box_count < self._box_num:
                self._spawner.step()
            if self._buffer is not None and self._buffer.active:
                self._buffer.step()
            if self._spawner.box_count > self._last_box_count:
                self._randomize_lighting()
                self._last_box_count = self._spawner.box_count
        self._world.step(render=render)
        if self._image_capture is not None and self._spawner is not None:
            self._image_capture.step(self._spawner.box_count, self._spawner.box_metadata)

    def step_physics_only(self, render: bool = True) -> None:
        """Advance physics without spawner/buffer logic.

        Use during pick-and-place motions where spawner and buffer
        must not interfere (no new spawns, no compaction).
        """
        if self._world is None:
            raise RuntimeError("Call reset() before step()")
        self._world.step(render=render)

    def fill_buffer(self) -> dict:
        """Fill the conveyor buffer with boxes and stop auto-spawning.

        Returns a status dict with occupancy info.
        """
        if self._buffer is None:
            raise RuntimeError("spawn_boxes=True required and reset() must be called first")
        return self._buffer.fill()

    def spawn_box(self) -> str:
        """Manually spawn a box on the conveyor and return its prim path.

        Raises
        ------
        RuntimeError
            If the environment was not created with spawn_boxes=True or reset()
            has not been called yet.
        """
        if self._spawner is None:
            raise RuntimeError("spawn_boxes=True required and reset() must be called first")
        return self._spawner.spawn()

    def reset_boxes(self) -> int:
        """Lightweight reset: hide all boxes, reset spawner+buffer state.

        Unlike reset(), does NOT reload the USD scene or recreate World.
        """
        hidden = 0
        if self._buffer is not None:
            hidden += self._buffer.reset()
        if self._spawner is not None:
            hidden += self._spawner.hide_all()
        self._last_box_count = 0
        return hidden

    def remove_buffer_box(self, index: int) -> tuple[str, int] | None:
        """Remove a single box from a buffer slot (human takes it).

        Returns (prim_path, hidden_count) or None if no box at slot.
        """
        import numpy as np

        if self._buffer is None:
            return None
        result = self._buffer.pop_box_at(index)
        if result is None:
            return None
        prim, path = result
        prim.set_linear_velocities(np.zeros((1, 3)))
        prim.set_angular_velocities(np.zeros((1, 3)))
        prim.set_world_poses(positions=np.array([[0.0, 0.0, -1000.0]]))
        self._buffer.release_box(prim)
        return path, 1

    @property
    def boxes(self) -> list:
        """All spawned box RigidPrim objects, or empty list if spawner inactive."""
        if self._spawner is None:
            return []
        return self._spawner.boxes

    @property
    def active_box(self) -> RigidPrim | None:
        """The box currently on the conveyor, or None."""
        if self._spawner is None:
            return None
        return self._spawner.active_box

    @property
    def done(self) -> bool:
        """True when the box limit has been reached and all images are captured."""
        if self._box_num is None:
            return False
        if self._image_capture is not None:
            return self._image_capture.capture_count >= self._box_num
        if self._spawner is not None:
            return self._spawner.box_count >= self._box_num
        return False

    @property
    def image_capture(self) -> BoxImageCapture | None:
        return self._image_capture

    def set_joint_positions(self, positions: list[float]) -> None:
        """Set the 5 active joint positions (radians).

        Parameters
        ----------
        positions:
            Values for [joint_1, joint_2, joint_3, joint_5, joint_6].
        """
        if len(positions) != N_JOINTS:
            raise ValueError(f"Expected {N_JOINTS} joint positions, got {len(positions)}")
        if self._robot is None:
            raise RuntimeError("load_robot=True required and reset() must be called first")

        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        full = np.zeros(self._robot.num_dof)
        full[:N_JOINTS] = positions
        self._robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=full)
        )

    def get_joint_positions(self) -> list[float]:
        """Return the 5 active joint positions (radians)."""
        if self._robot is None:
            raise RuntimeError("load_robot=True required and reset() must be called first")
        positions = self._robot.get_joint_positions()
        return positions[:N_JOINTS].tolist()

    @property
    def prim_path(self) -> str | None:
        """Articulation root prim path (available after reset() with load_robot=True)."""
        return self._prim_path

    def close(self) -> None:
        """Shut down the Isaac Sim application.

        Cleans up scene objects in the correct order to avoid the Isaac Sim 5.1
        RigidPrim.__del__ AttributeError ('_callbacks' not found) that occurs
        when prim wrappers are garbage-collected after the USD stage is torn down.
        """
        self._image_capture = None
        self._buffer = None
        if self._spawner is not None:
            self._spawner.clear()
            self._spawner = None
        self._robot = None
        if self._world is not None:
            self._world.stop()
            self._world.clear()
            self._world = None
        self._app.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warm_spawner_bboxes(self) -> None:
        """Pre-compute native bounding boxes for all box USD assets.

        ``BoxSpawner._resolve_variant`` lazily calls ``Usd.Stage.Open()``
        to compute bboxes.  Opening a secondary USD stage while Fabric is
        active triggers prototype cascade invalidation that crashes PhysX.
        Warming the cache here (after reset, before PLAY) avoids that.
        """
        if self._spawner is None or self._type_configs is None:
            return
        from drp_sim.box_spawner import BoxSpawner

        for config in self._type_configs.values():
            for usd_path in config.usd_paths:
                if usd_path not in self._spawner._bbox_cache:
                    self._spawner._bbox_cache[usd_path] = BoxSpawner._compute_native_bbox(usd_path)

    def _resolve_sticker_attacher(self) -> StickerAttacher | None:
        """Build a StickerAttacher from the spawn params, if configured."""
        metadata = self._spawn_params.sticker_metadata
        if not metadata:
            return None
        p = Path(metadata)
        if not p.is_absolute():
            textures_dir = Path(__file__).parent.parent.parent / "usd" / "assets" / "textures"
            p = textures_dir / metadata
        return StickerAttacher(metadata_path=p)

    def _setup_random_light(self) -> None:
        """Create shadow-free multi-directional lighting for uniform visibility.

        Uses four DistantLights (top, front, left, right) instead of a DomeLight
        because DomeLight is unreliable in headless replicator capture mode.
        All lights have shadows disabled so every surface is visible from any view.
        """
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()

        # (direction, rotateXYZ) pairs for four principal directions.
        # DistantLight default emission is -Z; rotations redirect it.
        fixed_lights = [
            ("/World/Light_Top", Gf.Vec3f(-90.0, 0.0, 0.0), 500),  # downward  (-Y)
            ("/World/Light_Front", Gf.Vec3f(0.0, 0.0, 0.0), 300),  # toward -Z
            ("/World/Light_Left", Gf.Vec3f(0.0, 90.0, 0.0), 300),  # toward -X
            ("/World/Light_Right", Gf.Vec3f(0.0, -90.0, 0.0), 300),  # toward +X
        ]
        for path, rotation, intensity in fixed_lights:
            lt = UsdLux.DistantLight.Define(stage, path)
            lt.CreateIntensityAttr(intensity)
            lt.CreateAngleAttr(1.0)
            UsdLux.ShadowAPI.Apply(lt.GetPrim()).CreateShadowEnableAttr(False)
            UsdGeom.Xformable(lt.GetPrim()).AddRotateXYZOp().Set(rotation)

        # Weak randomised fill light for domain randomisation.
        rand_path = "/World/DomainRandomLight"
        rand_lt = UsdLux.DistantLight.Define(stage, rand_path)
        rand_lt.CreateIntensityAttr(60)
        rand_lt.CreateAngleAttr(1.0)
        UsdLux.ShadowAPI.Apply(rand_lt.GetPrim()).CreateShadowEnableAttr(False)
        UsdGeom.Xformable(rand_lt.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        self._random_light_path = rand_path
        self._randomize_lighting()

    def _randomize_lighting(self) -> None:
        """Apply a small random tilt to the domain-randomisation light."""
        import omni.usd
        from pxr import Gf, UsdGeom

        if not self._random_light_path:
            return
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._random_light_path)
        if not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rx = random.uniform(-15.0, 15.0)
                ry = random.uniform(-15.0, 15.0)
                op.Set(Gf.Vec3f(rx, ry, 0.0))
                break

    def _import_urdf(self) -> str:
        """Import p3020 URDF into the current stage. Returns articulation root path."""
        import omni.kit.commands

        preprocess_urdf(self._urdf_path, _PROCESSED_URDF)

        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.distance_scale = 1.0
        import_config.default_position_drive_damping = 1e3

        _, robot_prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=_PROCESSED_URDF,
            import_config=import_config,
            dest_path="",
            get_articulation_root=True,
        )
        if robot_prim_path is None:
            raise RuntimeError(f"URDF import failed: {self._urdf_path} — check Isaac Sim logs")
        return robot_prim_path

    def _place_robot(self, prim_path: str, position: tuple[float, float, float]) -> None:
        """Translate the robot base prim before world.reset() anchors the fixed joint.

        The URDF importer creates ``/p3020/root_joint`` as the articulation root.
        The parent prim ``/p3020`` is what carries the world-space transform.
        """
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        robot_base = prim_path.rsplit("/", 1)[0]  # "/p3020/root_joint" → "/p3020"
        prim = stage.GetPrimAtPath(robot_base)
        if prim.IsValid():
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*position))
