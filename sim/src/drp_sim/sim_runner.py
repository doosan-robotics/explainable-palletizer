"""Thread-safe command queue bridging async FastAPI handlers to the sim main thread.

The SimRunner owns the simulation loop on the main thread. Async API handlers
call ``send_command()`` from the uvicorn thread, which enqueues a ``SimMessage``
and returns a ``concurrent.futures.Future``. The main-thread loop dequeues
messages and sets results on their futures, which the async handlers await via
``asyncio.wrap_future()``.

Architecture::

    [uvicorn thread]                          [main thread]
    FastAPI handler                           SimRunner.run()
        |                                         |
        +-- send_command(cmd, payload) ----------->|
        |       returns Future                     |
        |                                    _process_one(msg)
        |                                          |
        |<---- future.set_result(result) ----------+
        |
    await asyncio.wrap_future(future)

    WebSocket /sim/camera/stream  <----  frame_buffer  <----  main loop
        (reads latest JPEG)              (thread-safe)       (captures when subscribers > 0)
"""

from __future__ import annotations

import base64
import contextlib
import enum
import logging
import os
import queue
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from drp_sim.frame_buffer import FrameBuffer
from drp_sim.robot import HOME_JOINTS

logger = logging.getLogger(__name__)

_PHYSICS_HZ = 60
_CAMERA_WIDTH = int(os.environ.get("SIM_CAMERA_WIDTH", "960"))
_CAMERA_HEIGHT = int(os.environ.get("SIM_CAMERA_HEIGHT", "540"))
_JPEG_QUALITY = int(os.environ.get("SIM_JPEG_QUALITY", "80"))


# Conveyor pickup zone (world space, mixed_palletizing_scene.usd)
# Slots 1 and 2 are both reachable pickup positions (Y=-0.50 and Y=0.10).
# Search centre is the midpoint; tolerance covers both slots plus overshoot margin.
_PICKUP_X: float = 0.59
_PICKUP_Y: float = 0.40  # midpoint of slot 1 (0.10) and slot 2 (0.70)
_PICKUP_TOL: float = 0.40  # covers Y=0.0 to Y=0.80 — both pickup slots
_BOX_CENTER_Z: float = -0.206  # box centre z on conveyor

# Pallet slots and placement height
_PALLET_SLOTS: list[tuple[float, float]] = [
    (-0.657, -0.471),  # pallet 1 (/World/pallet_with_dollly_03)
    (-0.647, 0.819),  # pallet 2 (/World/pallet_with_dollly_02)
]
_SLOT_Z_C: float = -0.172  # pallet surface z (bbox top)
_SWING_VIA_XY_NEG: tuple[float, float] = (0.0, -0.40)  # swing toward pallet 1 (y<0)
_SWING_VIA_XY_POS: tuple[float, float] = (0.0, 0.40)  # swing toward pallet 2 (y>0)


# Custom cameras: created after env.reset() with explicit position/rotation/focal.
_CUSTOM_CAMERA_DEFS: dict[str, dict] = {
    "front": {
        "path": "/World/Cam_Front",
        "pos": (0.6, -2.5, 0.2),
        "rot": (90.0, 0.0, 0.0),
        "focal": 24.0,
    },
    "top": {
        "path": "/World/Cam_Top",
        "pos": (0.0, 1.2, 6.0),
        "rot": (0.0, 0.0, 0.0),
        "focal": 20.0,
    },
}
# Built-in cameras: already exist in the USD stage, only attach an annotator.
_BUILTIN_CAMERA_DEFS: dict[str, str] = {
    "persp": "/OmniverseKit_Persp",
}


class SimCommand(enum.Enum):
    PLAY = "play"
    PAUSE = "pause"
    RESET = "reset"
    STEP = "step"
    GET_STATE = "get_state"
    SET_JOINTS = "set_joints"
    SPAWN_BOX = "spawn_box"
    GET_BOX_IMAGES = "get_box_images"
    GO_HOME = "go_home"
    GET_CAMERA = "get_camera"
    MOVE_PLANNED = "move_planned"
    MOVE_CARTESIAN = "move_cartesian"
    FILL_BUFFER = "fill_buffer"
    CLEAR_CONVEYOR = "clear_conveyor"
    CLEAR_PALLET = "clear_pallet"
    PICK_PLACE = "pick_place"
    AUTO_PICK = "auto_pick"
    HUMAN_CALL = "human_call"
    REMOVE_BOX = "remove_box"
    GET_BUFFER_STATUS = "get_buffer_status"
    SHUTDOWN = "shutdown"


@dataclass
class SimMessage:
    command: SimCommand
    payload: dict[str, Any] = field(default_factory=dict)
    future: Future = field(default_factory=Future)


class SimRunner:
    """Main-thread simulation loop with a thread-safe command queue.

    Parameters
    ----------
    load_robot:
        Whether to import the P3020 URDF into the environment.
    spawn_boxes:
        Whether to enable the conveyor box spawner.
    """

    def __init__(self, *, load_robot: bool = True, spawn_boxes: bool = True) -> None:
        self._cmd_queue: queue.Queue[SimMessage] = queue.Queue()
        self._load_robot = load_robot
        self._spawn_boxes = spawn_boxes
        self._playing = False
        self._env = None
        self._motion_interface = None
        self._step_count = 0
        self._shutdown = False
        # Placed boxes to pin every step: list of (box_rigid, [x, y, z])
        self._placed_boxes: list[tuple[object, list[float]]] = []
        # Per-view frame buffers and annotators
        self._view_buffers: dict[str, FrameBuffer] = {
            "front": FrameBuffer(),
            "top": FrameBuffer(),
            "persp": FrameBuffer(),
        }
        self._view_annotators: dict[str, object] = {}
        # Legacy single buffer (for /sim/camera/stream endpoint)
        self.frame_buffer = self._view_buffers["front"]

    def send_command(self, command: SimCommand, payload: dict[str, Any] | None = None) -> Future:
        """Enqueue a command from any thread. Returns a Future for the result."""
        msg = SimMessage(command=command, payload=payload or {})
        self._cmd_queue.put(msg)
        return msg.future

    def get_frame_buffer(self, view: str) -> FrameBuffer:
        """Return the FrameBuffer for *view*, falling back to 'front'."""
        return self._view_buffers.get(view, self._view_buffers["front"])

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def sim_time(self) -> float:
        return self._step_count / _PHYSICS_HZ

    def run(self, sim_app: Any) -> None:
        """Main-thread blocking loop. Creates PalletizerEnv and processes the queue.

        Parameters
        ----------
        sim_app:
            The already-created ``SimulationApp`` instance.
        """
        from drp_sim.env import PalletizerEnv

        # Always use load_robot=False: the USD scene already contains the
        # p3020 articulation at /p3020/root_joint.  Importing the URDF on
        # top creates a duplicate that splits visual mesh (static) from
        # physics (moving), so the camera shows a frozen robot.
        logger.info("SimRunner: creating PalletizerEnv (load_robot=False, scene robot)")
        self._env = PalletizerEnv.from_app(
            sim_app,
            load_robot=False,
            spawn_boxes=self._spawn_boxes,
        )

        self._env.reset()

        # Wrap the existing scene robot (same approach as cli.py)
        if self._load_robot:
            self._wrap_scene_robot()

        self._init_cameras()
        self._warm_render(sim_app)
        self._attach_gripper_geometry()
        self._init_motion()
        self._teleport_to_home()
        logger.info("SimRunner: environment ready, entering main loop")

        while not self._shutdown:
            self._drain_queue()

            if self._playing and self._env is not None:
                self._env.step(render=True)
                self._step_count += 1
                self._pin_placed_boxes()

            sim_app.update()

            if any(buf.active for buf in self._view_buffers.values()):
                self._capture_frames()

        logger.info("SimRunner: shutting down")
        if self._env is not None:
            self._env.close()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _init_cameras(self) -> None:
        """Create custom USD cameras and attach RGB annotators for all views.

        Must be called after ``env.reset()`` so the USD stage is loaded.
        Custom cameras (front, top) are created from scratch; the built-in
        ``/OmniverseKit_Persp`` camera is reused as-is for the persp view.
        """
        try:
            import omni.replicator.core as rep
            import omni.usd
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()

            # Ensure RTX data window NDC settings exist for headless mode.
            import carb

            settings = carb.settings.get_settings()
            for i, default in enumerate([0.0, 0.0, 1.0, 1.0]):
                if settings.get(f"/rtx/dataWindowNDC/{i}") is None:
                    settings.set(f"/rtx/dataWindowNDC/{i}", default)

            # Create custom cameras
            for view, cam_def in _CUSTOM_CAMERA_DEFS.items():
                camera_path = cam_def["path"]
                if not stage.GetPrimAtPath(camera_path).IsValid():
                    cam = UsdGeom.Camera.Define(stage, camera_path)
                    cam.CreateProjectionAttr("perspective")
                    cam.CreateFocalLengthAttr(cam_def["focal"])
                    cam.CreateHorizontalApertureAttr(36.0)
                    xf = UsdGeom.Xformable(cam.GetPrim())
                    xf.AddTranslateOp().Set(Gf.Vec3d(*cam_def["pos"]))
                    xf.AddRotateXYZOp().Set(Gf.Vec3f(*cam_def["rot"]))

                rp = rep.create.render_product(camera_path, (_CAMERA_WIDTH, _CAMERA_HEIGHT))
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach([rp])
                self._view_annotators[view] = annotator
                logger.info("SimRunner: camera '%s' at %s (custom)", view, camera_path)

            # Attach annotators to built-in cameras
            for view, camera_path in _BUILTIN_CAMERA_DEFS.items():
                if not stage.GetPrimAtPath(camera_path).IsValid():
                    logger.warning("SimRunner: built-in camera %s not found", camera_path)
                    continue
                rp = rep.create.render_product(camera_path, (_CAMERA_WIDTH, _CAMERA_HEIGHT))
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach([rp])
                self._view_annotators[view] = annotator
                logger.info("SimRunner: camera '%s' at %s (built-in)", view, camera_path)

            n = len(self._view_annotators)
            logger.info(
                "SimRunner: %d cameras initialized (%dx%d)",
                n,
                _CAMERA_WIDTH,
                _CAMERA_HEIGHT,
            )
        except Exception:
            logger.warning("SimRunner: camera init failed, streaming disabled", exc_info=True)

    def _wrap_scene_robot(self) -> None:
        """Wrap the scene's existing /p3020/root_joint articulation.

        The USD scene already contains the p3020 robot with visual meshes
        and an articulation root.  We wrap it with Isaac Sim's Robot class
        so that joint commands move the visual mesh (not a hidden URDF copy).
        This mirrors the approach in cli.py ``_run_pick_mode``.
        """
        from isaacsim.core.api.robots import Robot

        from drp_sim._constants import _PROCESSED_URDF, _URDF_PATH, preprocess_urdf

        # Preprocess URDF (needed for cuRobo config, not for import)
        preprocess_urdf(_URDF_PATH, _PROCESSED_URDF)

        prim_path = "/p3020/root_joint"
        robot = self._env._world.scene.add(Robot(prim_path=prim_path, name="p3020"))
        self._env._world.reset()
        self._env._robot = robot
        self._env._prim_path = prim_path
        logger.info("SimRunner: wrapped existing scene robot at %s", prim_path)

    def _warm_render(self, sim_app: Any, frames: int = 5) -> None:
        """Pump a few render frames so annotator viewports are initialised.

        Without this, ``get_data()`` on newly-attached annotators raises
        because the replicator overscan parameters are still None.
        """
        if self._env is None or self._env._world is None:
            return
        for _ in range(frames):
            self._env._world.step(render=True)
            sim_app.update()
        logger.info("SimRunner: render warm-up complete (%d frames)", frames)

    def _attach_gripper_geometry(self) -> None:
        """Create visual-only VGC10 vacuum gripper geometry under link_6."""
        prim_path = self._env.prim_path if self._env else None
        if prim_path is None:
            return
        attach_vgc10_gripper(prim_path)

    def _init_motion(self) -> None:
        """Initialise cuRobo MotionInterface. No-op if cuRobo is absent or robot not loaded."""
        if not self._load_robot:
            return
        if self._env is None:
            return
        try:
            from drp_sim.motion_interface import MotionInterface

            logger.info("SimRunner: warming up cuRobo MotionInterface...")
            self._motion_interface = MotionInterface(self._env)
            logger.info("SimRunner: cuRobo ready")
        except Exception:
            logger.warning(
                "SimRunner: cuRobo unavailable, motion planning disabled",
                exc_info=True,
            )
            self._motion_interface = None

    def _teleport_to_home(self) -> None:
        """Instantly set robot joints to HOME_JOINTS via physics teleport.

        Uses set_joint_positions (instant teleport) instead of apply_action
        (drive target) so the robot appears at home pose immediately.
        Requires a few warm-up physics steps to have been run first.
        """
        if self._env is None or self._env._robot is None:
            return
        import numpy as np

        from drp_sim._constants import N_JOINTS

        n = self._env._robot.num_dof
        full = np.zeros(n)
        full[:N_JOINTS] = HOME_JOINTS[:N_JOINTS]
        self._env._robot.set_joint_positions(full)
        for _ in range(10):
            self._env._world.step(render=True)

    def _capture_frames(self) -> None:
        """Capture rendered frames for all views that have active subscribers."""
        if not self._view_annotators:
            return
        try:
            import cv2
            import numpy as np

            for view, annotator in self._view_annotators.items():
                buf = self._view_buffers[view]
                if not buf.active:
                    continue
                data = annotator.get_data()
                if data is None or not isinstance(data, np.ndarray) or data.size == 0:
                    continue
                if data.ndim < 3:
                    continue
                rgb = data[:, :, :3]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
                buf.put(jpeg.tobytes(), bgr.shape[1], bgr.shape[0])
        except Exception:
            if self._step_count % 300 == 0:
                logger.warning("SimRunner: frame capture failed", exc_info=True)

    def _snapshot(self) -> dict[str, Any]:
        """Return a single frame as base64 JPEG for the REST endpoint."""
        frame, w, h, _ = self.frame_buffer.get()
        if not frame:
            # Force a capture even if no subscribers
            self._capture_frames()
            frame, w, h, _ = self.frame_buffer.get()
        if not frame:
            return {"width": 0, "height": 0, "data": "", "encoding": "none"}
        return {
            "width": w,
            "height": h,
            "data": base64.b64encode(frame).decode("ascii"),
            "encoding": "jpeg/base64",
        }

    # ------------------------------------------------------------------
    # Queue processing
    # ------------------------------------------------------------------

    def _drain_queue(self) -> None:
        """Process all pending messages without blocking."""
        while True:
            try:
                msg = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            self._process_one(msg)

    def _process_one(self, msg: SimMessage) -> None:
        """Execute a single command and set the future result."""
        try:
            result = self._dispatch(msg.command, msg.payload)
            msg.future.set_result(result)
        except Exception as exc:
            logger.exception("Command %s raised (payload keys: %s)", msg.command, list(msg.payload))
            msg.future.set_exception(exc)

    def _dispatch(self, cmd: SimCommand, payload: dict[str, Any]) -> Any:
        if cmd == SimCommand.PLAY:
            if self._spawn_boxes and self._env is not None:
                self._env.fill_buffer()
            self._playing = True
            return {"status": "playing"}

        if cmd == SimCommand.PAUSE:
            self._playing = False
            return {"status": "paused"}

        if cmd == SimCommand.RESET:
            if self._env is not None:
                self._env.reset()
                self._init_cameras()
                self._attach_gripper_geometry()
                self._teleport_to_home()
            self._step_count = 0
            self._playing = False
            self._placed_boxes.clear()
            return {"status": "reset"}

        if cmd == SimCommand.STEP:
            if self._env is None:
                return {"error": "env not initialized"}
            n = payload.get("num_steps", 1)
            render = payload.get("render", True)
            for _ in range(n):
                self._env.step(render=render)
                self._step_count += 1
            return {"steps": n, "sim_time": self.sim_time}

        if cmd == SimCommand.GET_STATE:
            return self._get_state()

        if cmd == SimCommand.SET_JOINTS:
            positions = payload["joint_positions"]
            if self._env is not None:
                self._env.set_joint_positions(positions)
            return {"status": "ok"}

        if cmd == SimCommand.SPAWN_BOX:
            if self._env is None:
                return {"error": "env not initialized"}
            prim_path = self._env.spawn_box()
            return {"prim_path": prim_path, "box_count": len(self._env.boxes)}

        if cmd == SimCommand.GET_BOX_IMAGES:
            if self._env is None or self._env.image_capture is None:
                return {"images": []}
            return {"images": self._env.image_capture.drain()}

        if cmd == SimCommand.GO_HOME:
            if self._env is not None:
                self._env.set_joint_positions(list(HOME_JOINTS))
            return {"status": "ok"}

        if cmd == SimCommand.GET_CAMERA:
            return self._snapshot()

        if cmd == SimCommand.SHUTDOWN:
            self._shutdown = True
            return {"status": "shutting_down"}

        if cmd == SimCommand.MOVE_PLANNED:
            if self._motion_interface is None:
                return {"error": "Motion planning unavailable (cuRobo not loaded)"}
            target = payload["target"]
            execute = payload.get("execute", True)
            try:
                trajectory = self._motion_interface.move_to_joints(target, execute=execute)
            except RuntimeError as exc:
                return {"error": str(exc)}
            return {"trajectory": trajectory}

        if cmd == SimCommand.FILL_BUFFER:
            if self._env is None:
                return {"error": "env not initialized"}
            return self._env.fill_buffer()

        if cmd == SimCommand.HUMAN_CALL:
            if self._env is None:
                return {"error": "env not initialized"}
            index = payload.get("index", 0)
            result = self._env.remove_buffer_box(index)
            if result is None:
                return {"status": "empty", "slot": index}
            path, _ = result
            fill = self._env.fill_buffer()
            return {"status": "ok", "slot": index, "prim_path": path, **fill}

        if cmd == SimCommand.MOVE_CARTESIAN:
            if self._motion_interface is None:
                return {"error": "Motion planning unavailable (cuRobo not loaded)"}
            position = payload["position"]
            quaternion = payload["quaternion"]
            execute = payload.get("execute", True)
            try:
                trajectory = self._motion_interface.move_to_pose(
                    position, quaternion, execute=execute, orientation_constraint=True
                )
            except RuntimeError as exc:
                return {"error": str(exc)}
            return {"trajectory": trajectory}

        if cmd == SimCommand.CLEAR_CONVEYOR:
            return self._clear_boxes_zone(y_min=0.0, y_max=3.5)

        if cmd == SimCommand.CLEAR_PALLET:
            return self._clear_boxes_zone(y_min=-2.0, y_max=0.0)

        if cmd == SimCommand.PICK_PLACE:
            return self._handle_pick_and_place(payload)

        if cmd == SimCommand.AUTO_PICK:
            return self._auto_pick(payload)

        if cmd == SimCommand.REMOVE_BOX:
            return self._remove_box(payload)

        if cmd == SimCommand.GET_BUFFER_STATUS:
            buffer = getattr(self._env, "_buffer", None) if self._env else None
            if buffer is None:
                return {"occupied": 0, "capacity": 0, "slots": [], "in_transit": False}
            return {
                "occupied": buffer.occupied_count,
                "capacity": buffer.slot_count,
                "slots": buffer.slot_states,
                "in_transit": buffer._in_transit is not None,
            }

        return {"error": f"unknown command: {cmd}"}  # pragma: no cover

    def _get_state(self) -> dict[str, Any]:
        """Gather current sim state."""
        joint_positions: list[float] = []
        box_positions: list[dict[str, float]] = []

        if self._env is not None:
            with contextlib.suppress(RuntimeError):
                joint_positions = self._env.get_joint_positions()

            for box in self._env.boxes:
                try:
                    pos, _ = box.get_world_poses()
                    box_positions.append(
                        {
                            "x": float(pos[0, 0]),
                            "y": float(pos[0, 1]),
                            "z": float(pos[0, 2]),
                            "prim_path": box.prim_paths[0],
                        }
                    )
                except Exception:
                    pass

        return {
            "joint_positions": joint_positions,
            "box_positions": box_positions,
            "sim_time": self.sim_time,
        }

    def _clear_boxes_zone(self, y_min: float, y_max: float) -> dict[str, Any]:
        """Remove boxes whose world-Y is in [y_min, y_max] from the USD stage."""
        if self._env is None:
            return {"removed": 0}
        import omni.usd

        spawner = getattr(self._env, "_spawner", None)
        if spawner is None:
            return {"removed": 0}

        stage = omni.usd.get_context().get_stage()
        removed: set[str] = set()

        for box_prim, path, _step in list(spawner._boxes):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                removed.add(path)
                continue
            try:
                pos, _ = box_prim.get_world_poses()
                y = float(pos[0, 1])
            except Exception:
                continue
            if y_min <= y <= y_max:
                stage.RemovePrim(path)
                removed.add(path)

        if removed:
            spawner._boxes = [(b, p, s) for b, p, s in spawner._boxes if p not in removed]

        logger.info("Cleared %d boxes in Y=[%.1f, %.1f]", len(removed), y_min, y_max)
        return {"removed": len(removed)}

    def _remove_box(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Remove a box by popping the nearest buffer slot.

        The logical ``box_id`` (e.g. ``box_0001``) from the app has no direct
        mapping to the sim prim path.  We pop the first occupied buffer slot.
        """
        if self._env is None:
            return {"error": "env not initialized"}

        box_id: str = payload["box_id"]
        buffer = getattr(self._env, "_buffer", None)

        if buffer is not None:
            # Pop the first occupied slot and release to pool
            for i in range(buffer.slot_count):
                result = buffer.pop_box_at(i)
                if result is not None:
                    box_prim, prim_path = result
                    buffer.release_box(box_prim, prim_path=prim_path)
                    logger.info("Removed box %s (buffer slot %d -> %s)", box_id, i, prim_path)
                    buffer.fill()
                    return {"status": "ok", "box_id": box_id}

        logger.warning("No box found in buffer to remove for %s", box_id)
        return {"status": "not_found", "box_id": box_id}

    def _find_box_near(self, x: float, y: float, tol: float = 0.15) -> tuple[object, str] | None:
        """Return ``(box_prim, prim_path)`` for the first spawner box near (x, y)."""
        spawner = getattr(self._env, "_spawner", None) if self._env else None
        if spawner is None:
            return None
        for box_prim, path, _step in spawner._boxes:
            try:
                pos, _ = box_prim.get_world_poses()
                if abs(float(pos[0, 0]) - x) < tol and abs(float(pos[0, 1]) - y) < tol:
                    return (box_prim, path)
            except Exception:
                continue
        return None

    def _auto_pick(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Find a box in the pickup zone and place it on the requested pallet slot.

        Parameters
        ----------
        payload:
            ``{"slot": int}`` — pallet slot number 1-3.
        """
        if self._motion_interface is None:
            return {"status": "error", "message": "motion planning unavailable"}

        slot: int = int(payload.get("slot", 0))
        if slot not in (1, 2, 3):
            return {"status": "error", "message": f"invalid slot {slot!r}: must be 1, 2, or 3"}

        found = self._find_box_near(_PICKUP_X, _PICKUP_Y, tol=_PICKUP_TOL)
        if found is None:
            return {"status": "error", "message": "no box in pickup zone"}

        _box_prim, prim_path = found
        slot_x, slot_y = _PALLET_SLOTS[slot - 1]
        logger.info("auto_pick: slot %d (%.3f, %.3f)", slot, slot_x, slot_y)

        # Use unified pick_place handler with fixed pickup zone position
        result = self._handle_pick_and_place(
            {
                "box_prim": prim_path,
                "pick_position": [_PICKUP_X, _PICKUP_Y, _BOX_CENTER_Z],
                "drop_position": [slot_x, slot_y, _SLOT_Z_C],
            }
        )

        if result.get("status") == "ok":
            result["slot"] = slot
            result["success"] = True

        return result

    # ------------------------------------------------------------------
    # Pick-and-place orchestration
    # ------------------------------------------------------------------

    def _handle_pick_and_place(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Orchestrate a full pick-and-place motion sequence.

        Steps:
          1. Move above the pick position  (pin box at pick pos)
          2. Descend to pick               (pin box at pick pos)
          3. Attach: snap box to EE
          4. Lift straight up              (track box to EE)
          5. Swing via intermediate point   (avoid collisions)
          6. Move to high above pallet     (track box to EE)
          7. Carry to above drop position  (track box to EE)
          8. Descend to place              (track box to EE)
          9. Detach: snap box to final resting position

        Physics callbacks prevent the robot from pushing the box during
        approach and keep the box attached to the EE during carry.

        The ``drop_quaternion`` is a *logical* rotation for the box (identity
        or 90-deg Z).  It is composed with the gripper-down base orientation
        ``_EE_QUAT_DOWN`` so cuRobo plans the correct end-effector pose.
        """
        logger.info(
            "pick_and_place START: pick=%s drop=%s quat=%s",
            payload.get("pick_position"),
            payload.get("drop_position"),
            payload.get("drop_quaternion"),
        )
        if self._motion_interface is None:
            logger.error("pick_and_place: motion planning unavailable")
            return {"status": "error", "message": "motion planning unavailable"}
        if self._env is None:
            logger.error("pick_and_place: env not initialized")
            return {"status": "error", "message": "env not initialized"}

        import numpy as np

        pick_pos = list(payload["pick_position"])
        drop_pos = payload["drop_position"]
        drop_quat_logical: list[float] = payload.get("drop_quaternion", [1.0, 0.0, 0.0, 0.0])

        # Compose logical drop rotation with gripper-down base orientation.
        # EE base: 180-deg around X = [0, 1, 0, 0] (suction cup faces down).
        # Final EE quat = drop_quat_logical * ee_base (Hamilton product).
        ee_quat = _quat_multiply(drop_quat_logical, _EE_QUAT_DOWN)

        # Geometry constants (must match pick_loop.py / palletizing_demo.py)
        vgc10_len = 0.23
        above_clearance = 0.05
        lift_clearance = 0.15

        # --- Resolve the actual box prim from the buffer or by spatial search.
        # The app sends a logical box_id (e.g. "box_0003") which does NOT map
        # to the actual USD prim path (e.g. "/World/pool_box_0").  We must
        # find the box physically present at the pick position.
        buffer = getattr(self._env, "_buffer", None)
        box_prim_path: str | None = None
        box_rigid = None  # reuse the buffer's RigidPrim (batch API)

        if buffer is not None:
            # Find the nearest occupied buffer slot to the requested pick Y
            best_idx: int | None = None
            best_dist = float("inf")
            best_half_h = 0.0
            for i, slot in enumerate(buffer._slots):
                if slot is None:
                    continue
                dist = abs(slot.assigned_position[1] - pick_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
                    best_half_h = slot.box_half_h

            if best_idx is not None:
                result = buffer.pop_box_at(best_idx)
                if result is not None:
                    box_rigid, box_prim_path = result
                    # Adjust pick Z to the TOP of this box so gripper contacts
                    # the surface (different box heights).
                    slot_pos = buffer._compute_slot_position(best_idx)
                    pick_pos[2] = slot_pos[2] + best_half_h
                    # Use the actual buffer slot XY as pick position
                    pick_pos[0] = slot_pos[0]
                    pick_pos[1] = slot_pos[1]
                    logger.info(
                        "Resolved box from buffer slot %d: %s (half_h=%.3f, pick_z=%.3f)",
                        best_idx,
                        box_prim_path,
                        best_half_h,
                        pick_pos[2],
                    )

        # Fallback: search spawner boxes near the pick position
        if box_prim_path is None:
            found = self._find_box_near(pick_pos[0], pick_pos[1], tol=0.5)
            if found is not None:
                box_rigid, box_prim_path = found
                logger.info("Resolved box by spatial search: %s", box_prim_path)

        if box_prim_path is None or box_rigid is None:
            logger.error("pick_and_place: no box found near pick position %s", pick_pos)
            return {"status": "error", "message": f"no box found near pick position {pick_pos}"}

        def _snap(pos: tuple[float, ...] | list[float]) -> None:
            box_rigid.set_world_poses(positions=np.array([pos]))
            box_rigid.set_linear_velocities(np.array([[0.0, 0.0, 0.0]]))
            box_rigid.set_angular_velocities(np.array([[0.0, 0.0, 0.0]]))

        def _pre_pin() -> None:
            """Run BEFORE world.step(): pin buffer boxes + target box at pick pos."""
            if buffer is not None:
                buffer._enforce_slot_positions()
            _snap(pick_pos)

        def _pre_track() -> None:
            """Run BEFORE world.step(): pin buffer boxes + target box at EE."""
            if buffer is not None:
                buffer._enforce_slot_positions()
            ex, ey, ez = self._motion_interface.get_ee_position()
            _snap((ex, ey, ez - vgc10_len))

        def _post_pin() -> None:
            """Run AFTER world.step(): re-snap target box at pick pos."""
            _snap(pick_pos)

        def _post_track() -> None:
            """Run AFTER world.step(): re-snap target box to EE."""
            ex, ey, ez = self._motion_interface.get_ee_position()
            _snap((ex, ey, ez - vgc10_len))

        steps: list[str] = []

        # Initial snap: ensure box is exactly at pick position before any
        # robot motion begins (mirrors cli.py line 160).
        _snap(pick_pos)

        try:
            # 1. Approach above pick position (pin box before+after physics)
            # No orientation_constraint here: robot transitions from home
            # orientation to _EE_QUAT_DOWN freely. Constraint is applied from
            # step 2 onward, once EE has already reached gripper-down pose.
            above_pick = [pick_pos[0], pick_pos[1], pick_pos[2] + vgc10_len + above_clearance]
            logger.info("pick_and_place [1/9] above_pick: %s", above_pick)
            self._motion_interface.move_to_pose(
                above_pick,
                list(_EE_QUAT_DOWN),
                pre_step_callback=_pre_pin,
                step_callback=_post_pin,
            )
            steps.append("above_pick")

            # 2. Descend to pick (keep pinning before+after physics)
            pick_ee = [pick_pos[0], pick_pos[1], pick_pos[2] + vgc10_len]
            logger.info("pick_and_place [2/9] pick: %s", pick_ee)
            self._motion_interface.move_to_pose(
                pick_ee,
                list(_EE_QUAT_DOWN),
                pre_step_callback=_pre_pin,
                step_callback=_post_pin,
            )
            steps.append("pick")

            # 3. Attach: snap box to current EE position
            _post_track()
            steps.append("attach")
            logger.info("pick_and_place [3/9] attach")

            # 4. Lift straight up (box follows EE before+after physics)
            lift_z = above_pick[2] + lift_clearance
            lift = [pick_pos[0], pick_pos[1], lift_z]
            logger.info("pick_and_place [4/9] lift: %s", lift)
            self._motion_interface.move_to_pose(
                lift,
                list(_EE_QUAT_DOWN),
                pre_step_callback=_pre_track,
                step_callback=_post_track,
            )
            steps.append("lift")

            # 5. Swing via intermediate waypoint (high enough to clear pallet boxes)
            #    Pick swing side based on drop Y: negative Y -> pallet 1, positive Y -> pallet 2
            swing_xy = _SWING_VIA_XY_POS if drop_pos[1] > 0 else _SWING_VIA_XY_NEG
            swing_z = max(lift_z, 0.60)
            swing_via = [swing_xy[0], swing_xy[1], swing_z]
            logger.info("pick_and_place [5/9] swing_via: %s", swing_via)
            self._motion_interface.move_to_pose(
                swing_via,
                list(_EE_QUAT_DOWN),
                pre_step_callback=_pre_track,
                step_callback=_post_track,
            )
            steps.append("swing_via")

            # 6. Move to high above pallet
            pallet_high = [drop_pos[0], drop_pos[1], 0.70]
            logger.info("pick_and_place [6/9] pallet_high: %s", pallet_high)
            self._motion_interface.move_to_pose(
                pallet_high,
                ee_quat,
                pre_step_callback=_pre_track,
                step_callback=_post_track,
            )
            steps.append("pallet_high")

            # 7. Carry to above drop position (box follows EE)
            above_drop = [drop_pos[0], drop_pos[1], drop_pos[2] + vgc10_len + above_clearance]
            logger.info("pick_and_place [7/9] above_drop: %s", above_drop)
            self._motion_interface.move_to_pose(
                above_drop,
                ee_quat,
                pre_step_callback=_pre_track,
                step_callback=_post_track,
            )
            steps.append("above_drop")

            # 8. Descend to place (box follows EE)
            place_ee = [drop_pos[0], drop_pos[1], drop_pos[2] + vgc10_len]
            logger.info("pick_and_place [8/9] place: %s", place_ee)
            self._motion_interface.move_to_pose(
                place_ee,
                ee_quat,
                pre_step_callback=_pre_track,
                step_callback=_post_track,
            )
            steps.append("place")

            # 9. Detach: snap box to final resting position with correct orientation
            self._detach_box(box_prim_path, drop_pos, drop_quat_logical)
            _snap(drop_pos)
            steps.append("detach")
            logger.info("pick_and_place [9/9] detach at %s", drop_pos)

        except RuntimeError as exc:
            logger.error("pick_and_place FAILED at step %s: %s", steps, exc)
            # Refill buffer even on failure so new boxes spawn to replace the
            # popped box that was orphaned by the failed sequence.
            try:
                if buffer is not None:
                    buffer.fill()
            except Exception:
                pass
            return {"status": "error", "message": str(exc), "steps": steps}

        # Track the placed box so the main loop keeps it pinned at drop_pos.
        self._placed_boxes.append((box_rigid, list(drop_pos)))

        # Refill buffer after picking. The box prim stays at drop_pos (no pool).
        try:
            if buffer is not None:
                buffer.fill()
                logger.info("pick_and_place: buffer refill triggered")
        except Exception:
            logger.warning("Post-placement cleanup failed", exc_info=True)

        logger.info("pick_and_place SUCCESS: steps=%s box=%s", steps, box_prim_path)
        return {"status": "ok", "steps": steps, "success": True}

    def _attach_box(self, box_prim_path: str, vgc10_len: float) -> None:
        """Snap a box USD prim to the current EE position (teleport attach)."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        # Get link_6 world position
        robot_prim = self._env.prim_path
        if robot_prim is None:
            return
        base = robot_prim.rsplit("/", 1)[0]
        ee_pos = None
        for cand in (f"{robot_prim}/link_6", f"{base}/link_6"):
            prim = stage.GetPrimAtPath(cand)
            if prim.IsValid():
                t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
                ee_pos = (float(t[0]), float(t[1]), float(t[2]))
                break
        if ee_pos is None:
            return

        box = stage.GetPrimAtPath(box_prim_path)
        if not box.IsValid():
            return
        xf = UsdGeom.Xformable(box)
        for op in xf.GetOrderedXformOps():
            if "translate" in op.GetOpName():
                op.Set(Gf.Vec3d(ee_pos[0], ee_pos[1], ee_pos[2] - vgc10_len))
                return

    def _detach_box(
        self,
        box_prim_path: str,
        position: list[float],
        logical_quat: list[float],
    ) -> None:
        """Snap box to final resting position, then freeze it kinematic."""
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        box = stage.GetPrimAtPath(box_prim_path)
        if not box.IsValid():
            return
        xf = UsdGeom.Xformable(box)
        for op in xf.GetOrderedXformOps():
            if "translate" in op.GetOpName():
                op.Set(Gf.Vec3d(*position))
            if "orient" in op.GetOpName():
                # Use the precision that matches the attribute type (GfQuatd
                # for double, GfQuatf for float).  Box spawner creates orient
                # ops as GfQuatd; writing GfQuatf causes a USD type mismatch.
                attr = op.GetAttr()
                type_name = attr.GetTypeName().type.typeName if attr else ""
                w, x, y, z = logical_quat
                if "Quatf" in type_name:
                    op.Set(Gf.Quatf(w, x, y, z))
                else:
                    op.Set(Gf.Quatd(w, x, y, z))
        # Lock the box in place so physics cannot move it after placement.
        rb_api = UsdPhysics.RigidBodyAPI(box)
        if rb_api:
            rb_api.GetKinematicEnabledAttr().Set(True)

    def _pin_placed_boxes(self) -> None:
        """Pin all pallet-placed boxes at their drop positions every sim step.

        Pool prims use a Fabric tensor view; the USD-level kinematic flag set
        in _detach_box() may be overridden by the Fabric layer on the next
        physics tick.  Calling set_world_poses() every step is the reliable
        way to keep placed boxes frozen regardless of kinematic state.
        """
        import numpy as np

        dead: list[int] = []
        for i, (box_rigid, pos) in enumerate(self._placed_boxes):
            try:
                box_rigid.set_world_poses(positions=np.array([pos]))
            except Exception:
                dead.append(i)
        for i in reversed(dead):
            self._placed_boxes.pop(i)


# Gripper-down base orientation: 180-deg around X (suction cup faces down).
_EE_QUAT_DOWN: tuple[float, ...] = (0.0, 1.0, 0.0, 0.0)


def attach_vgc10_gripper(prim_path: str) -> None:
    """Create visual-only VGC10 vacuum gripper geometry under link_6.

    All prims are visual-only (no RigidBodyAPI / CollisionAPI) so they
    do not interfere with physics.  Geometry is defined in link_6 local
    space where +Z is the tool axis.

    Parameters
    ----------
    prim_path:
        Articulation root prim path, e.g. ``/p3020/root_joint``.
    """
    try:
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()
        base = prim_path.rsplit("/", 1)[0]

        link6_path = None
        for cand in (f"{prim_path}/link_6", f"{base}/link_6"):
            if stage.GetPrimAtPath(cand).IsValid():
                link6_path = cand
                break
        if link6_path is None:
            logger.warning("attach_vgc10_gripper: link_6 not found under %s", prim_path)
            return

        root_path = f"{link6_path}/VGC10"
        if stage.GetPrimAtPath(root_path).IsValid():
            return  # already attached

        UsdGeom.Xform.Define(stage, root_path)
        mat_root = f"{root_path}/Mats"
        UsdGeom.Xform.Define(stage, mat_root)

        def _mat(name: str, rgb: tuple[float, float, float], metallic: float = 0.1):
            p = f"{mat_root}/{name}"
            mat = UsdShade.Material.Define(stage, p)
            sh = UsdShade.Shader.Define(stage, f"{p}/PBR")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            return mat

        def _bind(prim, mat):
            UsdShade.MaterialBindingAPI.Apply(prim)
            UsdShade.MaterialBindingAPI(prim).Bind(mat)

        def _cyl(name: str, r: float, h: float, tz: float, mat):
            c = UsdGeom.Cylinder.Define(stage, f"{root_path}/{name}")
            c.CreateRadiusAttr(r)
            c.CreateHeightAttr(h)
            c.CreateAxisAttr("Z")
            UsdGeom.Xformable(c.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, tz))
            _bind(c.GetPrim(), mat)

        grey = _mat("Grey", (0.20, 0.20, 0.22), metallic=0.35)
        black = _mat("Black", (0.04, 0.04, 0.04), metallic=0.0)

        _cyl("Flange", r=0.053, h=0.012, tz=0.006, mat=grey)
        _cyl("Body", r=0.044, h=0.155, tz=0.0895, mat=grey)
        _cyl("Neck", r=0.030, h=0.028, tz=0.181, mat=grey)
        _cyl("CupHousing", r=0.065, h=0.022, tz=0.206, mat=grey)
        _cyl("Cup", r=0.060, h=0.016, tz=0.223, mat=black)
        logger.info("attach_vgc10_gripper: VGC10 attached at %s", link6_path)
    except Exception:
        logger.warning("attach_vgc10_gripper: failed", exc_info=True)


def _quat_multiply(q1: list[float], q2: tuple[float, ...] | list[float]) -> list[float]:
    """Hamilton product q1 * q2.  Quaternions are [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
