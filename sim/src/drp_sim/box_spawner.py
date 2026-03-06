"""Conveyor box spawner for the palletizer simulation.

Spawns box USD prims onto the conveyor with an initial linear velocity
to simulate belt movement without physics-based conveyor simulation, modeled
after the NVIDIA UR10 palletizing example's BinStackingTask pattern.

Each spawn first samples a box *type* (normal, fragile, heavy, damaged)
according to configurable weights, then assembles a random variant from
the per-type config (USD model, x/y/z size choices, sticker probability).
The native bounding box of each USD asset is computed once via
``UsdGeom.BBoxCache`` and the scale factor is derived as
``desired / native`` so the loaded box matches the requested real-world
dimensions exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim

    from drp_sim.sticker_attacher import StickerAttacher

_USD_DIR = Path(__file__).parent.parent.parent / "usd"
_ASSETS_DIR = _USD_DIR / "assets" / "boxes"


@dataclass(frozen=True, slots=True)
class BoxTypeConfig:
    """Per-type spawn configuration, similar to ML hyperparameter configs.

    Each field is a list of choices; one value is sampled uniformly at
    spawn time.  This lets you express a combinatorial space compactly
    without enumerating every (usd, x, y, z) tuple.
    """

    usd_paths: list[str]
    x_choices: list[float] = field(default_factory=lambda: [0.25])
    y_choices: list[float] = field(default_factory=lambda: [0.25])
    z_choices: list[float] = field(default_factory=lambda: [0.25])
    sticker_probability: float = 0.5
    visuals: list[str] = field(default_factory=list)


_DEFAULT_TYPE_CONFIGS: dict[str, BoxTypeConfig] = {
    "normal": BoxTypeConfig(
        usd_paths=[
            str(_ASSETS_DIR / "SM_CardBoxD_05.usd"),
            str(_ASSETS_DIR / "SM_CardBoxD_01.usd"),
        ],
        x_choices=[0.50, 0.25],
        y_choices=[0.50, 0.25],
        z_choices=[0.25],
        sticker_probability=1.0,
    ),
    "fragile": BoxTypeConfig(
        usd_paths=[
            str(_ASSETS_DIR / "SM_CardBoxD_05.usd"),
            str(_ASSETS_DIR / "SM_CardBoxD_01.usd"),
        ],
        x_choices=[0.25, 0.5],
        y_choices=[0.25, 0.5],
        z_choices=[0.25],
        sticker_probability=1.0,
    ),
    "heavy": BoxTypeConfig(
        usd_paths=[
            str(_ASSETS_DIR / "SM_CardBoxD_05.usd"),
            str(_ASSETS_DIR / "SM_CardBoxD_01.usd"),
        ],
        x_choices=[0.50],
        y_choices=[0.50],
        z_choices=[0.25, 0.5],
        sticker_probability=1.0,
    ),
    "damaged": BoxTypeConfig(
        usd_paths=[
            str(_ASSETS_DIR / "damaged" / "damaged_untaped.usd"),
            str(_ASSETS_DIR / "damaged" / "damaged_untaped2.usd"),
        ],
        x_choices=[0.50],
        y_choices=[0.25],
        z_choices=[0.25],
        sticker_probability=0.0,
    ),
}

_DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    "normal": 0.4,
    "fragile": 0.2,
    "heavy": 0.2,
    "damaged": 0.2,
}

_DEFAULT_SPAWN_POSITION = (0.59, 3.01, -0.20)
_DEFAULT_VELOCITY = (0.0, -0.3, 0.0)
_DEFAULT_SPAWN_INTERVAL = 10.0  # seconds

_COLOR_JITTER = 0.04


class BoxSpawner:
    """Spawns boxes on the conveyor belt with initial velocity.

    Each spawn samples a box type from ``type_weights``, assembles a
    random variant from ``type_configs``, and records metadata (size,
    weight, type).

    Parameters
    ----------
    world:
        isaacsim.core.api.World instance (must be initialised before use).
    type_configs:
        Per-type spawn configuration.  Each entry maps a type name to a
        ``BoxTypeConfig`` that defines USD choices, size choices per axis,
        and sticker probability.
    type_weights:
        Mapping of box type name to sampling weight.  Weights need not
        sum to 1.0 -- they are treated as relative probabilities.
    spawn_position:
        (x, y, z) world position where new boxes appear.
    velocity:
        (vx, vy, vz) initial linear velocity applied on spawn (m/s).
    spawn_interval:
        Time in seconds between automatic spawns during step().
    physics_dt:
        Simulation physics timestep in seconds.
    env_path:
        Parent prim path under which box prims are created.
    box_ttl:
        Time-to-live in seconds.  ``None`` to keep boxes indefinitely.
    """

    def __init__(
        self,
        world: World,
        type_configs: dict[str, BoxTypeConfig] | None = None,
        type_weights: dict[str, float] | None = None,
        spawn_position: tuple[float, float, float] = _DEFAULT_SPAWN_POSITION,
        velocity: tuple[float, float, float] = _DEFAULT_VELOCITY,
        spawn_interval: float = _DEFAULT_SPAWN_INTERVAL,
        physics_dt: float = 1.0 / 60.0,
        env_path: str = "/World",
        sticker_attacher: StickerAttacher | None = None,
        box_ttl: float | None = 15.0,
    ) -> None:
        self._world = world
        self._type_configs = type_configs if type_configs is not None else _DEFAULT_TYPE_CONFIGS
        self._type_weights = type_weights if type_weights is not None else _DEFAULT_TYPE_WEIGHTS
        self._spawn_position = np.array(spawn_position, dtype=float)
        self._velocity = np.array(velocity, dtype=float)
        self._spawn_interval_steps = max(1, round(spawn_interval / physics_dt))
        self._env_path = env_path
        self._sticker_attacher = sticker_attacher
        self._box_ttl_steps = max(1, round(box_ttl / physics_dt)) if box_ttl is not None else None
        self._boxes: list[tuple[RigidPrim, str, int]] = []
        self._box_count: int = 0
        self._step_count: int = 0
        self._box_metadata: list[dict[str, object]] = []
        self._auto_spawn_enabled: bool = True
        # usd_path -> (lo, hi) native bounding box, computed once per asset.
        self._bbox_cache: dict[
            str, tuple[tuple[float, float, float], tuple[float, float, float]]
        ] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def spawn(
        self,
        position: tuple[float, float, float] | None = None,
        velocity: tuple[float, float, float] | None = None,
    ) -> str:
        """Spawn a randomly chosen box variant and return its prim path.

        Parameters
        ----------
        position:
            Override spawn position. Uses the default conveyor start when ``None``.
        velocity:
            Override initial velocity. Uses the default conveyor velocity when ``None``.
        """
        import omni.usd
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, UsdGeom

        spawn_pos = (
            np.array(position, dtype=float) if position is not None else self._spawn_position
        )
        spawn_vel = np.array(velocity, dtype=float) if velocity is not None else self._velocity

        # 1. Determine box type: sequential from sticker attacher, or random
        next_type = self._sticker_attacher.peek_type() if self._sticker_attacher else None
        if next_type is not None and next_type not in self._type_configs:
            next_type = None
        box_type = next_type if next_type is not None else self._sample_type()
        config = self._type_configs[box_type]
        usd_idx = random.randrange(len(config.usd_paths))
        usd_path = config.usd_paths[usd_idx]
        box_size = (
            random.choice(config.x_choices),
            random.choice(config.y_choices),
            random.choice(config.z_choices),
        )

        scale, native_half, native_z_top = self._resolve_variant(usd_path, box_size)

        box_name = f"box_{self._box_count}"
        prim_path = f"{self._env_path}/{box_name}"

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)

        xformable = UsdGeom.Xformable(prim)
        xformable.AddScaleOp().Set(Gf.Vec3f(*scale))

        box_prim = RigidPrim(prim_paths_expr=prim_path, name=box_name)
        self._world.scene.add(box_prim)
        box_prim.initialize()

        orientation = np.array([(1.0, 0.0, 0.0, 0.0)], dtype=float)
        box_prim.set_world_poses(
            positions=spawn_pos[np.newaxis],
            orientations=orientation,
        )
        box_prim.set_linear_velocities(spawn_vel[np.newaxis])

        self._boxes.append((box_prim, prim_path, self._step_count))
        self._box_count += 1

        # 2. Attach sticker and determine weight
        weight: float | None = None
        visual: str | None = None
        box_half = (native_half[0], native_half[1], native_z_top)
        if self._sticker_attacher is not None:
            self._sticker_attacher._box_half = box_half
            if next_type is not None:
                result = self._sticker_attacher.attach_next(stage, prim_path, scale)
                if result is not None:
                    weight = result.weight
                    visual = result.visual
            elif config.sticker_probability > 0 and (random.random() < config.sticker_probability):
                result = self._sticker_attacher.attach(stage, prim_path, scale, box_type)
                if result is not None:
                    weight = result.weight
                    visual = result.visual

        if visual is None and config.visuals:
            visual = config.visuals[usd_idx]

        if weight is None:
            if box_type == "heavy":
                weight = random.uniform(15.0, 30.0)
            else:
                weight = random.uniform(5.0, 15.0)

        box_prim.set_masses(np.array([weight]))

        # 3. Record metadata
        meta: dict[str, object] = {
            "size": list(box_size),
            "weight": round(weight, 1),
            "type": box_type,
        }
        if visual is not None:
            meta["visual"] = visual
        self._box_metadata.append(meta)

        self._randomize_box_color(stage, prim_path)
        return prim_path

    def step(self) -> None:
        """Called each simulation step.

        Expires TTL-exceeded boxes, spawns a new box every spawn_interval
        seconds, and re-applies the conveyor velocity to every box still in
        the positive-Y lane so that boxes do not stall mid-belt.
        """
        self._expire_boxes()
        # Stop spawning when sticker attacher is exhausted (sequential mode)
        attacher_done = self._sticker_attacher is not None and self._sticker_attacher.done
        if (
            self._auto_spawn_enabled
            and not attacher_done
            and self._step_count > 0
            and self._step_count % self._spawn_interval_steps == 0
        ):
            self.spawn()

        # Re-apply conveyor velocity to all boxes still traveling on the belt.
        # Boxes that have been picked and placed are at negative Y (pallet side)
        # and are excluded by the y > 0.0 guard.
        for box, _, _ in self._boxes:
            try:
                pos, _ = box.get_world_poses()
                if float(pos[0, 1]) > 0.0:
                    box.set_linear_velocities(self._velocity[np.newaxis])
            except Exception:
                continue

        self._step_count += 1

    def hide_all(self) -> int:
        """Teleport all tracked boxes to z=-1000 and clear tracking.

        Box counter is NOT reset to avoid prim path conflicts with
        hidden-but-still-present USD prims.
        """
        hidden = 0
        for box, _path, _step in self._boxes:
            try:
                box.set_linear_velocities(np.zeros((1, 3)))
                box.set_angular_velocities(np.zeros((1, 3)))
                box.set_world_poses(positions=np.array([[0.0, 0.0, -1000.0]]))
            except Exception:
                pass
            hidden += 1
        self._boxes.clear()
        self._box_metadata.clear()
        self._step_count = 0
        return hidden

    def clear(self) -> None:
        """Release all RigidPrim references."""
        self._boxes.clear()
        self._box_count = 0
        self._step_count = 0
        self._box_metadata.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def auto_spawn_enabled(self) -> bool:
        """Whether interval-based auto-spawning is active."""
        return self._auto_spawn_enabled

    @auto_spawn_enabled.setter
    def auto_spawn_enabled(self, value: bool) -> None:
        self._auto_spawn_enabled = value

    @property
    def box_ttl_steps(self) -> int | None:
        """TTL in physics steps, or ``None`` if disabled."""
        return self._box_ttl_steps

    @box_ttl_steps.setter
    def box_ttl_steps(self, value: int | None) -> None:
        self._box_ttl_steps = value

    @property
    def box_count(self) -> int:
        return self._box_count

    @property
    def box_paths(self) -> list[str]:
        """Prim paths of all active boxes (same order as boxes)."""
        return [path for _, path, _ in self._boxes]

    @property
    def boxes(self) -> list:
        return [entry[0] for entry in self._boxes]

    @property
    def box_metadata(self) -> list[dict[str, object]]:
        """Per-box metadata collected during spawning."""
        return self._box_metadata

    @property
    def active_box(self) -> RigidPrim | None:
        """The box currently on the conveyor (y > 0, |x| < 0.7), or None."""
        for box, _, _ in reversed(self._boxes):
            try:
                positions, _ = box.get_world_poses()
            except Exception:
                continue
            x, y = float(positions[0, 0]), float(positions[0, 1])
            if y > 0.0 and abs(x) < 0.7:
                return box
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_type(self) -> str:
        """Sample a box type according to ``_type_weights``."""
        types = list(self._type_weights.keys())
        weights = [self._type_weights[t] for t in types]
        return random.choices(types, weights=weights, k=1)[0]

    @staticmethod
    def _compute_native_bbox(
        usd_path: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return ``(min_point, max_point)`` of the USD asset's bounding box.

        Values are in *stage units* which Isaac Sim treats as metres when
        loaded via ``add_reference_to_stage``.
        """
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(usd_path)
        root = stage.GetDefaultPrim()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
        lo, hi = rng.GetMin(), rng.GetMax()
        return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])

    def _resolve_variant(
        self, usd_path: str, box_size: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], tuple[float, float], float]:
        """Return ``(scale, (half_x, half_y), z_top)`` for a variant."""
        if usd_path not in self._bbox_cache:
            self._bbox_cache[usd_path] = self._compute_native_bbox(usd_path)
            lo, hi = self._bbox_cache[usd_path]
            nx, ny, nz = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
            print(f"[BoxSpawner] {Path(usd_path).stem}: native {nx:.4f} x {ny:.4f} x {nz:.4f}")
        lo, hi = self._bbox_cache[usd_path]
        nx, ny, nz = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
        scale = (box_size[0] / nx, box_size[1] / ny, box_size[2] / nz)
        return scale, (nx / 2, ny / 2), hi[2]

    @staticmethod
    def _randomize_box_color(stage: object, prim_path: str) -> None:
        """Apply slight random color jitter to the box material."""
        from pxr import Gf, Usd, UsdShade

        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            return
        for desc in Usd.PrimRange(root):
            if desc.GetTypeName() != "Shader":
                continue
            shader = UsdShade.Shader(desc)
            for name in ("diffuseColor", "diffuse_color_constant"):
                color_input = shader.GetInput(name)
                if not color_input or color_input.HasConnectedSource():
                    continue
                base = color_input.Get()
                if base is None:
                    continue
                r = float(base[0]) + random.uniform(-_COLOR_JITTER, _COLOR_JITTER)
                g = float(base[1]) + random.uniform(-_COLOR_JITTER, _COLOR_JITTER)
                b = float(base[2]) + random.uniform(-_COLOR_JITTER, _COLOR_JITTER)
                color_input.Set(
                    Gf.Vec3f(
                        max(0.0, min(1.0, r)),
                        max(0.0, min(1.0, g)),
                        max(0.0, min(1.0, b)),
                    )
                )
                return

    def _expire_boxes(self) -> None:
        """Deactivate boxes that have exceeded their TTL."""
        if self._box_ttl_steps is None:
            return
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        alive: list[tuple[RigidPrim, str, int]] = []
        for entry in self._boxes:
            box_prim, prim_path, spawn_step = entry
            if self._step_count - spawn_step >= self._box_ttl_steps:
                try:
                    box_prim.set_linear_velocities(np.zeros((1, 3)))
                    box_prim.set_world_poses(positions=np.array([[0.0, 0.0, -1000.0]]))
                except Exception:
                    pass
                try:
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        UsdGeom.Imageable(prim).MakeInvisible()
                except Exception:
                    pass
            else:
                alive.append(entry)
        self._boxes = alive
