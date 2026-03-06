"""Pallet pattern image generator using solver + Isaac Sim.

Orchestrates the pallet solver to compute box placements, spawns the
boxes as static USD prims on the pallet, then captures top-down RGB
images with metadata.  No physics simulation is needed -- boxes are
positioned directly by the solver.

Usage (via CLI)::

    uv run drp-sim --generate-pallet-pattern /tmp/pallet-out --num-patterns 50

Or programmatically::

    gen = PalletPatternGenerator(world, type_configs, ...)
    gen.setup()   # discovers pallet prim, scales to 1m x 1m, creates camera
    gen.run(num_patterns=50, seed=42)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

from drp_sim.box_spawner import BoxSpawner, BoxTypeConfig
from drp_sim.pallet_solver import (
    CELL_SIZE,
    GridBox,
    Orientation,
    Placement,
    SolverConstraints,
    grid_to_world,
    solve_greedy,
    solve_random,
)

if TYPE_CHECKING:
    from isaacsim.core.api import World

    from drp_sim.sticker_attacher import StickerAttacher

_DEFAULT_RESOLUTION = (960, 704)
_DEFAULT_CAMERA_HEIGHT = 4.5  # metres above pallet surface
_RENDER_SETTLE_FRAMES = 30
_GREEDY_PROBABILITY = 0.3
_PALLET_PRIM_PATH = "/World/pallet_with_dollly_02"
_TARGET_PALLET_SIZE = 1.0  # metres (1m x 1m)

# Pre-computed 90-degree Z-rotation quaternion (w, x, y, z).
_QUAT_90_Z = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))


class PalletPatternGenerator:
    """Generates pallet pattern images by solving placement and rendering in Isaac Sim.

    Parameters
    ----------
    world:
        Initialised Isaac Sim World instance.
    type_configs:
        Box type definitions from ``box_spawn.yaml``.
    type_weights:
        Sampling weights per box type.
    output_dir:
        Directory for output images and metadata.
    pallet_prim_path:
        USD prim path of the pallet in the loaded scene.  The pallet is
        scaled to 1 m x 1 m and its centre becomes the camera target.
    sticker_attacher:
        Optional sticker attacher for box decoration.
    min_boxes:
        Minimum boxes per pattern.
    max_boxes:
        Maximum boxes per pattern.
    resolution:
        (width, height) in pixels for captured images.
    camera_height:
        Camera height above the pallet surface.
    preview_seconds:
        Seconds to pause after each pattern for visual inspection.
        Set to 0.0 to disable.
    """

    def __init__(
        self,
        world: World,
        type_configs: dict[str, BoxTypeConfig],
        type_weights: dict[str, float],
        output_dir: str | Path,
        pallet_prim_path: str = _PALLET_PRIM_PATH,
        sticker_attacher: StickerAttacher | None = None,
        min_boxes: int = 3,
        max_boxes: int = 12,
        resolution: tuple[int, int] = _DEFAULT_RESOLUTION,
        camera_height: float = _DEFAULT_CAMERA_HEIGHT,
        preview_seconds: float = 3.0,
    ) -> None:
        self._world = world
        self._type_configs = type_configs
        self._type_weights = type_weights
        self._output_dir = Path(output_dir)
        self._pallet_prim_path = pallet_prim_path
        self._sticker_attacher = sticker_attacher
        self._min_boxes = min_boxes
        self._max_boxes = max_boxes
        self._resolution = resolution
        self._camera_height = camera_height
        self._preview_seconds = preview_seconds
        self._annotator: object | None = None
        self._pallet_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
        # usd_path -> (lo, hi) bounding box cache
        self._bbox_cache: dict[
            str, tuple[tuple[float, float, float], tuple[float, float, float]]
        ] = {}
        self._pattern_count = 0
        self._metadata_buffer: list[dict] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Discover the pallet prim, scale it to 1 m x 1 m, and create the camera.

        Must be called after ``World.reset()`` so the USD stage is loaded.
        The pallet centre becomes the camera target, and the pallet surface
        top becomes z = 0 for box stacking (``pallet_origin``).
        """
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()

        # -- Discover pallet prim and compute bounding box --
        pallet_prim = stage.GetPrimAtPath(self._pallet_prim_path)
        if not pallet_prim.IsValid():
            raise RuntimeError(f"Pallet prim not found: {self._pallet_prim_path}")

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = cache.ComputeWorldBound(pallet_prim).ComputeAlignedRange()
        lo, hi = bbox.GetMin(), bbox.GetMax()
        native_w = hi[0] - lo[0]
        native_d = hi[1] - lo[1]
        center_x = (lo[0] + hi[0]) / 2
        center_y = (lo[1] + hi[1]) / 2
        surface_z = hi[2]  # top of pallet

        print(
            f"[PalletPatternGenerator] Pallet bbox: "
            f"({lo[0]:.3f},{lo[1]:.3f},{lo[2]:.3f}) -> ({hi[0]:.3f},{hi[1]:.3f},{hi[2]:.3f})"
        )
        print(f"[PalletPatternGenerator] Pallet native size: {native_w:.3f} x {native_d:.3f}")

        # -- Scale pallet to TARGET_PALLET_SIZE x TARGET_PALLET_SIZE --
        scale_x = _TARGET_PALLET_SIZE / native_w if native_w > 0 else 1.0
        scale_y = _TARGET_PALLET_SIZE / native_d if native_d > 0 else 1.0

        xf_pallet = UsdGeom.Xformable(pallet_prim)
        # Check if a scale op already exists; if so, update it
        existing_scale = None
        for op in xf_pallet.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                existing_scale = op
                break
        if existing_scale is not None:
            prev = existing_scale.Get()
            new_scale = Gf.Vec3f(float(prev[0]) * scale_x, float(prev[1]) * scale_y, float(prev[2]))
            existing_scale.Set(new_scale)
        else:
            xf_pallet.AddScaleOp().Set(Gf.Vec3f(scale_x, scale_y, 1.0))

        # Re-compute bbox after scaling
        cache.Clear()
        bbox = cache.ComputeWorldBound(pallet_prim).ComputeAlignedRange()
        lo, hi = bbox.GetMin(), bbox.GetMax()
        center_x = (lo[0] + hi[0]) / 2
        center_y = (lo[1] + hi[1]) / 2
        surface_z = hi[2]

        # Pallet origin = corner (min-x, min-y) at the surface top
        half = _TARGET_PALLET_SIZE / 2
        self._pallet_origin = (center_x - half, center_y - half, surface_z)

        print(
            f"[PalletPatternGenerator] Pallet origin (corner): {self._pallet_origin}, "
            f"centre: ({center_x:.3f}, {center_y:.3f}, {surface_z:.3f})"
        )

        # -- Camera directly above pallet centre --
        cam_path = "/World/PalletCaptureCam"
        UsdGeom.Camera.Define(stage, cam_path)
        xf_cam = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
        xf_cam.AddTranslateOp().Set(Gf.Vec3d(center_x, center_y, surface_z + self._camera_height))
        # Default UsdGeom.Camera looks along -Z, perfect for top-down

        rp = rep.create.render_product(cam_path, self._resolution)
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotator.attach([rp])

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pattern_count = self._next_index(self._output_dir)
        print(f"[PalletPatternGenerator] Output: {self._output_dir} (start={self._pattern_count})")

    def run(self, num_patterns: int, seed: int | None = 42) -> None:
        """Generate *num_patterns* pallet pattern images.

        Parameters
        ----------
        num_patterns:
            Number of patterns to generate.
        seed:
            Base random seed.  Each pattern derives its own seed as
            ``base_seed + pattern_index`` for reproducibility.
        """
        import numpy as np

        if self._annotator is None:
            raise RuntimeError("Call setup() before run()")

        base_seed = seed if seed is not None else random.randint(0, 2**31)
        constraints = SolverConstraints()

        for i in range(num_patterns):
            if not self._world.is_playing():
                print("[PalletPatternGenerator] Simulation stopped, ending generation.")
                break

            pattern_seed = base_seed + i
            rng = random.Random(pattern_seed)
            np.random.seed(pattern_seed)
            random.seed(pattern_seed)

            num_boxes = rng.randint(self._min_boxes, self._max_boxes)
            queue = self._generate_queue(num_boxes, rng)

            # Choose solver: 30% greedy for compact patterns, 70% random for diversity
            if rng.random() < _GREEDY_PROBABILITY:
                placements = solve_greedy(queue, constraints)
                solver_name = "greedy"
            else:
                placements = solve_random(queue, rng, constraints)
                solver_name = "random"

            if not placements:
                print(f"[PalletPatternGenerator] Pattern {i}: no valid placements, skipping")
                continue

            prim_paths = self._spawn_pattern(placements)
            self._settle_render()
            self._randomize_lighting()

            fill_ratio = self._compute_fill_ratio(placements)
            metadata = {
                "image": f"pallet_{self._pattern_count:04d}.png",
                "num_boxes": len(placements),
                "solver": solver_name,
                "seed": pattern_seed,
                "fill_ratio": round(fill_ratio, 3),
                "boxes": [
                    {
                        "type": p.box_type,
                        "size": list(p.real_size),
                        "grid_pos": [p.grid_x, p.grid_y, p.grid_z],
                        "world_pos": list(grid_to_world(p, self._pallet_origin)),
                        "orientation": p.orientation.value,
                    }
                    for p in placements
                ],
            }

            self._capture(metadata)

            if not self._preview_wait():
                self._clear_pattern(prim_paths)
                self._settle_render()
                print("[PalletPatternGenerator] Simulation stopped, ending generation.")
                break

            self._clear_pattern(prim_paths)
            self._settle_render()  # flush removed prims before next pattern

            print(
                f"[PalletPatternGenerator] Pattern {self._pattern_count - 1}: "
                f"{len(placements)} boxes, {solver_name}, fill={fill_ratio:.1%}"
            )

        self._flush_metadata()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_queue(self, n: int, rng: random.Random) -> list[GridBox]:
        """Sample *n* GridBox instances from type configs and weights."""
        types = list(self._type_weights.keys())
        weights = [self._type_weights[t] for t in types]
        queue: list[GridBox] = []

        for i in range(n):
            box_type = rng.choices(types, weights=weights, k=1)[0]
            config = self._type_configs[box_type]
            sx = rng.choice(config.x_choices)
            sy = rng.choice(config.y_choices)
            sz = rng.choice(config.z_choices)
            # Convert metres to grid cells
            gw = max(1, round(sx / CELL_SIZE))
            gd = max(1, round(sy / CELL_SIZE))
            gh = max(1, round(sz / CELL_SIZE))
            queue.append(
                GridBox(box_id=i, w=gw, d=gd, h=gh, box_type=box_type, real_size=(sx, sy, sz))
            )
        return queue

    def _spawn_pattern(self, placements: list[Placement]) -> list[str]:
        """Spawn all boxes at solver-determined positions. Returns prim paths.

        Each prim is placed using TRS xform order (Translate, Orient, Scale)
        and the translate is corrected for the native mesh origin offset so the
        box's geometric centre lands at the solver-computed world position.
        """
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim_paths: list[str] = []

        for p in placements:
            config = self._type_configs[p.box_type]
            usd_path = random.choice(config.usd_paths)

            # Native bounding box of the USD asset
            lo, hi = self._get_native_bbox(usd_path)
            nx = hi[0] - lo[0]
            ny = hi[1] - lo[1]
            nz = hi[2] - lo[2]

            # Native geometric centre (may differ from prim origin)
            nc_x = (lo[0] + hi[0]) / 2
            nc_y = (lo[1] + hi[1]) / 2
            nc_z = (lo[2] + hi[2]) / 2

            # Scale to match desired real size.
            # TRS order: Scale applies first (local), then Orient rotates.
            # After Scale, local extents = (nx*sx, ny*sy, nz*sz).
            # After 90-deg Z rot, world extents = (ny*sy, nx*sx, nz*sz).
            rx, ry, rz = p.real_size
            if p.orientation is Orientation.ROTATED:
                # ny*sy = rx, nx*sx = ry  =>  sy = rx/ny, sx = ry/nx
                scale = (ry / nx, rx / ny, rz / nz)
            else:
                scale = (rx / nx, ry / ny, rz / nz)

            # Scaled centre offset (in local space before rotation)
            sc_x = nc_x * scale[0]
            sc_y = nc_y * scale[1]
            sc_z = nc_z * scale[2]

            # Desired world-space geometric centre from solver
            wx, wy, wz = grid_to_world(p, self._pallet_origin)

            # Rotate scaled centre offset to world space
            if p.orientation is Orientation.ROTATED:
                # 90-deg CCW around Z: (x, y, z) -> (-y, x, z)
                off_x, off_y, off_z = -sc_y, sc_x, sc_z
            else:
                off_x, off_y, off_z = sc_x, sc_y, sc_z

            # Translate = desired centre - rotated scaled centre offset
            tx = wx - off_x
            ty = wy - off_y
            tz = wz - off_z

            prim_path = f"/World/pallet_box_{p.box_id}"
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

            prim = stage.GetPrimAtPath(prim_path)
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()

            # TRS order: Translate, Orient, Scale
            xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
            if p.orientation is Orientation.ROTATED:
                w, x, y, z = _QUAT_90_Z
                xformable.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            xformable.AddScaleOp().Set(Gf.Vec3f(*scale))

            # Sticker
            if self._sticker_attacher is not None and config.sticker_probability > 0:
                native_half = (nx / 2, ny / 2)
                z_top = hi[2]
                self._sticker_attacher._box_half = (native_half[0], native_half[1], z_top)
                if random.random() < config.sticker_probability:
                    self._sticker_attacher.attach(stage, prim_path, scale, p.box_type)

            # Color jitter
            BoxSpawner._randomize_box_color(stage, prim_path)

            prim_paths.append(prim_path)

        return prim_paths

    def _settle_render(self) -> None:
        """Step the world (render only) to flush the render pipeline."""
        for _ in range(_RENDER_SETTLE_FRAMES):
            self._world.step(render=True)

    def _preview_wait(self) -> bool:
        """Render-step for ``preview_seconds``, keeping the GUI responsive.

        Returns ``True`` if the preview completed normally, ``False`` if
        the simulation was paused or stopped (caller should break).
        """
        if self._preview_seconds <= 0:
            return True
        fps = 60.0
        steps = int(self._preview_seconds * fps)
        for _ in range(steps):
            if not self._world.is_playing():
                return False
            self._world.step(render=True)
        return True

    def _randomize_lighting(self) -> None:
        """Apply random tilt to the domain-randomisation light."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        path = "/World/DomainRandomLight"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rx = random.uniform(-15.0, 15.0)
                ry = random.uniform(-15.0, 15.0)
                op.Set(Gf.Vec3f(rx, ry, 0.0))
                break

    def _capture(self, metadata: dict) -> None:
        """Capture RGB image and buffer metadata."""
        from PIL import Image

        data = self._annotator.get_data()
        if data is None:
            return
        if isinstance(data, dict):
            data = data["data"]
        if data.size == 0:
            return

        img = Image.fromarray(data[:, :, :3])
        image_name = f"pallet_{self._pattern_count:04d}.png"
        img.save(str(self._output_dir / image_name))

        self._metadata_buffer.append(metadata)
        self._pattern_count += 1

    def _flush_metadata(self) -> None:
        """Write buffered metadata entries to ``metadata.json``."""
        if not self._metadata_buffer:
            return
        meta_path = self._output_dir / "metadata.json"
        entries: list[dict] = []
        if meta_path.exists():
            entries = json.loads(meta_path.read_text())
        entries.extend(self._metadata_buffer)
        meta_path.write_text(json.dumps(entries, indent=2) + "\n")
        self._metadata_buffer.clear()

    def _clear_pattern(self, prim_paths: list[str]) -> None:
        """Remove spawned box prims from the stage."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for path in prim_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(path)

    def _get_native_bbox(
        self, usd_path: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Cached native bounding box lookup."""
        if usd_path not in self._bbox_cache:
            self._bbox_cache[usd_path] = BoxSpawner._compute_native_bbox(usd_path)
        return self._bbox_cache[usd_path]

    @staticmethod
    def _compute_fill_ratio(placements: list[Placement]) -> float:
        """Fraction of the 4x4 ground grid covered by box footprints."""
        from drp_sim.pallet_solver import GRID_D, GRID_W

        covered: set[tuple[int, int]] = set()
        for p in placements:
            rx, ry, _rz = p.real_size
            gw = max(1, round(rx / CELL_SIZE))
            gd = max(1, round(ry / CELL_SIZE))
            for x in range(p.grid_x, p.grid_x + gw):
                for y in range(p.grid_y, p.grid_y + gd):
                    covered.add((x, y))
        return len(covered) / (GRID_W * GRID_D)

    @staticmethod
    def _next_index(directory: Path) -> int:
        """Return the next available index by scanning existing ``pallet_*.png``."""
        max_idx = -1
        for p in directory.glob("pallet_*.png"):
            parts = p.stem.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
        return max_idx + 1
