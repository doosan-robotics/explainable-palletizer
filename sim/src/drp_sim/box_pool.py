"""Lazy object pool for USD box prims with embedded stickers.

Boxes are created on first acquire and recycled via release, avoiding
expensive ``add_reference_to_stage`` + ``RigidPrim.initialize()`` on
repeated spawns.  Warm-up only pre-caches native bounding boxes so that
no physics actors exist until actually needed -- keeping the PhysX tensor
view clean and stable.

Pool key is ``(usd_asset_path, box_size)`` so that the ScaleOp is set once
at creation and never mutated afterwards.  Changing the ScaleOp on an
already-initialized RigidPrim triggers a Fabric prototype cascade
invalidation that crashes the PhysX tensor view.

IMPORTANT: ``release()`` disables kinematic mode before zeroing velocities
to avoid ``PxRigidDynamic::setAngularVelocity: Body must be non-kinematic!``
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim

    from drp_sim.box_spawner import BoxTypeConfig

logger = logging.getLogger(__name__)

# (usd_path, (x, y, z)) -> deque[PooledBox]
_PoolKey = tuple[str, tuple[float, float, float]]


@dataclass
class EmbeddedSticker:
    """Sticker child prim embedded in a pooled box.

    The mesh and material are created once during warm-up and reused
    across acquire/release cycles.  ``MakeVisible`` / ``MakeInvisible``
    toggles display without affecting PhysX (stickers are non-physics prims).
    """

    mesh_path: str
    mat_path: str
    visible: bool = False

    def update(
        self,
        stage: object,
        image_path: str,
        aspect: float,
        box_half: tuple[float, float, float],
        parent_scale: tuple[float, float, float],
    ) -> None:
        """Update sticker geometry and texture for a new box variant."""
        from drp_sim.sticker_attacher import update_sticker_geometry, update_sticker_texture

        update_sticker_geometry(stage, self.mesh_path, aspect, box_half, parent_scale)
        update_sticker_texture(stage, self.mat_path, image_path)

    def show(self, stage: object) -> None:
        """Make the sticker mesh visible."""
        from pxr import UsdGeom

        prim = stage.GetPrimAtPath(self.mesh_path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeVisible()
        self.visible = True

    def hide(self, stage: object) -> None:
        """Make the sticker mesh invisible."""
        from pxr import UsdGeom

        prim = stage.GetPrimAtPath(self.mesh_path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
        self.visible = False


@dataclass
class PooledBox:
    """A box managed by the pool."""

    box_prim: RigidPrim
    prim_path: str
    usd_path: str
    box_size: tuple[float, float, float]
    sticker: EmbeddedSticker
    native_bbox: tuple[tuple[float, float, float], tuple[float, float, float]]


class BoxPool:
    """Lazily creates and recycles USD box prims.

    Pools are keyed by ``(usd_path, box_size)`` so that each prim's
    ScaleOp is immutable after ``RigidPrim.initialize()``.  Mutating
    the ScaleOp on a live physics prim triggers a Fabric prototype
    cascade that invalidates the PhysX tensor view.

    Parameters
    ----------
    world:
        Isaac Sim World instance (must be initialised before warm-up).
    env_path:
        Parent prim path under which pooled box prims are created.
    """

    def __init__(self, world: World, env_path: str = "/World") -> None:
        self._world = world
        self._env_path = env_path
        self._pools: dict[_PoolKey, deque[PooledBox]] = {}
        self._in_use: dict[str, PooledBox] = {}
        self._box_count: int = 0
        self._bbox_cache: dict[
            str, tuple[tuple[float, float, float], tuple[float, float, float]]
        ] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def warm_up(self, type_configs: dict[str, BoxTypeConfig]) -> None:
        """Pre-cache native bounding boxes for all USD assets.

        No physics prims are created here -- boxes are created lazily on
        first ``acquire()`` and recycled after ``release()``.  This keeps
        the PhysX scene clean: only actively used boxes exist as actors.
        """
        from pathlib import Path

        assets: set[str] = set()
        for config in type_configs.values():
            for usd_path in config.usd_paths:
                if usd_path not in assets:
                    assets.add(usd_path)
                    self._get_native_bbox(usd_path)

        names = {Path(a).stem for a in assets}
        logger.info("[BoxPool] cached %d asset bboxes (%s)", len(assets), names)

    def acquire(
        self,
        usd_path: str,
        box_size: tuple[float, float, float],
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> PooledBox:
        """Acquire a box from the pool, positioning it at the given pose.

        If the pool for ``(usd_path, box_size)`` is empty, a new box is
        created on the fly (the pool grows dynamically).

        Boxes returned by this method are always in dynamic (non-kinematic)
        mode so velocity setters are safe.
        """
        import omni.usd

        key: _PoolKey = (usd_path, box_size)

        if key not in self._pools:
            self._pools[key] = deque()

        if self._pools[key]:
            pooled = self._pools[key].popleft()
        else:
            pooled = self._create_pooled_box(usd_path, box_size)

        stage = omni.usd.get_context().get_stage()

        # Scale was set at creation time -- no ScaleOp mutation needed.
        # Just set pose and velocity.
        orientation = np.array([(1.0, 0.0, 0.0, 0.0)], dtype=float)
        pooled.box_prim.set_world_poses(positions=position[np.newaxis], orientations=orientation)
        pooled.box_prim.set_linear_velocities(velocity[np.newaxis])
        pooled.box_prim.set_angular_velocities(np.zeros((1, 3)))

        # Ensure sticker is hidden (spawner will show if needed)
        pooled.sticker.hide(stage)

        # Make box visible again (was hidden when released to pool)
        from pxr import UsdGeom

        prim = stage.GetPrimAtPath(pooled.prim_path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeVisible()

        self._in_use[pooled.prim_path] = pooled
        return pooled

    def release(self, prim_path: str) -> None:
        """Release a box back to its pool.

        Disables kinematic mode first (if enabled), then zeros velocities
        and teleports to z=-1000.  This ordering prevents the PhysX error
        ``PxRigidDynamic::setAngularVelocity: Body must be non-kinematic!``
        """
        import omni.usd

        pooled = self._in_use.pop(prim_path, None)
        if pooled is None:
            return

        # Disable kinematic BEFORE touching velocities
        self._ensure_dynamic(pooled.box_prim)

        try:
            pooled.box_prim.set_linear_velocities(np.zeros((1, 3)))
            pooled.box_prim.set_angular_velocities(np.zeros((1, 3)))
            pooled.box_prim.set_world_poses(positions=np.array([[0.0, 0.0, -1000.0]]))
        except Exception:
            pass

        try:
            stage = omni.usd.get_context().get_stage()
            pooled.sticker.hide(stage)
            from pxr import UsdGeom

            prim = stage.GetPrimAtPath(pooled.prim_path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        except Exception:
            pass

        key: _PoolKey = (pooled.usd_path, pooled.box_size)
        self._pools.setdefault(key, deque()).append(pooled)

    def abandon(self, prim_path: str) -> None:
        """Remove a box from pool tracking WITHOUT recycling the prim.

        Use this for boxes placed on the pallet: the prim stays at its
        current world position and is never reused by the pool.
        """
        self._in_use.pop(prim_path, None)

    def release_all(self) -> int:
        """Release all in-use boxes back to their pools."""
        paths = list(self._in_use.keys())
        for path in paths:
            self.release(path)
        return len(paths)

    def get_pooled(self, prim_path: str) -> PooledBox | None:
        """Look up an in-use PooledBox by prim path."""
        return self._in_use.get(prim_path)

    def stats(self) -> dict:
        """Return pool utilisation statistics."""
        available = sum(len(d) for d in self._pools.values())
        return {
            "available": available,
            "in_use": len(self._in_use),
            "total": available + len(self._in_use),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_dynamic(box_prim: RigidPrim) -> None:
        """Disable kinematic mode so velocity setters are safe.

        Batch-created prims (prim_paths_expr) lack ``.prim``; skip them
        to avoid invalidating the Fabric tensor view.
        """
        if not hasattr(box_prim, "prim"):
            return
        try:
            from pxr import UsdPhysics

            rb_api = UsdPhysics.RigidBodyAPI(box_prim.prim)
            if not rb_api:
                return
            kinematic_attr = rb_api.GetKinematicEnabledAttr()
            if kinematic_attr and kinematic_attr.Get():
                rb_api.CreateKinematicEnabledAttr().Set(False)
        except Exception:
            pass

    def _create_pooled_box(self, usd_path: str, box_size: tuple[float, float, float]) -> PooledBox:
        """Create a new USD box prim with an embedded sticker.

        Called lazily on first ``acquire()`` for a given pool key, or
        when the recycled pool is empty.  The ScaleOp is computed from
        ``box_size`` and set once -- it must never be mutated after
        ``RigidPrim.initialize()`` to avoid Fabric prototype invalidation.
        """
        import omni.usd
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, UsdGeom

        box_name = f"pool_box_{self._box_count}"
        prim_path = f"{self._env_path}/{box_name}"
        self._box_count += 1

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        native_bbox = self._get_native_bbox(usd_path)
        lo, hi = native_bbox
        nx, ny, nz = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
        scale = (box_size[0] / nx, box_size[1] / ny, box_size[2] / nz)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -1000.0))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))

        box_prim = RigidPrim(prim_paths_expr=prim_path, name=box_name)
        self._world.scene.add(box_prim)
        box_prim.initialize()

        sticker = self._create_embedded_sticker(stage, prim_path)

        return PooledBox(
            box_prim=box_prim,
            prim_path=prim_path,
            usd_path=usd_path,
            box_size=box_size,
            sticker=sticker,
            native_bbox=native_bbox,
        )

    def _get_native_bbox(
        self, usd_path: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Compute and cache the native bounding box for a USD asset."""
        if usd_path not in self._bbox_cache:
            from drp_sim.box_spawner import BoxSpawner

            self._bbox_cache[usd_path] = BoxSpawner._compute_native_bbox(usd_path)
        return self._bbox_cache[usd_path]

    @staticmethod
    def _create_embedded_sticker(stage: object, box_prim_path: str) -> EmbeddedSticker:
        """Create a sticker mesh + material as children, initially hidden."""
        from pxr import UsdGeom

        from drp_sim.sticker_attacher import build_sticker_material, create_sticker_mesh

        sticker_path = f"{box_prim_path}/Sticker"
        mat_path = f"{box_prim_path}/Looks/StickerMat"

        build_sticker_material(stage, mat_path)
        create_sticker_mesh(stage, sticker_path, mat_path)

        prim = stage.GetPrimAtPath(sticker_path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()

        return EmbeddedSticker(mesh_path=sticker_path, mat_path=mat_path, visible=False)
