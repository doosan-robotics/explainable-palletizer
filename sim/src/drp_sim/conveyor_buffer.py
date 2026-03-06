"""Bounded conveyor buffer with compaction toward -Y.

Boxes spawn at the conveyor start and ride the belt toward the buffer.
When a box crosses the snap threshold (Y < SNAP_Y), it teleports to its
assigned slot with identity rotation, then becomes kinematic (no physics
drift).  Slots fill from -Y (closest to robot) toward +Y.

On refill after a pick, existing boxes compact toward -Y first, then
new boxes fill the remaining +Y slots via conveyor.

Example with 3 slots (0=most -Y, 2=most +Y):
    fill:  [0,0,0] -> [1,0,0] -> [1,1,0] -> [1,1,1]
    pick:  [1,1,1] -> [0,1,1]
    fill:  compact -> [1,1,0] -> spawn+ride -> [1,1,1]
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaacsim.core.prims import RigidPrim

    from drp_sim.box_pool import BoxPool
    from drp_sim.box_spawner import BoxSpawner

logger = logging.getLogger(__name__)

# Identity quaternion (w, x, y, z) — Isaac Sim convention
_IDENTITY_QUAT = np.array([[1.0, 0.0, 0.0, 0.0]])


@dataclass
class BufferSlot:
    """One slot in the conveyor buffer."""

    box_prim: RigidPrim
    prim_path: str
    y_length: float
    box_half_h: float
    assigned_position: tuple[float, float, float]


@dataclass
class _InTransit:
    """A box riding the conveyor toward a buffer slot."""

    box_prim: RigidPrim
    prim_path: str
    target_slot: int
    y_length: float
    box_half_h: float


class ConveyorBuffer:
    """Bounded buffer with conveyor fill and -Y compaction.

    Parameters
    ----------
    spawner:
        The ``BoxSpawner`` used to create new box prims.
    endpoint:
        (x, y, z) world position of slot 0 (closest to robot, most -Y).
    length:
        Maximum number of boxes the buffer holds.
    conveyor_velocity:
        (vx, vy, vz) belt velocity applied to in-transit boxes.
    snap_margin:
        Distance threshold relative to the target slot to snap the box.
    """

    _SLOT_STRIDE_Y = 0.9
    """Fixed Y-spacing between consecutive buffer slots."""

    _SIM_DT = 1.0 / 60.0
    """Fallback Isaac Sim physics timestep (60 Hz)."""

    def __init__(
        self,
        spawner: BoxSpawner,
        endpoint: tuple[float, float, float],
        length: int = 3,
        conveyor_velocity: tuple[float, float, float] = (0.0, -0.15, 0.0),
        snap_margin: float = 0.05,
        pool: BoxPool | None = None,
    ) -> None:
        self._spawner = spawner
        self._endpoint = endpoint
        self._length = length
        self._conveyor_velocity = np.array(conveyor_velocity, dtype=float)
        self._snap_margin = snap_margin
        self._pool = pool
        self._slots: list[BufferSlot | None] = [None] * length
        self._active = False
        self._pending_slots: list[int] = []
        self._in_transit: _InTransit | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> int:
        """Hide all buffer boxes (slots + in-transit) and reset to empty state.

        When a pool is available, boxes are released back for recycling.
        """
        hidden = 0
        if self._in_transit is not None:
            if self._pool is not None:
                self._pool.release(self._in_transit.prim_path)
            else:
                try:
                    self._in_transit.box_prim.set_linear_velocities(np.zeros((1, 3)))
                    self._in_transit.box_prim.set_angular_velocities(np.zeros((1, 3)))
                    self._in_transit.box_prim.set_world_poses(
                        positions=np.array([[0.0, 0.0, -1000.0]])
                    )
                except Exception:
                    pass
            hidden += 1
            self._in_transit = None
        for slot in self._slots:
            if slot is None:
                continue
            if self._pool is not None:
                self._pool.release(slot.prim_path)
            else:
                try:
                    slot.box_prim.set_linear_velocities(np.zeros((1, 3)))
                    slot.box_prim.set_angular_velocities(np.zeros((1, 3)))
                    slot.box_prim.set_world_poses(positions=np.array([[0.0, 0.0, -1000.0]]))
                except Exception:
                    pass
            hidden += 1
        self._slots = [None] * self._length
        self._pending_slots.clear()
        self._active = False
        return hidden

    def fill(self) -> dict:
        self._active = True
        self._spawner.auto_spawn_enabled = False
        self._spawner.box_ttl_steps = None

        self._compact()

        for i in range(self._length):
            if self._slots[i] is None and i not in self._pending_slots:
                self._pending_slots.append(i)

        logger.info(
            "Buffer fill: occupied=%d/%d pending=%s in_transit=%s",
            self.occupied_count,
            self._length,
            self._pending_slots,
            self._in_transit is not None,
        )

        return {
            "status": "filling",
            "occupied": self.occupied_count,
            "capacity": self._length,
        }

    def step(self) -> None:
        if not self._active:
            return

        if self._in_transit is not None:
            self._check_arrival()

        self._enforce_slot_positions()

        if self._in_transit is None and self._pending_slots:
            self._dispatch_next()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def slot_count(self) -> int:
        return self._length

    @property
    def occupied_count(self) -> int:
        return sum(1 for s in self._slots if s is not None)

    @property
    def slot_states(self) -> list[int]:
        return [1 if s is not None else 0 for s in self._slots]

    @property
    def buffer_boxes(self) -> list[tuple[RigidPrim, str]]:
        return [(s.box_prim, s.prim_path) for s in self._slots if s is not None]

    def pop_nearest_box(self) -> tuple[RigidPrim, str] | None:
        """Remove slot 0 (most -Y, closest to robot). Alias for ``pop_box_at(0)``."""
        return self.pop_box_at(0)

    def pop_box_at(self, index: int) -> tuple[RigidPrim, str] | None:
        if index < 0 or index >= self._length:
            logger.warning("pop_box_at: index %d out of range [0, %d)", index, self._length)
            return None

        slot = self._slots[index]
        if slot is None:
            return None

        result = (slot.box_prim, slot.prim_path)
        self._slots[index] = None

        # NOTE: intentionally keep kinematic enabled here. Disabling it
        # changes the Fabric physics state, which can invalidate the mesh
        # prototype before cuRobo refreshes its collision snapshot, causing
        # a segfault.  The caller (pick-and-place) uses _snap_rigid for
        # positioning anyway, so physics is not needed on the box.
        # Call release_box() after the pick sequence is complete.

        logger.debug(
            "Popped buffer slot %d (y=%.3f, %s)",
            index,
            slot.assigned_position[1],
            slot.prim_path,
        )
        return result

    def release_box(self, box_prim: RigidPrim, prim_path: str | None = None) -> None:
        """Release a previously popped box back to the pool or disable kinematic.

        Call this **after** the pick-and-place sequence is complete so that
        cuRobo's collision world has time to refresh before the physics
        state changes.
        """
        if self._pool is not None and prim_path is not None:
            self._pool.release(prim_path)
        else:
            # Hide first so the box is invisible even if the teleport fails.
            if prim_path is not None:
                self._hide_prim(prim_path)
            # Keep kinematic enabled: box stays frozen at z=-1000, no physics drift.
            # Disabling kinematic mid-step can invalidate cuRobo's collision snapshot.
            with contextlib.suppress(Exception):
                box_prim.set_world_poses(positions=np.array([[0.0, 0.0, -1000.0]]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compact(self) -> None:
        alive: list[BufferSlot] = []
        for slot in self._slots:
            if slot is None:
                continue
            if self._is_consumed(slot):
                logger.warning("Compact: dropping dead prim %s", slot.prim_path)
                continue
            alive.append(slot)

        self._slots = [None] * self._length

        for i, slot in enumerate(alive):
            new_pos = self._compute_slot_position(i)
            slot.assigned_position = new_pos
            try:
                self._pin_to_position(slot.box_prim, new_pos)
            except Exception:
                logger.warning("Compact: prim %s died during move", slot.prim_path)
                continue
            self._slots[i] = slot

        if alive:
            logger.debug("Compacted %d boxes toward -Y", len(alive))

    def _dispatch_next(self) -> None:
        if not self._pending_slots:
            return

        target_slot = self._pending_slots.pop(0)

        prim_path = self._spawner.spawn()
        box_prim = self._spawner.boxes[-1]
        meta = self._spawner.box_metadata[-1]
        y_length = float(meta["size"][1])
        box_half_h = float(meta["size"][2]) / 2.0

        # 초기 속도/회전 완벽 초기화
        box_prim.set_linear_velocities(np.zeros((1, 3)))
        box_prim.set_angular_velocities(np.zeros((1, 3)))

        self._set_kinematic(box_prim, enabled=True)

        self._in_transit = _InTransit(
            box_prim=box_prim,
            prim_path=prim_path,
            target_slot=target_slot,
            y_length=y_length,
            box_half_h=box_half_h,
        )

        self._spawner._boxes = [entry for entry in self._spawner._boxes if entry[1] != prim_path]

        logger.debug("Dispatched box %s to slot %d", prim_path, target_slot)

    def _check_arrival(self) -> None:
        transit = self._in_transit
        assert transit is not None

        try:
            pos, _ = transit.box_prim.get_world_poses()
            current_y = float(pos[0, 1])
        except Exception:
            logger.warning("In-transit box %s lost", transit.prim_path)
            if transit.target_slot not in self._pending_slots:
                self._pending_slots.append(transit.target_slot)
            self._in_transit = None
            return

        target_pos = self._compute_slot_position(transit.target_slot)
        target_y = target_pos[1]

        if current_y <= target_y + self._snap_margin:
            self._snap_to_slot(transit)
            self._in_transit = None
        else:
            try:
                import omni.timeline

                dt = omni.timeline.get_timeline_interface().get_time_step()
                if dt <= 0:
                    dt = self._SIM_DT
            except Exception:
                dt = self._SIM_DT

            new_pos = pos[0] + self._conveyor_velocity * dt
            new_y = float(new_pos[1])

            if new_y <= target_y + self._snap_margin:
                self._snap_to_slot(transit)
                self._in_transit = None
            else:
                transit.box_prim.set_world_poses(
                    positions=new_pos[np.newaxis],
                    orientations=_IDENTITY_QUAT,
                )
                # Zero velocities every step to counteract physics drift.
                # _set_kinematic silently fails for prim_paths_expr prims,
                # so without this the box accumulates physics momentum and
                # overshoots into occupied slots.
                transit.box_prim.set_linear_velocities(np.zeros((1, 3)))
                transit.box_prim.set_angular_velocities(np.zeros((1, 3)))

    def _snap_to_slot(self, transit: _InTransit) -> None:
        idx = transit.target_slot
        position = self._compute_slot_position(idx)

        try:
            self._pin_to_position(transit.box_prim, position)
            self._set_kinematic(transit.box_prim, enabled=True)
        except Exception:
            logger.warning("Snap failed for %s, requeueing slot %d", transit.prim_path, idx)
            if idx not in self._pending_slots:
                self._pending_slots.append(idx)
            return

        self._slots[idx] = BufferSlot(
            box_prim=transit.box_prim,
            prim_path=transit.prim_path,
            y_length=transit.y_length,
            box_half_h=transit.box_half_h,
            assigned_position=position,
        )

        logger.debug("Slot %d filled at y=%.3f (%s)", idx, position[1], transit.prim_path)

    def _evict_dead_slots(self) -> None:
        for i, slot in enumerate(self._slots):
            if slot is None:
                continue
            if self._is_consumed(slot):
                logger.warning("Slot %d prim invalidated, evicting", i)
                self._slots[i] = None
                if i not in self._pending_slots:
                    self._pending_slots.append(i)

    def _enforce_slot_positions(self) -> None:
        """Pin every slot box at its assigned position.

        Velocity zeroing is intentionally omitted: slot boxes are kinematic,
        and calling set_linear/angular_velocities on kinematic bodies raises a
        PhysX error that can corrupt the Fabric tensor state.  Position-only
        pinning is sufficient because kinematic bodies ignore physics forces.
        """
        for slot in self._slots:
            if slot is None:
                continue
            with contextlib.suppress(Exception):
                self._pin_to_position(slot.box_prim, slot.assigned_position)

    # ------------------------------------------------------------------
    # Prim helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hide_prim(prim_path: str) -> None:
        """Make a USD prim invisible so it is not rendered."""
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        except Exception:
            pass

    @staticmethod
    def _pin_to_position(
        box_prim: RigidPrim,
        position: tuple[float, float, float],
    ) -> None:
        box_prim.set_world_poses(
            positions=np.array([position]),
            orientations=_IDENTITY_QUAT,
        )

    @staticmethod
    def _set_kinematic(box_prim: RigidPrim, *, enabled: bool) -> None:
        """Best-effort kinematic toggle.

        Batch-created prims (BoxPool, prim_paths_expr) lack the `.prim`
        attribute and modifying their RigidBody physics attributes can
        invalidate the Fabric tensor view, causing a segfault.  For these
        prims we skip entirely and rely on ``_enforce_slot_positions()``
        to pin boxes every step instead.
        """
        if not hasattr(box_prim, "prim"):
            return
        try:
            from pxr import UsdPhysics

            rb_api = UsdPhysics.RigidBodyAPI(box_prim.prim)
            if not rb_api:
                rb_api = UsdPhysics.RigidBodyAPI.Apply(box_prim.prim)

            rb_api.CreateKinematicEnabledAttr().Set(enabled)

        except Exception as e:
            logger.debug("Could not set kinematic=%s on %s: %s", enabled, box_prim.name, e)

    @staticmethod
    def _is_consumed(slot: BufferSlot) -> bool:
        try:
            slot.box_prim.get_world_poses()
        except Exception:
            return True
        return False

    def _compute_slot_position(self, index: int) -> tuple[float, float, float]:
        x, ep_y, z = self._endpoint
        return (x, ep_y + index * self._SLOT_STRIDE_Y, z)
