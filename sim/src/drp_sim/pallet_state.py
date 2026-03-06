"""Per-pallet box tracking and reset logic.

Manages placed-box state for each pallet and provides a reset operation
that hides all placed boxes (teleport + MakeInvisible).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PalletConfig:
    """Immutable description of a pallet's slot layout."""

    name: str
    slots: list[tuple[float, float]]
    slot_z: float
    high_z: float


@dataclass(slots=True)
class _PlacedBox:
    """A box placed on a pallet slot."""

    prim_path: str
    box_prim: object  # RigidPrim at runtime


class PalletState:
    """Tracks placed boxes and current slot index for a single pallet."""

    def __init__(self, config: PalletConfig) -> None:
        self.config = config
        self._slot_idx: int = 0
        self._placed: list[_PlacedBox] = []

    @property
    def slot_idx(self) -> int:
        return self._slot_idx

    @property
    def placed_count(self) -> int:
        return len(self._placed)

    @property
    def current_slot_xy(self) -> tuple[float, float]:
        return self.config.slots[self._slot_idx % len(self.config.slots)]

    def place_box(self, prim_path: str, box_prim: object) -> None:
        """Record a placed box and advance the slot index."""
        self._placed.append(_PlacedBox(prim_path=prim_path, box_prim=box_prim))
        self._slot_idx += 1

    def reset(self) -> list[_PlacedBox]:
        """Return all placed boxes and reset slot index to 0."""
        boxes = list(self._placed)
        self._placed.clear()
        self._slot_idx = 0
        return boxes


class PalletManager:
    """Manages multiple pallets with an active-pallet cursor."""

    def __init__(self, configs: list[PalletConfig]) -> None:
        if not configs:
            raise ValueError("At least one PalletConfig required")
        self._pallets: list[PalletState] = [PalletState(c) for c in configs]
        self._active_idx: int = 0

    @property
    def active_pallet(self) -> PalletState:
        return self._pallets[self._active_idx]

    @property
    def active_idx(self) -> int:
        return self._active_idx

    @active_idx.setter
    def active_idx(self, value: int) -> None:
        if not 0 <= value < len(self._pallets):
            raise IndexError(f"Pallet index {value} out of range [0, {len(self._pallets)})")
        self._active_idx = value

    @property
    def pallet_count(self) -> int:
        return len(self._pallets)

    def place_box(self, prim_path: str, box_prim: object) -> None:
        """Place a box on the active pallet's current slot."""
        self.active_pallet.place_box(prim_path, box_prim)

    def current_slot_xy(self) -> tuple[float, float]:
        """Return the active pallet's current slot (x, y)."""
        return self.active_pallet.current_slot_xy

    def slot_z(self) -> float:
        """Return the active pallet's box-center z."""
        return self.active_pallet.config.slot_z

    def high_z(self) -> float:
        """Return the active pallet's high waypoint z."""
        return self.active_pallet.config.high_z

    def reset_all(self) -> list[_PlacedBox]:
        """Reset ALL pallets and return all placed boxes. Resets active index to 0."""
        all_boxes: list[_PlacedBox] = []
        for pallet in self._pallets:
            all_boxes.extend(pallet.reset())
        self._active_idx = 0
        return all_boxes

    def reset_pallet(self, index: int | None = None) -> list[_PlacedBox]:
        """Reset a pallet: return placed boxes and reset slot index.

        Args:
            index: 0-based pallet index, or ``None`` for the active pallet.
        """
        idx = self._active_idx if index is None else index
        if not 0 <= idx < len(self._pallets):
            raise IndexError(f"Pallet index {idx} out of range [0, {len(self._pallets)})")
        return self._pallets[idx].reset()


def hide_boxes(placed: list[_PlacedBox]) -> int:
    """Teleport boxes to z=-1000 and zero velocities.

    Avoids any USD/UsdGeom calls (MakeInvisible, ComputeWorldBound, etc.)
    because they trigger Fabric prototype cascade invalidation that crashes
    cuRobo's collision checker.
    """
    import numpy as np

    hidden = 0
    for pb in placed:
        try:
            pb.box_prim.set_linear_velocities(np.zeros((1, 3)))
            pb.box_prim.set_angular_velocities(np.zeros((1, 3)))
            pb.box_prim.set_world_poses(positions=np.array([[0.0, 0.0, -1000.0]]))
        except Exception:
            pass
        hidden += 1
    return hidden
