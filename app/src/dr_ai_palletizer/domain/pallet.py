"""3D pallet state management and position-finding algorithm.

Manages a 4d x 4d x 4d occupancy grid per pallet. Provides functions to
find valid placement positions for a given box shape, place boxes, and
query fill fraction.

Box rotation on even layers
  - Even z-layers (z=1, z=3, ...) rotate 90 degrees (swap w <-> ln)
    for brick-wall interlocking stability, matching training data.
  - find_valid_positions and place_box apply the rotation automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from dr_ai_palletizer.domain.models import (
    PALLET_GRID_SIZE,
    BoxShape,
)

# ---------------------------------------------------------------------------
# Placed box record
# ---------------------------------------------------------------------------


class PlacedBox(NamedTuple):
    box_id: int
    shape: BoxShape
    x: int
    y: int
    z: int
    weight_kg: float


# ---------------------------------------------------------------------------
# Pallet state
# ---------------------------------------------------------------------------


@dataclass
class PalletState:
    """Immutable-style 3D pallet with occupancy grid.

    Grid axes: x (width, left-right), y (length, top-bottom), z (height).
    Cell value 0 means empty; any other value is a placed box ID.
    """

    id: int
    grid: np.ndarray  # shape (GRID, GRID, GRID), dtype uint16
    placed_boxes: dict[int, PlacedBox] = field(default_factory=dict)
    total_weight_kg: float = 0.0
    max_weight_kg: float = 500.0
    _next_box_id: int = 1

    @staticmethod
    def empty(pallet_id: int, max_weight_kg: float = 500.0) -> PalletState:
        """Create a fresh empty pallet."""
        grid = np.zeros(
            (PALLET_GRID_SIZE, PALLET_GRID_SIZE, PALLET_GRID_SIZE),
            dtype=np.uint16,
        )
        return PalletState(id=pallet_id, grid=grid, max_weight_kg=max_weight_kg)


# ---------------------------------------------------------------------------
# Filling strategy helpers
# ---------------------------------------------------------------------------


def _layer_is_even(z: int) -> bool:
    """Return True if z corresponds to an even layer (2nd, 4th ...).

    Layer numbering is 1-indexed: z=0 is layer 1 (odd), z=1 is layer 2 (even).
    """
    return z % 2 == 1


def effective_shape(shape: BoxShape, z: int) -> BoxShape:
    """Return the shape to use at a given z-level.

    Even layers rotate 90 degrees (swap w and ln) for interlocking stability.
    """
    if _layer_is_even(z):
        return BoxShape(w=shape.ln, ln=shape.w, h=shape.h)
    return shape


def _compute_available_origins(grid: np.ndarray, z: int) -> set[tuple[int, int]]:
    """Compute (x, y) origins unlocked by the wave-front at z-layer."""
    unlocked: set[tuple[int, int]] = {(0, 0)}
    g = grid.shape[0]

    layer_slice = grid[:, :, z]
    box_ids = [int(v) for v in np.unique(layer_slice) if v != 0]

    for bid in box_ids:
        xs, ys = np.where(layer_slice == bid)
        bx, by = int(xs.min()), int(ys.min())
        bw = int(xs.max()) - bx + 1
        bln = int(ys.max()) - by + 1

        rx = bx + bw
        if rx < g:
            unlocked.add((rx, by))

        dy = by + bln
        if dy < g:
            unlocked.add((bx, dy))

    return unlocked


def _fill_sort_key(pos: tuple[int, int, int]) -> tuple[int, int, int]:
    """Sort key that implements layer-alternating fill direction."""
    x, y, z = pos
    if _layer_is_even(z):
        return (z, y, x)
    return (z, x, y)


# ---------------------------------------------------------------------------
# Position finding
# ---------------------------------------------------------------------------


def find_valid_positions(
    pallet: PalletState,
    shape: BoxShape,
    weight_kg: float,
) -> list[tuple[int, int, int]]:
    """Find valid (x, y, z) positions using wave-front expansion.

    On even layers the box shape is rotated 90 degrees (w <-> ln swap)
    to match the interlocking pattern used during training.
    """
    if pallet.total_weight_kg + weight_kg > pallet.max_weight_kg:
        return []

    g = PALLET_GRID_SIZE
    h = shape.h
    positions: list[tuple[int, int, int]] = []

    for z in range(g - h + 1):
        eff = effective_shape(shape, z)
        w, ln = eff.w, eff.ln
        available = _compute_available_origins(pallet.grid, z)

        for x, y in available:
            if x + w > g or y + ln > g:
                continue

            region = pallet.grid[x : x + w, y : y + ln, z : z + h]
            if np.any(region != 0):
                continue

            if z > 0:
                support = pallet.grid[x : x + w, y : y + ln, z - 1]
                if not np.all(support != 0):
                    continue

            positions.append((x, y, z))

    positions.sort(key=_fill_sort_key)
    return positions


# ---------------------------------------------------------------------------
# Box placement (returns new state)
# ---------------------------------------------------------------------------


def place_box(
    pallet: PalletState,
    shape: BoxShape,
    pos: tuple[int, int, int],
    weight_kg: float,
) -> PalletState:
    """Place a box and return a new PalletState (does not mutate the original).

    Applies even-layer rotation automatically so the grid records the
    actual (possibly rotated) footprint.
    """
    x, y, z = pos
    eff = effective_shape(shape, z)
    w, ln, h = eff

    new_grid = pallet.grid.copy()
    region = new_grid[x : x + w, y : y + ln, z : z + h]
    if np.any(region != 0):
        import logging

        logging.getLogger(__name__).warning(
            "place_box: cells already occupied at (%d,%d,%d) shape=%s, skipping",
            x,
            y,
            z,
            eff,
        )
        return pallet

    box_id = pallet._next_box_id
    new_grid[x : x + w, y : y + ln, z : z + h] = box_id

    placed = PlacedBox(
        box_id=box_id,
        shape=eff,
        x=x,
        y=y,
        z=z,
        weight_kg=weight_kg,
    )
    new_placed = {**pallet.placed_boxes, box_id: placed}

    return PalletState(
        id=pallet.id,
        grid=new_grid,
        placed_boxes=new_placed,
        total_weight_kg=pallet.total_weight_kg + weight_kg,
        max_weight_kg=pallet.max_weight_kg,
        _next_box_id=box_id + 1,
    )


# ---------------------------------------------------------------------------
# Fill fraction
# ---------------------------------------------------------------------------


def fill_fraction(pallet: PalletState) -> float:
    """Return the fraction of occupied cells (0.0 to 1.0)."""
    total = pallet.grid.size
    occupied = int(np.count_nonzero(pallet.grid))
    return occupied / total
