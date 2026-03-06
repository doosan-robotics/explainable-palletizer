"""Grid-based pallet packing solver.

Pure Python solver with no Isaac Sim dependency.  Boxes are discretised
onto a 4x4xH occupancy grid (1 cell = 0.25 m) and placed using either a
deterministic greedy strategy or a weighted-random strategy for dataset
diversity.

Stacking constraints (configurable via ``SolverConstraints``):
    - **fragile**: nothing may be placed on top of a fragile box
    - **max_layers**: each (x, y) column has a maximum box count
    - **heavy**: heavy boxes are placed first (greedy) or strongly
      weighted toward z = 0 (random)
    - **excluded_types**: certain box types are filtered from the queue

Grid coordinate system:
    - X: width  (0..GRID_W-1), maps to world +X from pallet origin
    - Y: depth  (0..GRID_D-1), maps to world +Y from pallet origin
    - Z: height (0..GRID_H-1), maps to world +Z from pallet origin
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

# -- Constants ----------------------------------------------------------------

CELL_SIZE: float = 0.25  # metres per grid cell
GRID_W: int = 4  # 1.0 m / 0.25 m
GRID_D: int = 4
GRID_H: int = 8  # max ~2.0 m stack height


# -- Types --------------------------------------------------------------------


class Orientation(Enum):
    """Box orientation on the pallet."""

    NORMAL = "normal"
    ROTATED = "rotated"  # swaps width and depth


@dataclass(frozen=True, slots=True)
class GridBox:
    """A box expressed in grid-cell dimensions.

    ``w`` and ``d`` are in cells (1 or 2 for 0.25 m / 0.50 m boxes).
    ``real_size`` holds the original metric dimensions (x, y, z) in metres.
    """

    box_id: int
    w: int  # grid cells along X
    d: int  # grid cells along Y
    h: int  # grid cells along Z
    box_type: str
    real_size: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Placement:
    """A solved box placement on the pallet grid."""

    box_id: int
    grid_x: int
    grid_y: int
    grid_z: int
    orientation: Orientation
    real_size: tuple[float, float, float]
    box_type: str


@dataclass(frozen=True, slots=True)
class SolverConstraints:
    """Pallet stacking rules.

    Parameters
    ----------
    max_layers:
        Maximum number of boxes stacked in any (x, y) column.
    excluded_types:
        Box types filtered from the queue before solving.
    fragile_types:
        Types whose upper surface is blocked -- nothing may stack on top.
    heavy_types:
        Types that should be placed as low as possible.
    """

    max_layers: int = 2
    excluded_types: frozenset[str] = field(default_factory=lambda: frozenset({"damaged"}))
    fragile_types: frozenset[str] = field(default_factory=lambda: frozenset({"fragile"}))
    heavy_types: frozenset[str] = field(default_factory=lambda: frozenset({"heavy"}))


# -- Grid helpers -------------------------------------------------------------

Grid = list[list[list[bool]]]
"""3-D occupancy grid: grid[x][y][z] == True means occupied."""

LayerGrid = list[list[int]]
"""2-D layer count: layers[x][y] == number of boxes in that column."""


def empty_grid() -> Grid:
    """Create an empty GRID_W x GRID_D x GRID_H occupancy grid."""
    return [[[False] * GRID_H for _ in range(GRID_D)] for _ in range(GRID_W)]


def empty_layer_grid() -> LayerGrid:
    """Create a zeroed GRID_W x GRID_D layer count grid."""
    return [[0] * GRID_D for _ in range(GRID_W)]


def _copy_grid(grid: Grid) -> Grid:
    return [[[grid[x][y][z] for z in range(GRID_H)] for y in range(GRID_D)] for x in range(GRID_W)]


def can_place(grid: Grid, x: int, y: int, z: int, w: int, d: int, h: int) -> bool:
    """Check whether a box of size (w, d, h) fits at position (x, y, z).

    Returns ``False`` if out of bounds or any cell is already occupied.
    """
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > GRID_W or y + d > GRID_D or z + h > GRID_H:
        return False
    for gx in range(x, x + w):
        for gy in range(y, y + d):
            for gz in range(z, z + h):
                if grid[gx][gy][gz]:
                    return False
    return True


def _not_blocked(blocked: Grid, x: int, y: int, z: int, w: int, d: int, h: int) -> bool:
    """Return True if none of the target cells are blocked (above fragile)."""
    for gx in range(x, x + w):
        for gy in range(y, y + d):
            for gz in range(z, z + h):
                if blocked[gx][gy][gz]:
                    return False
    return True


def _within_layer_limit(layers: LayerGrid, x: int, y: int, w: int, d: int, max_layers: int) -> bool:
    """Return True if all columns in the footprint have room for another box."""
    for gx in range(x, x + w):
        for gy in range(y, y + d):
            if layers[gx][gy] >= max_layers:
                return False
    return True


def find_placement_z(grid: Grid, x: int, y: int, w: int, d: int, h: int) -> int | None:
    """Find the lowest z where a box can rest with full support.

    At z == 0 the box sits on the pallet surface (always supported).
    At z > 0 every cell in the footprint must have an occupied cell
    directly below (z - 1) to provide support.

    Returns ``None`` if no valid z exists within grid bounds.
    """
    for z in range(GRID_H):
        if z + h > GRID_H:
            return None
        # Check support: z == 0 is the pallet surface (always OK)
        if z > 0:
            supported = all(grid[gx][gy][z - 1] for gx in range(x, x + w) for gy in range(y, y + d))
            if not supported:
                continue
        if can_place(grid, x, y, z, w, d, h):
            return z
    return None


def _find_placement_z_constrained(
    grid: Grid, blocked: Grid, x: int, y: int, w: int, d: int, h: int
) -> int | None:
    """Like ``find_placement_z`` but also checks the blocked grid."""
    for z in range(GRID_H):
        if z + h > GRID_H:
            return None
        if z > 0:
            supported = all(grid[gx][gy][z - 1] for gx in range(x, x + w) for gy in range(y, y + d))
            if not supported:
                continue
        if can_place(grid, x, y, z, w, d, h) and _not_blocked(blocked, x, y, z, w, d, h):
            return z
    return None


def place_box(grid: Grid, x: int, y: int, z: int, w: int, d: int, h: int) -> Grid:
    """Return a new grid with cells (x..x+w, y..y+d, z..z+h) marked occupied."""
    new = _copy_grid(grid)
    for gx in range(x, x + w):
        for gy in range(y, y + d):
            for gz in range(z, z + h):
                new[gx][gy][gz] = True
    return new


def _block_above(blocked: Grid, x: int, y: int, z_top: int, w: int, d: int) -> None:
    """Mark all cells above a fragile box as blocked (mutates *blocked*)."""
    for gx in range(x, x + w):
        for gy in range(y, y + d):
            for gz in range(z_top, GRID_H):
                blocked[gx][gy][gz] = True


def _increment_layers(layers: LayerGrid, x: int, y: int, w: int, d: int) -> None:
    """Increment layer count for all columns in the footprint (mutates *layers*)."""
    for gx in range(x, x + w):
        for gy in range(y, y + d):
            layers[gx][gy] += 1


# -- Solvers ------------------------------------------------------------------

_DEFAULT_CONSTRAINTS = SolverConstraints()


def _valid_placements_constrained(
    grid: Grid,
    blocked: Grid,
    layers: LayerGrid,
    box: GridBox,
    constraints: SolverConstraints,
) -> list[tuple[int, int, int, int, int, Orientation]]:
    """Enumerate valid placements respecting blocked cells and layer limits."""
    results: list[tuple[int, int, int, int, int, Orientation]] = []
    orientations = [(box.w, box.d, Orientation.NORMAL)]
    if box.w != box.d:
        orientations.append((box.d, box.w, Orientation.ROTATED))

    for w, d, ori in orientations:
        h = box.h
        for x in range(GRID_W - w + 1):
            for y in range(GRID_D - d + 1):
                if not _within_layer_limit(layers, x, y, w, d, constraints.max_layers):
                    continue
                z = _find_placement_z_constrained(grid, blocked, x, y, w, d, h)
                if z is not None:
                    results.append((x, y, z, w, d, ori))
    return results


def _make_placement(box: GridBox, x: int, y: int, z: int, ori: Orientation) -> Placement:
    rx, ry, rz = box.real_size
    real = (ry, rx, rz) if ori is Orientation.ROTATED else (rx, ry, rz)
    return Placement(
        box_id=box.box_id,
        grid_x=x,
        grid_y=y,
        grid_z=z,
        orientation=ori,
        real_size=real,
        box_type=box.box_type,
    )


def _prepare_queue(queue: list[GridBox], constraints: SolverConstraints) -> list[GridBox]:
    """Filter excluded types from the queue."""
    return [b for b in queue if b.box_type not in constraints.excluded_types]


def solve_greedy(
    queue: list[GridBox], constraints: SolverConstraints | None = None
) -> list[Placement]:
    """Place boxes greedily: lowest z first, then smallest x, then smallest y.

    When constraints are provided:
    - Excluded types are filtered out.
    - Heavy boxes are placed first so they occupy ground positions.
    - Fragile boxes block all cells above them.
    - Each column respects the max layer limit.

    Boxes that cannot be placed are silently skipped.
    """
    c = constraints or _DEFAULT_CONSTRAINTS
    work = _prepare_queue(queue, c)

    # Sort: heavy first, then by original order
    work = sorted(work, key=lambda b: (0 if b.box_type in c.heavy_types else 1, b.box_id))

    grid = empty_grid()
    blocked = empty_grid()
    layers = empty_layer_grid()
    placements: list[Placement] = []

    for box in work:
        candidates = _valid_placements_constrained(grid, blocked, layers, box, c)
        if not candidates:
            continue
        candidates.sort(key=lambda cand: (cand[2], cand[0], cand[1]))
        x, y, z, w, d, ori = candidates[0]
        grid = place_box(grid, x, y, z, w, d, box.h)
        _increment_layers(layers, x, y, w, d)

        if box.box_type in c.fragile_types:
            _block_above(blocked, x, y, z + box.h, w, d)

        placements.append(_make_placement(box, x, y, z, ori))

    return placements


def solve_random(
    queue: list[GridBox],
    rng: random.Random,
    constraints: SolverConstraints | None = None,
) -> list[Placement]:
    """Place boxes with weighted-random selection favouring lower z.

    When constraints are provided the same rules as ``solve_greedy`` apply.
    Heavy boxes receive a 10x weight bonus for z = 0 placements to keep
    them at the bottom of the stack.

    Boxes that cannot be placed are silently skipped.
    """
    c = constraints or _DEFAULT_CONSTRAINTS
    work = _prepare_queue(queue, c)

    grid = empty_grid()
    blocked = empty_grid()
    layers = empty_layer_grid()
    placements: list[Placement] = []

    for box in work:
        candidates = _valid_placements_constrained(grid, blocked, layers, box, c)
        if not candidates:
            continue

        weights = [1.0 / (cand[2] + 1) for cand in candidates]

        # Heavy boxes: 10x preference for ground level
        if box.box_type in c.heavy_types:
            weights = [
                w * 10.0 if cand[2] == 0 else w for w, cand in zip(weights, candidates, strict=True)
            ]

        (x, y, z, w, d, ori) = rng.choices(candidates, weights=weights, k=1)[0]
        grid = place_box(grid, x, y, z, w, d, box.h)
        _increment_layers(layers, x, y, w, d)

        if box.box_type in c.fragile_types:
            _block_above(blocked, x, y, z + box.h, w, d)

        placements.append(_make_placement(box, x, y, z, ori))

    return placements


# -- Coordinate mapping -------------------------------------------------------


def grid_to_world(
    placement: Placement,
    pallet_origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert a grid placement to a world-space box centre.

    The box centre is offset by half the real size so the box corner
    aligns with the grid cell edge at ``pallet_origin + grid_pos * CELL_SIZE``.
    """
    ox, oy, oz = pallet_origin
    rx, ry, rz = placement.real_size
    world_x = ox + placement.grid_x * CELL_SIZE + rx / 2
    world_y = oy + placement.grid_y * CELL_SIZE + ry / 2
    world_z = oz + placement.grid_z * CELL_SIZE + rz / 2
    return (world_x, world_y, world_z)
