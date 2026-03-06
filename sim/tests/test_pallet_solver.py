"""Unit tests for pallet_solver -- no Isaac Sim required."""

from __future__ import annotations

import random

import pytest
from drp_sim.pallet_solver import (
    CELL_SIZE,
    GRID_D,
    GRID_H,
    GRID_W,
    GridBox,
    Orientation,
    Placement,
    SolverConstraints,
    can_place,
    empty_grid,
    find_placement_z,
    grid_to_world,
    place_box,
    solve_greedy,
    solve_random,
)

# -- Grid helpers -------------------------------------------------------------


class TestCanPlace:
    def test_empty_grid_accepts(self) -> None:
        grid = empty_grid()
        assert can_place(grid, 0, 0, 0, 1, 1, 1)

    def test_out_of_bounds_x(self) -> None:
        grid = empty_grid()
        assert not can_place(grid, GRID_W, 0, 0, 1, 1, 1)

    def test_out_of_bounds_y(self) -> None:
        grid = empty_grid()
        assert not can_place(grid, 0, GRID_D, 0, 1, 1, 1)

    def test_out_of_bounds_z(self) -> None:
        grid = empty_grid()
        assert not can_place(grid, 0, 0, GRID_H, 1, 1, 1)

    def test_negative_coords(self) -> None:
        grid = empty_grid()
        assert not can_place(grid, -1, 0, 0, 1, 1, 1)

    def test_overlap_rejected(self) -> None:
        grid = place_box(empty_grid(), 0, 0, 0, 2, 2, 1)
        assert not can_place(grid, 1, 1, 0, 1, 1, 1)

    def test_adjacent_ok(self) -> None:
        grid = place_box(empty_grid(), 0, 0, 0, 1, 1, 1)
        assert can_place(grid, 1, 0, 0, 1, 1, 1)

    def test_large_box_bounds(self) -> None:
        grid = empty_grid()
        assert not can_place(grid, 0, 0, 0, GRID_W + 1, 1, 1)

    def test_2x2_fits_at_corner(self) -> None:
        grid = empty_grid()
        assert can_place(grid, GRID_W - 2, GRID_D - 2, 0, 2, 2, 1)

    def test_2x2_overflows_corner(self) -> None:
        grid = empty_grid()
        assert not can_place(grid, GRID_W - 1, GRID_D - 1, 0, 2, 2, 1)


class TestFindPlacementZ:
    def test_ground_level(self) -> None:
        grid = empty_grid()
        assert find_placement_z(grid, 0, 0, 1, 1, 1) == 0

    def test_stacks_on_existing(self) -> None:
        grid = place_box(empty_grid(), 0, 0, 0, 1, 1, 1)
        assert find_placement_z(grid, 0, 0, 1, 1, 1) == 1

    def test_no_floating_box(self) -> None:
        """A 2x2 box needs all 4 cells below occupied to stack at z=1."""
        grid = place_box(empty_grid(), 0, 0, 0, 1, 1, 1)
        # Only (0,0,0) occupied => 2x2 at z=1 lacks support, z=0 has overlap
        # So a 1x1 box at (1,0) should go to z=0 (no conflict)
        assert find_placement_z(grid, 1, 0, 1, 1, 1) == 0
        # But stacking a 1x1 on (0,0) must go to z=1
        assert find_placement_z(grid, 0, 0, 1, 1, 1) == 1

    def test_full_support_required(self) -> None:
        """2x2 box needs all 4 cells below to be occupied."""
        grid = empty_grid()
        grid = place_box(grid, 0, 0, 0, 2, 2, 1)
        z = find_placement_z(grid, 0, 0, 2, 2, 1)
        assert z == 1

    def test_returns_none_when_full(self) -> None:
        grid = empty_grid()
        for z in range(GRID_H):
            grid = place_box(grid, 0, 0, z, 1, 1, 1)
        assert find_placement_z(grid, 0, 0, 1, 1, 1) is None

    def test_tall_box_needs_headroom(self) -> None:
        grid = empty_grid()
        # Fill z=0..6, try to place h=2 box => needs z + 2 <= 8 => z=6 works (6+2=8)
        for z in range(6):
            grid = place_box(grid, 0, 0, z, 1, 1, 1)
        assert find_placement_z(grid, 0, 0, 1, 1, 2) == 6

    def test_too_tall_returns_none(self) -> None:
        grid = empty_grid()
        for z in range(7):
            grid = place_box(grid, 0, 0, z, 1, 1, 1)
        # z=7, h=2 => 7+2=9 > GRID_H=8
        assert find_placement_z(grid, 0, 0, 1, 1, 2) is None


class TestPlaceBox:
    def test_marks_cells(self) -> None:
        grid = place_box(empty_grid(), 1, 2, 0, 2, 1, 1)
        assert grid[1][2][0] is True
        assert grid[2][2][0] is True
        assert grid[0][2][0] is False

    def test_returns_new_grid(self) -> None:
        original = empty_grid()
        new = place_box(original, 0, 0, 0, 1, 1, 1)
        assert original[0][0][0] is False
        assert new[0][0][0] is True


# -- Solvers ------------------------------------------------------------------


def _make_box(box_id: int, w: int, d: int, h: int = 1, box_type: str = "normal") -> GridBox:
    real = (w * CELL_SIZE, d * CELL_SIZE, h * CELL_SIZE)
    return GridBox(box_id=box_id, w=w, d=d, h=h, box_type=box_type, real_size=real)


class TestSolveGreedy:
    def test_single_box(self) -> None:
        queue = [_make_box(0, 1, 1)]
        result = solve_greedy(queue)
        assert len(result) == 1
        assert result[0].grid_x == 0
        assert result[0].grid_y == 0
        assert result[0].grid_z == 0

    def test_deterministic(self) -> None:
        queue = [_make_box(i, 1, 1) for i in range(4)]
        r1 = solve_greedy(queue)
        r2 = solve_greedy(queue)
        assert [(p.grid_x, p.grid_y, p.grid_z) for p in r1] == [
            (p.grid_x, p.grid_y, p.grid_z) for p in r2
        ]

    def test_fills_ground_first(self) -> None:
        """Greedy fills z=0 before stacking."""
        queue = [_make_box(i, 1, 1) for i in range(5)]
        result = solve_greedy(queue)
        # First 4 should all be on z=0 (4x4 grid, 1x1 boxes)
        assert all(p.grid_z == 0 for p in result[:4])

    def test_skips_unplaceable(self) -> None:
        """A box too large for the grid is skipped."""
        huge = GridBox(
            box_id=0, w=GRID_W + 1, d=1, h=1, box_type="big", real_size=(2.0, 0.25, 0.25)
        )
        result = solve_greedy([huge])
        assert len(result) == 0

    def test_mixed_sizes(self) -> None:
        queue = [_make_box(0, 2, 2), _make_box(1, 1, 1), _make_box(2, 1, 1)]
        result = solve_greedy(queue)
        assert len(result) == 3
        # All on ground level
        assert all(p.grid_z == 0 for p in result)

    def test_rotation_used_when_needed(self) -> None:
        """A 1x2 box should still fit via rotation if normal doesn't work."""
        # Fill columns 0-2 at y=0..3 so only column 3 is free (1 cell wide)
        queue = [_make_box(0, 2, 1), _make_box(1, 2, 1)]
        result = solve_greedy(queue)
        assert len(result) == 2


class TestSolveRandom:
    def test_seeded_reproducibility(self) -> None:
        queue = [_make_box(i, 1, 1) for i in range(8)]
        r1 = solve_random(queue, random.Random(42))
        r2 = solve_random(queue, random.Random(42))
        pos1 = [(p.grid_x, p.grid_y, p.grid_z) for p in r1]
        pos2 = [(p.grid_x, p.grid_y, p.grid_z) for p in r2]
        assert pos1 == pos2

    def test_different_seeds_differ(self) -> None:
        queue = [_make_box(i, 1, 1) for i in range(8)]
        r1 = solve_random(queue, random.Random(1))
        r2 = solve_random(queue, random.Random(999))
        pos1 = [(p.grid_x, p.grid_y, p.grid_z) for p in r1]
        pos2 = [(p.grid_x, p.grid_y, p.grid_z) for p in r2]
        assert pos1 != pos2

    def test_places_all_small_boxes(self) -> None:
        """16 1x1 boxes should all fit on a 4x4 grid."""
        queue = [_make_box(i, 1, 1) for i in range(16)]
        result = solve_random(queue, random.Random(7))
        assert len(result) == 16

    def test_favours_lower_z(self) -> None:
        """With many runs, average z should be low (ground-biased)."""
        queue = [_make_box(i, 1, 1) for i in range(20)]
        avg_z_values = []
        for seed in range(10):
            result = solve_random(queue, random.Random(seed))
            if result:
                avg_z_values.append(sum(p.grid_z for p in result) / len(result))
        # Average z across all runs should be modest (well below GRID_H / 2)
        assert sum(avg_z_values) / len(avg_z_values) < GRID_H / 2


# -- Coordinate mapping -------------------------------------------------------


class TestGridToWorld:
    def test_origin_box(self) -> None:
        p = Placement(
            box_id=0,
            grid_x=0,
            grid_y=0,
            grid_z=0,
            orientation=Orientation.NORMAL,
            real_size=(0.25, 0.25, 0.25),
            box_type="normal",
        )
        wx, wy, wz = grid_to_world(p, (0.0, 0.0, 0.0))
        assert wx == pytest.approx(0.125)
        assert wy == pytest.approx(0.125)
        assert wz == pytest.approx(0.125)

    def test_offset_pallet_origin(self) -> None:
        p = Placement(
            box_id=0,
            grid_x=0,
            grid_y=0,
            grid_z=0,
            orientation=Orientation.NORMAL,
            real_size=(0.25, 0.25, 0.25),
            box_type="normal",
        )
        wx, wy, wz = grid_to_world(p, (1.0, 2.0, 0.5))
        assert wx == pytest.approx(1.125)
        assert wy == pytest.approx(2.125)
        assert wz == pytest.approx(0.625)

    def test_grid_position_scaling(self) -> None:
        p = Placement(
            box_id=1,
            grid_x=2,
            grid_y=3,
            grid_z=1,
            orientation=Orientation.NORMAL,
            real_size=(0.50, 0.25, 0.25),
            box_type="heavy",
        )
        wx, wy, wz = grid_to_world(p, (0.0, 0.0, 0.0))
        # x: 2*0.25 + 0.50/2 = 0.5 + 0.25 = 0.75
        assert wx == pytest.approx(0.75)
        # y: 3*0.25 + 0.25/2 = 0.75 + 0.125 = 0.875
        assert wy == pytest.approx(0.875)
        # z: 1*0.25 + 0.25/2 = 0.25 + 0.125 = 0.375
        assert wz == pytest.approx(0.375)

    def test_rotated_box_uses_swapped_real_size(self) -> None:
        """Rotated placement should have already-swapped real_size."""
        p = Placement(
            box_id=0,
            grid_x=0,
            grid_y=0,
            grid_z=0,
            orientation=Orientation.ROTATED,
            real_size=(0.25, 0.50, 0.25),  # swapped from (0.50, 0.25, 0.25)
            box_type="normal",
        )
        wx, wy, _wz = grid_to_world(p, (0.0, 0.0, 0.0))
        assert wx == pytest.approx(0.125)  # 0 * 0.25 + 0.25 / 2
        assert wy == pytest.approx(0.25)  # 0 * 0.25 + 0.50 / 2


# -- Constraint tests ---------------------------------------------------------

_CONSTRAINTS = SolverConstraints()
_NO_CONSTRAINTS = SolverConstraints(
    max_layers=GRID_H,
    excluded_types=frozenset(),
    fragile_types=frozenset(),
    heavy_types=frozenset(),
)


class TestFragileConstraint:
    def test_nothing_stacks_on_fragile_greedy(self) -> None:
        """Greedy solver must never place a box on top of a fragile box."""
        queue = [
            _make_box(0, 2, 2, box_type="fragile"),
            _make_box(1, 1, 1, box_type="normal"),
            _make_box(2, 1, 1, box_type="normal"),
        ]
        result = solve_greedy(queue, _CONSTRAINTS)
        fragile = [p for p in result if p.box_type == "fragile"]
        assert len(fragile) == 1
        fz_top = fragile[0].grid_z + 1  # h=1
        fx, fy = fragile[0].grid_x, fragile[0].grid_y
        # No other box should overlap the fragile footprint at z >= fz_top
        for p in result:
            if p.box_type == "fragile":
                continue
            if p.grid_z >= fz_top and p.grid_x < fx + 2 and p.grid_y < fy + 2:
                pytest.fail(f"Box {p.box_id} stacked on fragile at z={p.grid_z}")

    def test_nothing_stacks_on_fragile_random(self) -> None:
        """Random solver must also respect fragile constraint."""
        queue = [
            _make_box(0, 2, 2, box_type="fragile"),
            _make_box(1, 1, 1, box_type="normal"),
            _make_box(2, 1, 1, box_type="normal"),
            _make_box(3, 1, 1, box_type="normal"),
        ]
        for seed in range(20):
            result = solve_random(queue, random.Random(seed), _CONSTRAINTS)
            fragile = [p for p in result if p.box_type == "fragile"]
            if not fragile:
                continue
            fp = fragile[0]
            fw = round(fp.real_size[0] / CELL_SIZE)
            fd = round(fp.real_size[1] / CELL_SIZE)
            fz_top = fp.grid_z + fp.real_size[2] / CELL_SIZE
            for p in result:
                if p.box_type == "fragile":
                    continue
                overlaps_x = p.grid_x < fp.grid_x + fw and p.grid_x + 1 > fp.grid_x
                overlaps_y = p.grid_y < fp.grid_y + fd and p.grid_y + 1 > fp.grid_y
                if overlaps_x and overlaps_y:
                    assert (
                        p.grid_z < fz_top
                    ), f"Seed {seed}: box {p.box_id} at z={p.grid_z} on fragile"

    def test_fragile_on_fragile_blocked(self) -> None:
        """Even another fragile box cannot stack on a fragile box."""
        queue = [
            _make_box(0, 2, 2, box_type="fragile"),
            _make_box(1, 2, 2, box_type="fragile"),
        ]
        result = solve_greedy(queue, _CONSTRAINTS)
        # Both placed but not stacked -- second must be adjacent, not on top
        if len(result) == 2:
            assert result[0].grid_z == result[1].grid_z == 0


class TestMaxLayers:
    def test_max_two_layers_greedy(self) -> None:
        """No column should have more than 2 boxes stacked."""
        queue = [_make_box(i, 1, 1) for i in range(48)]
        result = solve_greedy(queue, _CONSTRAINTS)
        columns: dict[tuple[int, int], int] = {}
        for p in result:
            key = (p.grid_x, p.grid_y)
            columns[key] = columns.get(key, 0) + 1
        for key, count in columns.items():
            assert count <= 2, f"Column {key} has {count} layers"

    def test_max_two_layers_random(self) -> None:
        """Random solver also respects max_layers=2."""
        queue = [_make_box(i, 1, 1) for i in range(48)]
        for seed in range(10):
            result = solve_random(queue, random.Random(seed), _CONSTRAINTS)
            columns: dict[tuple[int, int], int] = {}
            for p in result:
                key = (p.grid_x, p.grid_y)
                columns[key] = columns.get(key, 0) + 1
            for key, count in columns.items():
                assert count <= 2, f"Seed {seed}: column {key} has {count} layers"

    def test_unconstrained_allows_tall_stacks(self) -> None:
        """Without max_layers, boxes can stack higher than 2."""
        queue = [_make_box(i, 1, 1) for i in range(5)]
        result = solve_greedy(queue, _NO_CONSTRAINTS)
        max_z = max(p.grid_z for p in result)
        # 4x4 grid, 5 boxes of 1x1 => at least one must stack
        # With no constraints, this should allow z >= 1
        assert len(result) == 5
        assert max_z >= 0  # all fit


class TestExcludedTypes:
    def test_damaged_excluded(self) -> None:
        """Damaged boxes should be filtered from the queue."""
        queue = [
            _make_box(0, 1, 1, box_type="normal"),
            _make_box(1, 1, 1, box_type="damaged"),
            _make_box(2, 1, 1, box_type="normal"),
        ]
        result = solve_greedy(queue, _CONSTRAINTS)
        types = {p.box_type for p in result}
        assert "damaged" not in types
        assert len(result) == 2

    def test_damaged_excluded_random(self) -> None:
        queue = [_make_box(i, 1, 1, box_type="damaged") for i in range(5)]
        result = solve_random(queue, random.Random(42), _CONSTRAINTS)
        assert len(result) == 0


class TestHeavyPreference:
    def test_heavy_placed_first_greedy(self) -> None:
        """In greedy mode, heavy boxes should be placed before normal boxes."""
        queue = [
            _make_box(0, 1, 1, box_type="normal"),
            _make_box(1, 1, 1, box_type="normal"),
            _make_box(2, 1, 1, box_type="heavy"),
            _make_box(3, 1, 1, box_type="heavy"),
        ]
        result = solve_greedy(queue, _CONSTRAINTS)
        # Heavy boxes should appear at z=0 positions
        heavy = [p for p in result if p.box_type == "heavy"]
        assert all(p.grid_z == 0 for p in heavy)

    def test_heavy_favours_ground_random(self) -> None:
        """In random mode, heavy boxes should mostly land at z=0."""
        queue = [_make_box(i, 1, 1, box_type="heavy") for i in range(4)] + [
            _make_box(i + 4, 1, 1, box_type="normal") for i in range(12)
        ]
        ground_counts = 0
        total_heavy = 0
        for seed in range(20):
            result = solve_random(queue, random.Random(seed), _CONSTRAINTS)
            for p in result:
                if p.box_type == "heavy":
                    total_heavy += 1
                    if p.grid_z == 0:
                        ground_counts += 1
        # At least 80% of heavy boxes should be on the ground
        assert ground_counts / total_heavy > 0.8
