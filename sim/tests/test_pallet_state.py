"""Tests for drp_sim.pallet_state."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from drp_sim.pallet_state import (
    PalletConfig,
    PalletManager,
    PalletState,
    _PlacedBox,
    hide_boxes,
)


@pytest.fixture
def config_2x2() -> PalletConfig:
    return PalletConfig(
        name="test_pallet",
        slots=[(-1.0, -1.0), (-0.5, -1.0), (-1.0, -0.5), (-0.5, -0.5)],
        slot_z=-0.25,
        high_z=0.70,
    )


def _mock_prim() -> MagicMock:
    return MagicMock()


class TestPalletState:
    def test_place_box_advances_slot_idx(self, config_2x2: PalletConfig) -> None:
        state = PalletState(config_2x2)
        assert state.slot_idx == 0
        state.place_box("/box_0", _mock_prim())
        assert state.slot_idx == 1
        state.place_box("/box_1", _mock_prim())
        assert state.slot_idx == 2

    def test_reset_returns_placed_and_resets(self, config_2x2: PalletConfig) -> None:
        state = PalletState(config_2x2)
        state.place_box("/box_0", _mock_prim())
        state.place_box("/box_1", _mock_prim())
        boxes = state.reset()
        assert len(boxes) == 2
        assert boxes[0].prim_path == "/box_0"
        assert boxes[1].prim_path == "/box_1"
        assert state.slot_idx == 0
        assert state.placed_count == 0

    def test_current_slot_xy_cycles(self, config_2x2: PalletConfig) -> None:
        state = PalletState(config_2x2)
        assert state.current_slot_xy == (-1.0, -1.0)
        state.place_box("/b0", _mock_prim())
        assert state.current_slot_xy == (-0.5, -1.0)
        state.place_box("/b1", _mock_prim())
        assert state.current_slot_xy == (-1.0, -0.5)
        state.place_box("/b2", _mock_prim())
        assert state.current_slot_xy == (-0.5, -0.5)
        state.place_box("/b3", _mock_prim())
        assert state.current_slot_xy == (-1.0, -1.0)

    def test_placed_count(self, config_2x2: PalletConfig) -> None:
        state = PalletState(config_2x2)
        assert state.placed_count == 0
        state.place_box("/b0", _mock_prim())
        assert state.placed_count == 1


class TestPalletManager:
    def test_reset_pallet_by_index(self) -> None:
        configs = [
            PalletConfig("p1", [(0, 0)], slot_z=-0.2, high_z=0.7),
            PalletConfig("p2", [(1, 1)], slot_z=-0.3, high_z=0.8),
        ]
        mgr = PalletManager(configs)
        mgr._pallets[1].place_box("/b0", _mock_prim())
        boxes = mgr.reset_pallet(1)
        assert len(boxes) == 1
        assert mgr._pallets[1].slot_idx == 0
        assert mgr._pallets[0].slot_idx == 0

    def test_reset_pallet_none_resets_active(self) -> None:
        configs = [
            PalletConfig("p1", [(0, 0)], slot_z=-0.2, high_z=0.7),
        ]
        mgr = PalletManager(configs)
        mgr.place_box("/b0", _mock_prim())
        boxes = mgr.reset_pallet(None)
        assert len(boxes) == 1
        assert mgr.active_pallet.slot_idx == 0

    def test_index_out_of_range_raises(self) -> None:
        configs = [PalletConfig("p1", [(0, 0)], slot_z=-0.2, high_z=0.7)]
        mgr = PalletManager(configs)
        with pytest.raises(IndexError):
            mgr.reset_pallet(5)

    def test_active_idx_setter_validates(self) -> None:
        configs = [PalletConfig("p1", [(0, 0)], slot_z=-0.2, high_z=0.7)]
        mgr = PalletManager(configs)
        with pytest.raises(IndexError):
            mgr.active_idx = 3

    def test_place_box_delegates_to_active(self) -> None:
        configs = [
            PalletConfig("p1", [(0, 0), (1, 1)], slot_z=-0.2, high_z=0.7),
        ]
        mgr = PalletManager(configs)
        mgr.place_box("/b0", _mock_prim())
        assert mgr.active_pallet.slot_idx == 1
        assert mgr.active_pallet.placed_count == 1

    def test_current_slot_xy_delegates(self) -> None:
        configs = [
            PalletConfig("p1", [(0.1, 0.2), (0.3, 0.4)], slot_z=-0.2, high_z=0.7),
        ]
        mgr = PalletManager(configs)
        assert mgr.current_slot_xy() == (0.1, 0.2)

    def test_slot_z_and_high_z(self) -> None:
        configs = [PalletConfig("p1", [(0, 0)], slot_z=-0.25, high_z=0.70)]
        mgr = PalletManager(configs)
        assert mgr.slot_z() == -0.25
        assert mgr.high_z() == 0.70

    def test_reset_all_resets_all_pallets(self) -> None:
        configs = [
            PalletConfig("p1", [(0, 0), (1, 1)], slot_z=-0.2, high_z=0.7),
            PalletConfig("p2", [(2, 2), (3, 3)], slot_z=-0.3, high_z=0.8),
        ]
        mgr = PalletManager(configs)
        mgr._pallets[0].place_box("/b0", _mock_prim())
        mgr._pallets[0].place_box("/b1", _mock_prim())
        mgr._pallets[1].place_box("/b2", _mock_prim())
        mgr.active_idx = 1

        boxes = mgr.reset_all()

        assert len(boxes) == 3
        assert [b.prim_path for b in boxes] == ["/b0", "/b1", "/b2"]
        assert mgr._pallets[0].slot_idx == 0
        assert mgr._pallets[1].slot_idx == 0
        assert mgr.active_idx == 0

    def test_empty_configs_raises(self) -> None:
        with pytest.raises(ValueError):
            PalletManager([])


class TestHideBoxes:
    def test_hide_boxes_calls_prim_methods(self) -> None:
        prim = _mock_prim()
        boxes = [_PlacedBox(prim_path="/box_0", box_prim=prim)]
        count = hide_boxes(boxes)
        assert count == 1
        prim.set_linear_velocities.assert_called_once()
        prim.set_world_poses.assert_called_once()

    def test_hide_boxes_empty(self) -> None:
        assert hide_boxes([]) == 0
