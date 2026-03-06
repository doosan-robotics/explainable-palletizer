"""Unit tests for ConveyorBuffer -- no Isaac Sim required.

Tests conveyor-based filling with snap threshold, compaction toward -Y,
kinematic toggle, and dead-prim eviction using a mock BoxSpawner.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from drp_sim.conveyor_buffer import BufferSlot, ConveyorBuffer

_ENDPOINT = (0.59, -0.75, -0.20)
_VELOCITY = (0.0, -0.15, 0.0)
_SNAP_MARGIN = 0.05
_STRIDE = ConveyorBuffer._SLOT_STRIDE_Y


def _make_mock_spawner(y_length: float = 0.25) -> MagicMock:
    """Create a mock BoxSpawner that tracks spawns."""
    spawner = MagicMock()
    spawner.auto_spawn_enabled = True
    spawner.box_ttl_steps = None
    spawner.boxes = []
    spawner.box_metadata = []
    spawner._boxes = []

    def fake_spawn(position=None, velocity=None):
        box = MagicMock()
        box.name = f"box_{len(spawner.boxes)}"
        pos = np.array([0.59, 3.01, -0.20], dtype=float)
        box.get_world_poses.return_value = (pos[np.newaxis], None)

        def _set_poses(positions=None, orientations=None):
            if positions is not None:
                box.get_world_poses.return_value = (positions, orientations)

        box.set_world_poses.side_effect = _set_poses
        # Mock prim attribute for kinematic toggle
        box.prim = MagicMock()
        path = f"/World/box_{len(spawner.boxes)}"
        spawner.boxes.append(box)
        spawner._boxes.append((box, path, 0))
        meta = {"size": [0.25, y_length, 0.25], "weight": 5.0, "type": "normal"}
        spawner.box_metadata.append(meta)
        return path

    spawner.spawn.side_effect = fake_spawn
    return spawner


def _move_box_to_y(box_mock: MagicMock, y: float) -> None:
    """Simulate a box reaching a Y position on the conveyor."""
    pos = np.array([[0.59, y, -0.20]])
    box_mock.get_world_poses.return_value = (pos, None)


def _slot_snap_y(buf: ConveyorBuffer, slot_idx: int) -> float:
    """Compute the Y threshold at which a box snaps to the given slot."""
    return buf._compute_slot_position(slot_idx)[1]


def _fill_via_conveyor(buf: ConveyorBuffer) -> None:
    """Drive the buffer step loop until all pending slots are filled."""
    for _ in range(100):
        if buf._in_transit is None and not buf._pending_slots:
            break
        if buf._in_transit is not None:
            y = _slot_snap_y(buf, buf._in_transit.target_slot)
            _move_box_to_y(buf._in_transit.box_prim, y)
        buf.step()


class TestConveyorBufferInit:
    def test_initial_state(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(spawner, endpoint=_ENDPOINT, length=3)
        assert buf.slot_count == 3
        assert buf.occupied_count == 0
        assert buf.active is False
        assert buf.buffer_boxes == []
        assert buf.slot_states == [0, 0, 0]


class TestFill:
    def test_fill_defers_dispatch_to_step(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        result = buf.fill()

        assert result["status"] == "filling"
        assert result["capacity"] == 3
        assert spawner.spawn.call_count == 0
        assert buf._in_transit is None

        buf.step()
        assert spawner.spawn.call_count == 1
        assert buf._in_transit is not None

    def test_fill_populates_all_slots(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        _fill_via_conveyor(buf)

        assert buf.occupied_count == 3
        assert buf.slot_states == [1, 1, 1]
        assert spawner.spawn.call_count == 3

    def test_fill_disables_auto_spawn_and_ttl(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        assert spawner.auto_spawn_enabled is False
        assert spawner.box_ttl_steps is None

    def test_fill_activates_buffer(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        assert buf.active is True


class TestFillOrder:
    def test_fills_from_minus_y_first(self) -> None:
        """Slot 0 (most -Y) fills first, then 1, then 2."""
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()

        buf.step()
        assert buf._in_transit.target_slot == 0

        _move_box_to_y(buf._in_transit.box_prim, _slot_snap_y(buf, 0))
        buf.step()
        assert buf.slot_states == [1, 0, 0]

        assert buf._in_transit.target_slot == 1

    def test_full_fill_order_1_0_0_to_1_1_1(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()

        # [1,0,0]
        buf.step()
        _move_box_to_y(buf._in_transit.box_prim, _slot_snap_y(buf, 0))
        buf.step()
        assert buf.slot_states == [1, 0, 0]

        # [1,1,0]
        _move_box_to_y(buf._in_transit.box_prim, _slot_snap_y(buf, 1))
        buf.step()
        assert buf.slot_states == [1, 1, 0]

        # [1,1,1]
        _move_box_to_y(buf._in_transit.box_prim, _slot_snap_y(buf, 2))
        buf.step()
        assert buf.slot_states == [1, 1, 1]


class TestConveyorArrival:
    def test_box_rides_conveyor_until_snap(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=1,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        buf.step()

        # Still above threshold (well above slot 0 position)
        _move_box_to_y(buf._in_transit.box_prim, 2.0)
        buf.step()
        assert buf.occupied_count == 0

        # At slot position -> triggers snap
        _move_box_to_y(buf._in_transit.box_prim, _slot_snap_y(buf, 0))
        buf.step()
        assert buf.occupied_count == 1
        assert buf._in_transit is None

    def test_position_updated_during_transit(self) -> None:
        """Kinematic translation: set_world_poses called each step to move box."""
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=1,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        buf.step()

        transit = buf._in_transit
        _move_box_to_y(transit.box_prim, 2.0)
        buf.step()

        transit.box_prim.set_world_poses.assert_called()

    def test_overshoot_guard_snaps_instead_of_overshooting(self) -> None:
        """If next position would cross threshold, snap immediately."""
        spawner = _make_mock_spawner()
        # Use a very fast velocity so one step overshoots the slot
        fast_vel = (0.0, -50.0, 0.0)
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=1,
            conveyor_velocity=fast_vel,
        )
        buf.fill()
        buf.step()

        # Place box just above the snap threshold
        slot_y = _slot_snap_y(buf, 0)
        _move_box_to_y(buf._in_transit.box_prim, slot_y + 0.10)
        buf.step()

        # Should have snapped to slot instead of overshooting
        assert buf.occupied_count == 1
        assert buf._in_transit is None

    def test_lost_box_requeues_slot(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=1,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        buf.step()

        buf._in_transit.box_prim.get_world_poses.side_effect = RuntimeError("gone")
        buf.step()

        assert buf._in_transit is not None
        assert spawner.spawn.call_count == 2

    def test_box_snapped_to_exact_slot_position(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=1,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        buf.step()

        box = buf._in_transit.box_prim
        _move_box_to_y(box, _slot_snap_y(buf, 0))
        buf.step()

        expected = buf._compute_slot_position(0)
        placed_pos = box.set_world_poses.call_args.kwargs["positions"]
        np.testing.assert_allclose(placed_pos[0], expected, atol=1e-6)

    def test_identity_rotation_on_snap(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=1,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        buf.step()

        box = buf._in_transit.box_prim
        _move_box_to_y(box, _slot_snap_y(buf, 0))
        buf.step()

        orient = box.set_world_poses.call_args.kwargs["orientations"]
        np.testing.assert_allclose(orient, [[1, 0, 0, 0]], atol=1e-6)

    def test_box_removed_from_spawner_on_arrival(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=1,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        buf.step()

        path = buf._in_transit.prim_path
        _move_box_to_y(buf._in_transit.box_prim, _slot_snap_y(buf, 0))
        buf.step()

        assert not any(entry[1] == path for entry in spawner._boxes)


class TestKinematic:
    @patch("drp_sim.conveyor_buffer.ConveyorBuffer._set_kinematic")
    def test_kinematic_enabled_on_snap(self, mock_kin) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=1,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        buf.step()

        box = buf._in_transit.box_prim
        _move_box_to_y(box, _slot_snap_y(buf, 0))
        buf.step()

        mock_kin.assert_any_call(box, enabled=True)

    @patch("drp_sim.conveyor_buffer.ConveyorBuffer._set_kinematic")
    def test_kinematic_disabled_on_release(self, mock_kin) -> None:
        """Kinematic is disabled via release_box(), not pop_box_at()."""
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=1,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        _fill_via_conveyor(buf)

        prim, _ = buf.pop_nearest_box()

        # pop_box_at no longer disables kinematic (Fabric sync safety).
        # Only enabled=True calls from dispatch/snap should exist.
        assert not any(c == ((prim,), {"enabled": False}) for c in mock_kin.call_args_list)

        # release_box disables kinematic after pick is complete
        buf.release_box(prim)
        mock_kin.assert_called_with(prim, enabled=False)


class TestCompaction:
    def test_compact_shifts_boxes_toward_minus_y(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        _fill_via_conveyor(buf)

        buf.pop_nearest_box()
        assert buf.slot_states == [0, 1, 1]

        buf.fill()
        assert buf.slot_states == [1, 1, 0]

    def test_compact_updates_assigned_positions(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        _fill_via_conveyor(buf)

        buf.pop_nearest_box()
        buf.fill()

        assert buf._slots[0].assigned_position[1] == pytest.approx(_ENDPOINT[1])
        assert buf._slots[1].assigned_position[1] == pytest.approx(
            _ENDPOINT[1] + _STRIDE,
        )

    def test_refill_after_pick(self) -> None:
        """pick -> fill -> compact -> new box fills +Y-most empty slot."""
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        _fill_via_conveyor(buf)
        initial_spawns = spawner.spawn.call_count

        buf.pop_nearest_box()
        buf.fill()
        _fill_via_conveyor(buf)

        assert buf.slot_states == [1, 1, 1]
        assert spawner.spawn.call_count == initial_spawns + 1

    def test_compact_drops_dead_prims(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        _fill_via_conveyor(buf)

        buf._slots[1].box_prim.get_world_poses.side_effect = RuntimeError("gone")
        buf.fill()

        alive_count = sum(1 for s in buf._slots if s is not None)
        assert alive_count == 2

    def test_compact_noop_when_already_packed(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        _fill_via_conveyor(buf)

        pos_before = [s.assigned_position for s in buf._slots]
        buf.fill()
        pos_after = [s.assigned_position for s in buf._slots]
        assert pos_before == pos_after


class TestSlotPositionCalculation:
    def test_first_slot_at_endpoint(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(spawner, endpoint=_ENDPOINT, length=3)
        pos = buf._compute_slot_position(0)
        assert pos[1] == pytest.approx(_ENDPOINT[1])

    def test_outermost_slot_center(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(spawner, endpoint=_ENDPOINT, length=3)
        pos = buf._compute_slot_position(2)
        assert pos[1] == pytest.approx(_ENDPOINT[1] + 2 * _STRIDE)

    def test_fixed_stride_spacing(self) -> None:
        spawner = _make_mock_spawner(y_length=0.50)
        origin = (0.0, 0.0, 0.0)
        buf = ConveyorBuffer(
            spawner,
            endpoint=origin,
            length=3,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        _fill_via_conveyor(buf)

        y_positions = [buf._slots[i].assigned_position[1] for i in range(3)]
        assert y_positions[0] == pytest.approx(0.0)
        assert y_positions[1] == pytest.approx(_STRIDE)
        assert y_positions[2] == pytest.approx(2 * _STRIDE)


class TestEviction:
    def test_dead_prim_evicted_and_replaced(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=2,
            conveyor_velocity=_VELOCITY,
        )
        buf.fill()
        _fill_via_conveyor(buf)
        initial_spawns = spawner.spawn.call_count

        buf._slots[0].box_prim.get_world_poses.side_effect = RuntimeError("dead")
        buf.step()

        assert buf._slots[0] is None
        assert spawner.spawn.call_count == initial_spawns + 1
        assert buf._in_transit is not None
        assert buf._in_transit.target_slot == 0


class TestConsumedDetection:
    def test_accessible_prim_not_consumed(self) -> None:
        box_prim = MagicMock()
        pos = np.array([[0.59, -0.75, -0.20]])
        box_prim.get_world_poses.return_value = (pos, None)

        slot = BufferSlot(
            box_prim=box_prim,
            prim_path="/World/box_0",
            y_length=0.25,
            box_half_h=0.125,
            assigned_position=(0.59, -0.75, -0.20),
        )
        assert ConveyorBuffer._is_consumed(slot) is False

    def test_exception_treated_as_consumed(self) -> None:
        box_prim = MagicMock()
        box_prim.get_world_poses.side_effect = RuntimeError("prim gone")

        slot = BufferSlot(
            box_prim=box_prim,
            prim_path="/World/box_0",
            y_length=0.25,
            box_half_h=0.125,
            assigned_position=(0.59, -0.75, -0.20),
        )
        assert ConveyorBuffer._is_consumed(slot) is True


class TestStepMisc:
    def test_step_noop_when_inactive(self) -> None:
        spawner = _make_mock_spawner()
        buf = ConveyorBuffer(
            spawner,
            endpoint=_ENDPOINT,
            length=2,
            conveyor_velocity=_VELOCITY,
        )
        buf.step()
        assert spawner.spawn.call_count == 0
