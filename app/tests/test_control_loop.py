"""Tests for the ControlLoop state machine."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from dr_ai_palletizer.control_loop import ControlLoop


def _make_deps() -> tuple[MagicMock, MagicMock, AsyncMock]:
    """Create mock sim_client, inference_client, and broadcast_event."""
    sim = MagicMock()
    sim.get_pick_positions = AsyncMock(
        return_value=[
            {"x": 0.59, "y": 3.01, "z": -0.20},
            {"x": 0.59, "y": 2.51, "z": -0.20},
            {"x": 0.59, "y": 2.01, "z": -0.20},
        ]
    )
    sim.get_pallet_centers = AsyncMock(
        return_value=[
            {"x": 0.80, "y": -0.10, "z": 0.0},
            {"x": 0.80, "y": 0.40, "z": 0.0},
        ]
    )
    sim.get_box_images = AsyncMock(return_value=[])
    sim.remove_box = AsyncMock(return_value={"ok": True})
    sim.pick_and_place = AsyncMock(return_value={"ok": True})

    inference = MagicMock()
    inference.get_action = AsyncMock(return_value="")

    broadcast = AsyncMock()

    return sim, inference, broadcast


class TestControlLoopLifecycle:
    def test_initial_state_is_idle(self) -> None:
        sim, inference, broadcast = _make_deps()
        loop = ControlLoop(sim, inference, broadcast)
        assert loop.state == "idle"

    @pytest.mark.anyio
    async def test_start_transitions_to_initializing(self) -> None:
        sim, inference, broadcast = _make_deps()
        loop = ControlLoop(sim, inference, broadcast)
        task = asyncio.create_task(loop.start())
        await asyncio.sleep(0.1)
        assert loop.state in ("initializing", "running")
        loop.pause()
        await asyncio.sleep(0.1)
        assert loop.state == "paused"
        loop.reset()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.anyio
    async def test_reset_returns_to_idle(self) -> None:
        sim, inference, broadcast = _make_deps()
        loop = ControlLoop(sim, inference, broadcast)
        task = asyncio.create_task(loop.start())
        await asyncio.sleep(0.1)
        loop.reset()
        await asyncio.sleep(0.1)
        assert loop.state == "idle"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestControlLoopProcessing:
    @pytest.mark.anyio
    async def test_processes_box_images(self) -> None:
        sim, inference, broadcast = _make_deps()
        box_image = {
            "box_id": "box_0001",
            "image_b64": "AAAA",
            "size": [0.25, 0.25, 0.25],
            "weight": 10.0,
            "type": "normal",
        }
        sim.get_box_images = AsyncMock(
            side_effect=[[box_image], [], [], [], [], []],
        )
        inference.get_action = AsyncMock(
            return_value=(
                "<think>\nTest reasoning.\n</think>\n\n"
                "<answer>\n"
                '{"action": "WAIT", "reason": "testing"}\n'
                "</answer>"
            )
        )
        loop = ControlLoop(sim, inference, broadcast)
        task = asyncio.create_task(loop.start())
        await asyncio.sleep(0.5)
        loop.reset()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        inference.get_action.assert_called_once()
