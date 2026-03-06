"""Pytest configuration and shared fixtures for drp_sim tests.

Adds sim/src to sys.path so drp_sim is importable without an editable install.
"""

from __future__ import annotations

import sys
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drp_sim.api import create_app
from drp_sim.sim_runner import SimCommand, SimRunner
from fastapi.testclient import TestClient

_SIM_SRC = str(Path(__file__).parent.parent / "src")
if _SIM_SRC not in sys.path:
    sys.path.insert(0, _SIM_SRC)


@pytest.fixture()
def runner() -> SimRunner:
    """A SimRunner instance (no Isaac Sim)."""
    return SimRunner(load_robot=True, spawn_boxes=True)


@pytest.fixture()
def test_client() -> TestClient:
    """A FastAPI TestClient backed by a mock SimRunner."""
    mock_runner = MagicMock(spec=SimRunner)
    mock_runner.sim_time = 0.0
    mock_runner.is_playing = False

    # frame_buffer needed by the WebSocket endpoint
    mock_frame_buffer = MagicMock()
    mock_frame_buffer.subscriber_count = 0
    mock_runner.frame_buffer = mock_frame_buffer

    def _send_command(cmd, payload=None):
        future: Future = Future()
        if cmd in (SimCommand.MOVE_PLANNED, SimCommand.MOVE_CARTESIAN):
            future.set_result({"trajectory": [[0.0] * 5], "waypoints": 1})
        else:
            future.set_result({"status": "ok"})
        return future

    mock_runner.send_command.side_effect = _send_command

    app = create_app(mock_runner)
    return TestClient(app)
