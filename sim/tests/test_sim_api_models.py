"""Tests for sim server Pydantic request/response models."""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest
from drp_sim.api import create_app
from drp_sim.api_models import (
    CameraResponse,
    MoveCartesianRequest,
    MovePlannedRequest,
    MoveRequest,
    SimHealthResponse,
    SimStateResponse,
    SpawnBoxResponse,
    StepRequest,
)
from drp_sim.sim_runner import SimCommand, SimRunner
from fastapi.testclient import TestClient
from pydantic import ValidationError


@pytest.fixture()
def test_client() -> TestClient:
    """A FastAPI TestClient backed by a mock SimRunner."""
    mock_runner = MagicMock(spec=SimRunner)
    mock_runner.sim_time = 0.0
    mock_runner.is_playing = False
    mock_frame_buffer = MagicMock()
    mock_frame_buffer.subscriber_count = 0
    mock_runner.frame_buffer = mock_frame_buffer

    def _send_command(cmd, payload=None):
        future: Future = Future()
        if cmd in (SimCommand.MOVE_PLANNED, SimCommand.MOVE_CARTESIAN):
            future.set_result({"trajectory": [[0.0] * 5]})
        else:
            future.set_result({"status": "ok"})
        return future

    mock_runner.send_command.side_effect = _send_command
    app = create_app(mock_runner)
    return TestClient(app)


class TestSimHealthResponse:
    def test_defaults(self) -> None:
        resp = SimHealthResponse()
        assert resp.status == "ok"
        assert resp.sim_time == 0.0
        assert resp.is_playing is False

    def test_custom_values(self) -> None:
        resp = SimHealthResponse(status="degraded", sim_time=12.5, is_playing=True)
        assert resp.status == "degraded"
        assert resp.sim_time == 12.5
        assert resp.is_playing is True


class TestSimStateResponse:
    def test_defaults(self) -> None:
        resp = SimStateResponse()
        assert resp.joint_positions == []
        assert resp.box_positions == []
        assert resp.sim_time == 0.0

    def test_with_data(self) -> None:
        resp = SimStateResponse(
            joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5],
            box_positions=[{"x": 1.0, "y": 2.0, "z": 3.0, "prim_path": "/World/box_0"}],
            sim_time=5.0,
        )
        assert len(resp.joint_positions) == 5
        assert resp.box_positions[0].x == 1.0


class TestMoveRequest:
    def test_valid_5_joints(self) -> None:
        req = MoveRequest(joint_positions=[0.0, 0.1, 0.2, 0.3, 0.4])
        assert len(req.joint_positions) == 5

    def test_rejects_fewer_than_5(self) -> None:
        with pytest.raises(ValidationError):
            MoveRequest(joint_positions=[0.0, 0.1])

    def test_rejects_more_than_5(self) -> None:
        with pytest.raises(ValidationError):
            MoveRequest(joint_positions=[0.0] * 6)

    def test_requires_joint_positions(self) -> None:
        with pytest.raises(ValidationError):
            MoveRequest()


class TestStepRequest:
    def test_defaults(self) -> None:
        req = StepRequest()
        assert req.num_steps == 1
        assert req.render is True

    def test_custom(self) -> None:
        req = StepRequest(num_steps=10, render=False)
        assert req.num_steps == 10
        assert req.render is False

    def test_rejects_zero_steps(self) -> None:
        with pytest.raises(ValidationError):
            StepRequest(num_steps=0)

    def test_rejects_negative_steps(self) -> None:
        with pytest.raises(ValidationError):
            StepRequest(num_steps=-1)

    def test_rejects_too_many_steps(self) -> None:
        with pytest.raises(ValidationError):
            StepRequest(num_steps=1001)


class TestSpawnBoxResponse:
    def test_creation(self) -> None:
        resp = SpawnBoxResponse(prim_path="/World/box_0", box_count=1)
        assert resp.prim_path == "/World/box_0"
        assert resp.box_count == 1


class TestCameraResponse:
    def test_defaults(self) -> None:
        resp = CameraResponse()
        assert resp.width == 0
        assert resp.height == 0
        assert resp.data == ""
        assert resp.encoding == "none"


def test_move_planned_request_validates_length() -> None:
    with pytest.raises(ValidationError):
        MovePlannedRequest(target=[0.1, 0.2, 0.3])  # only 3 elements — must fail

    req = MovePlannedRequest(target=[0.0] * 5)
    assert req.execute is True  # default


def test_move_cartesian_request_validates_lengths() -> None:
    with pytest.raises(ValidationError):
        MoveCartesianRequest(position=[0.0, 0.0], quaternion=[1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValidationError):
        MoveCartesianRequest(position=[0.0, 0.0, 0.5], quaternion=[1.0, 0.0])

    req = MoveCartesianRequest(position=[0.5, 0.0, 0.3], quaternion=[1.0, 0.0, 0.0, 0.0])
    assert req.execute is True


def test_motion_endpoints_registered(test_client) -> None:
    routes = {r.path for r in test_client.app.routes}
    assert "/sim/robot/move_planned" in routes
    assert "/sim/robot/move_cartesian" in routes


def test_move_motion_response_waypoints_computed() -> None:
    from drp_sim.api_models import MoveMotionResponse

    resp = MoveMotionResponse(trajectory=[[0.0] * 5, [0.1] * 5])
    assert resp.waypoints == 2
    assert len(resp.trajectory) == 2

    assert MoveMotionResponse(trajectory=[]).waypoints == 0


def test_move_planned_endpoint_returns_motion_response(test_client) -> None:
    resp = test_client.post("/sim/robot/move_planned", json={"target": [0.0] * 5})
    assert resp.status_code == 200
    data = resp.json()
    assert "trajectory" in data
    assert "waypoints" in data


def test_move_cartesian_endpoint_returns_motion_response(test_client) -> None:
    resp = test_client.post(
        "/sim/robot/move_cartesian",
        json={"position": [0.5, 0.0, 0.3], "quaternion": [1.0, 0.0, 0.0, 0.0]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "trajectory" in data
    assert "waypoints" in data
