"""Tests for SimRunner command queue and dispatch logic.

Tests the pure-Python queue mechanics and dispatch without Isaac Sim.
The PalletizerEnv is mocked to avoid GPU/SimulationApp dependencies.
"""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest
from drp_sim.robot import HOME_JOINTS
from drp_sim.sim_runner import SimCommand, SimMessage, SimRunner


@pytest.fixture()
def runner() -> SimRunner:
    return SimRunner(load_robot=True, spawn_boxes=True)


class TestSimCommand:
    def test_all_commands_exist(self) -> None:
        expected = {
            "PLAY",
            "PAUSE",
            "RESET",
            "STEP",
            "GET_STATE",
            "SET_JOINTS",
            "SPAWN_BOX",
            "GET_BOX_IMAGES",
            "GO_HOME",
            "GET_CAMERA",
            "MOVE_PLANNED",
            "MOVE_CARTESIAN",
            "FILL_BUFFER",
            "CLEAR_CONVEYOR",
            "CLEAR_PALLET",
            "PICK_PLACE",
            "AUTO_PICK",
            "HUMAN_CALL",
            "REMOVE_BOX",
            "GET_BUFFER_STATUS",
            "SHUTDOWN",
        }
        actual = {cmd.name for cmd in SimCommand}
        assert actual == expected


class TestSimMessage:
    def test_default_payload_is_empty_dict(self) -> None:
        msg = SimMessage(command=SimCommand.PLAY)
        assert msg.payload == {}
        assert isinstance(msg.future, Future)

    def test_custom_payload(self) -> None:
        msg = SimMessage(command=SimCommand.STEP, payload={"num_steps": 5})
        assert msg.payload["num_steps"] == 5


class TestSendCommand:
    def test_returns_future(self, runner: SimRunner) -> None:
        future = runner.send_command(SimCommand.PLAY)
        assert isinstance(future, Future)

    def test_enqueues_message(self, runner: SimRunner) -> None:
        runner.send_command(SimCommand.PAUSE, {"key": "val"})
        assert not runner._cmd_queue.empty()


class TestDispatch:
    """Test _dispatch directly without running the full main loop."""

    def test_play(self, runner: SimRunner) -> None:
        result = runner._dispatch(SimCommand.PLAY, {})
        assert result == {"status": "playing"}
        assert runner.is_playing is True

    def test_pause(self, runner: SimRunner) -> None:
        runner._playing = True
        result = runner._dispatch(SimCommand.PAUSE, {})
        assert result == {"status": "paused"}
        assert runner.is_playing is False

    def test_reset_without_env(self, runner: SimRunner) -> None:
        runner._playing = True
        result = runner._dispatch(SimCommand.RESET, {})
        assert result == {"status": "reset"}
        assert runner.is_playing is False
        assert runner._step_count == 0

    def test_step_without_env(self, runner: SimRunner) -> None:
        result = runner._dispatch(SimCommand.STEP, {"num_steps": 1})
        assert "error" in result

    def test_step_with_mock_env(self, runner: SimRunner) -> None:
        mock_env = MagicMock()
        runner._env = mock_env
        result = runner._dispatch(SimCommand.STEP, {"num_steps": 3, "render": False})
        assert result["steps"] == 3
        assert mock_env.step.call_count == 3
        assert runner._step_count == 3

    def test_set_joints_with_mock_env(self, runner: SimRunner) -> None:
        mock_env = MagicMock()
        runner._env = mock_env
        positions = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = runner._dispatch(SimCommand.SET_JOINTS, {"joint_positions": positions})
        assert result == {"status": "ok"}
        mock_env.set_joint_positions.assert_called_once_with(positions)

    def test_spawn_box_with_mock_env(self, runner: SimRunner) -> None:
        mock_env = MagicMock()
        mock_env.spawn_box.return_value = "/World/box_0"
        mock_env.boxes = [MagicMock()]
        runner._env = mock_env
        result = runner._dispatch(SimCommand.SPAWN_BOX, {})
        assert result["prim_path"] == "/World/box_0"
        assert result["box_count"] == 1

    def test_fill_buffer_without_env(self, runner: SimRunner) -> None:
        result = runner._dispatch(SimCommand.FILL_BUFFER, {})
        assert "error" in result

    def test_fill_buffer_with_mock_env(self, runner: SimRunner) -> None:
        mock_env = MagicMock()
        mock_env.fill_buffer.return_value = {
            "status": "filling",
            "occupied": 3,
            "capacity": 3,
        }
        runner._env = mock_env
        result = runner._dispatch(SimCommand.FILL_BUFFER, {})
        assert result["status"] == "filling"
        assert result["occupied"] == 3
        mock_env.fill_buffer.assert_called_once()

    def test_remove_box_without_env(self, runner: SimRunner) -> None:
        result = runner._dispatch(SimCommand.REMOVE_BOX, {"box_id": "box_0001"})
        assert "error" in result

    def test_go_home_with_mock_env(self, runner: SimRunner) -> None:
        mock_env = MagicMock()
        runner._env = mock_env
        result = runner._dispatch(SimCommand.GO_HOME, {})
        assert result == {"status": "ok"}
        mock_env.set_joint_positions.assert_called_once_with(list(HOME_JOINTS))

    def test_get_camera_returns_placeholder(self, runner: SimRunner) -> None:
        result = runner._dispatch(SimCommand.GET_CAMERA, {})
        assert result["width"] == 0
        assert result["encoding"] == "none"

    def test_shutdown(self, runner: SimRunner) -> None:
        result = runner._dispatch(SimCommand.SHUTDOWN, {})
        assert result == {"status": "shutting_down"}
        assert runner._shutdown is True


class TestGetState:
    def test_empty_state_without_env(self, runner: SimRunner) -> None:
        result = runner._get_state()
        assert result["joint_positions"] == []
        assert result["box_positions"] == []
        assert result["sim_time"] == 0.0

    def test_state_with_mock_env(self, runner: SimRunner) -> None:
        import numpy as np

        mock_env = MagicMock()
        mock_env.get_joint_positions.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_box = MagicMock()
        mock_box.get_world_poses.return_value = (np.array([[1.0, 2.0, 3.0]]), None)
        mock_box.prim_path = "/World/box_0"
        mock_env.boxes = [mock_box]
        runner._env = mock_env

        result = runner._get_state()
        assert result["joint_positions"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert len(result["box_positions"]) == 1
        assert result["box_positions"][0]["x"] == 1.0

    def test_state_handles_robot_not_loaded(self, runner: SimRunner) -> None:
        mock_env = MagicMock()
        mock_env.get_joint_positions.side_effect = RuntimeError("no robot")
        mock_env.boxes = []
        runner._env = mock_env

        result = runner._get_state()
        assert result["joint_positions"] == []


class TestProcessOne:
    def test_sets_future_result(self, runner: SimRunner) -> None:
        msg = SimMessage(command=SimCommand.PLAY)
        runner._process_one(msg)
        assert msg.future.result() == {"status": "playing"}

    def test_sets_future_exception_on_error(self, runner: SimRunner) -> None:
        msg = SimMessage(command=SimCommand.SET_JOINTS, payload={})
        runner._process_one(msg)
        with pytest.raises(KeyError):
            msg.future.result()


class TestSimTime:
    def test_initial_zero(self, runner: SimRunner) -> None:
        assert runner.sim_time == 0.0

    def test_increments_with_steps(self, runner: SimRunner) -> None:
        mock_env = MagicMock()
        runner._env = mock_env
        runner._dispatch(SimCommand.STEP, {"num_steps": 60})
        assert runner.sim_time == pytest.approx(1.0)


class TestSnapshot:
    def test_empty_when_no_annotator(self, runner: SimRunner) -> None:
        result = runner._snapshot()
        assert result["encoding"] == "none"
        assert result["width"] == 0

    def test_returns_base64_when_frame_available(self, runner: SimRunner) -> None:
        runner.frame_buffer.put(b"\xff\xd8fake_jpeg", 640, 480)
        result = runner._snapshot()
        assert result["encoding"] == "jpeg/base64"
        assert result["width"] == 640
        assert result["height"] == 480
        assert len(result["data"]) > 0


class TestFrameBufferIntegration:
    def test_runner_has_frame_buffer(self, runner: SimRunner) -> None:
        assert runner.frame_buffer is not None
        assert runner.frame_buffer.active is False

    def test_get_camera_uses_snapshot(self, runner: SimRunner) -> None:
        runner.frame_buffer.put(b"\xff\xd8jpeg", 320, 240)
        result = runner._dispatch(SimCommand.GET_CAMERA, {})
        assert result["encoding"] == "jpeg/base64"
        assert result["width"] == 320
