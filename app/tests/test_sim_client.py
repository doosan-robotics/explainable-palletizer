"""Unit tests for SimClient.move_planned and SimClient.move_cartesian."""

from __future__ import annotations

import json

import httpx
import pytest
from dr_ai_palletizer.clients.sim_client import SimClient

pytestmark = pytest.mark.anyio

TRAJECTORY_RESPONSE = {"trajectory": [[0.1, 0.2, 0.3, 0.4, 0.5]], "waypoints": 1}


def _make_client(handler) -> SimClient:
    """Build a SimClient with a mock transport using the given request handler."""
    client = SimClient.__new__(SimClient)
    client._client = httpx.AsyncClient(
        base_url="http://sim-server:8100",
        transport=httpx.MockTransport(handler),
    )
    return client


def _ok_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path in ("/sim/robot/move_planned", "/sim/robot/move_cartesian"):
        return httpx.Response(200, json=TRAJECTORY_RESPONSE)
    if request.url.path == "/sim/robot/pick_and_place":
        return httpx.Response(200, json={"status": "ok", "steps": []})
    return httpx.Response(404)


class TestMovePlanned:
    async def test_sim_client_has_move_planned(self) -> None:
        """move_planned exists and calls POST /sim/robot/move_planned."""
        client = _make_client(_ok_handler)
        result = await client.move_planned(target=[0.0, 0.1, 0.2, 0.3, 0.4])
        assert result == TRAJECTORY_RESPONSE

    async def test_move_planned_sends_correct_payload(self) -> None:
        """move_planned serialises target and execute into the request body."""
        captured: list[httpx.Request] = []

        def capturing_handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=TRAJECTORY_RESPONSE)

        client = _make_client(capturing_handler)
        target = [1.0, 2.0, 3.0, 4.0, 5.0]
        await client.move_planned(target=target, execute=False)

        assert len(captured) == 1
        body = json.loads(captured[0].content)
        assert body["target"] == target
        assert body["execute"] is False

    async def test_move_planned_uses_120s_timeout(self) -> None:
        """move_planned passes timeout=120.0 to the underlying post call."""
        recorded_kwargs: list[dict] = []

        class _FakeAsyncClient:
            async def post(self, url: str, **kwargs) -> httpx.Response:
                recorded_kwargs.append(kwargs)
                request = httpx.Request("POST", f"http://sim-server:8100{url}")
                return httpx.Response(200, json=TRAJECTORY_RESPONSE, request=request)

        client = SimClient.__new__(SimClient)
        client._client = _FakeAsyncClient()

        await client.move_planned(target=[0.0, 0.0, 0.0, 0.0, 0.0])

        assert len(recorded_kwargs) == 1
        assert recorded_kwargs[0]["timeout"] == pytest.approx(120.0)

    async def test_move_planned_raises_on_http_error(self) -> None:
        """move_planned propagates HTTP errors via raise_for_status."""

        def error_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "server error"})

        client = _make_client(error_handler)

        with pytest.raises(httpx.HTTPStatusError):
            await client.move_planned(target=[0.0, 0.0, 0.0, 0.0, 0.0])


class TestMoveCartesian:
    async def test_sim_client_has_move_cartesian(self) -> None:
        """move_cartesian exists and calls POST /sim/robot/move_cartesian."""
        client = _make_client(_ok_handler)
        result = await client.move_cartesian(
            position=[0.5, 0.0, 0.8], quaternion=[0.0, 0.0, 0.0, 1.0]
        )
        assert result == TRAJECTORY_RESPONSE

    async def test_move_cartesian_sends_correct_payload(self) -> None:
        """move_cartesian serialises position, quaternion and execute into the body."""
        captured: list[httpx.Request] = []

        def capturing_handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=TRAJECTORY_RESPONSE)

        client = _make_client(capturing_handler)
        position = [0.1, 0.2, 0.3]
        quaternion = [0.0, 0.0, 0.707, 0.707]
        await client.move_cartesian(position=position, quaternion=quaternion, execute=False)

        assert len(captured) == 1
        body = json.loads(captured[0].content)
        assert body["position"] == position
        assert body["quaternion"] == quaternion
        assert body["execute"] is False

    async def test_move_cartesian_uses_120s_timeout(self) -> None:
        """move_cartesian passes timeout=120.0 to the underlying post call."""
        recorded_kwargs: list[dict] = []

        class _FakeAsyncClient:
            async def post(self, url: str, **kwargs) -> httpx.Response:
                recorded_kwargs.append(kwargs)
                request = httpx.Request("POST", f"http://sim-server:8100{url}")
                return httpx.Response(200, json=TRAJECTORY_RESPONSE, request=request)

        client = SimClient.__new__(SimClient)
        client._client = _FakeAsyncClient()

        await client.move_cartesian(position=[0.0, 0.0, 0.0], quaternion=[0.0, 0.0, 0.0, 1.0])

        assert len(recorded_kwargs) == 1
        assert recorded_kwargs[0]["timeout"] == pytest.approx(120.0)

    async def test_move_cartesian_raises_on_http_error(self) -> None:
        """move_cartesian propagates HTTP errors via raise_for_status."""

        def error_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"detail": "unavailable"})

        client = _make_client(error_handler)

        with pytest.raises(httpx.HTTPStatusError):
            await client.move_cartesian(position=[0.0, 0.0, 0.0], quaternion=[0.0, 0.0, 0.0, 1.0])


class TestSimClientHTTPMethods:
    """Tests for SimClient methods that call real HTTP endpoints."""

    async def test_get_pick_positions_returns_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/sim/geometry/pick_positions"
            return httpx.Response(
                200,
                json={
                    "positions": [
                        {"x": 0.59, "y": 0.10, "z": -0.206},
                        {"x": 0.59, "y": 0.70, "z": -0.206},
                        {"x": 0.59, "y": 1.30, "z": -0.206},
                    ]
                },
            )

        client = _make_client(handler)
        positions = await client.get_pick_positions()
        assert len(positions) == 3
        assert all("x" in p and "y" in p and "z" in p for p in positions)

    async def test_get_pallet_centers_returns_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/sim/geometry/pallet_centers"
            return httpx.Response(
                200,
                json={
                    "centers": [
                        {"x": -0.657, "y": -0.471, "z": -0.251},
                        {"x": -0.397, "y": -0.471, "z": -0.251},
                    ]
                },
            )

        client = _make_client(handler)
        centers = await client.get_pallet_centers()
        assert len(centers) == 2

    async def test_remove_box_calls_endpoint(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/sim/boxes/remove"
            body = json.loads(request.content)
            assert body["box_id"] == "box_0001"
            return httpx.Response(200, json={"status": "ok", "box_id": "box_0001"})

        client = _make_client(handler)
        result = await client.remove_box("box_0001")
        assert result["status"] == "ok"

    async def test_pick_and_place_calls_endpoint(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/sim/robot/pick_place"
            body = json.loads(request.content)
            assert body["box_prim"] == "/World/box_0001"
            assert body["pick_position"] == [0.5, 3.0, -0.2]
            assert body["drop_position"] == [0.8, -0.1, 0.125]
            assert body["drop_quaternion"] == [0.7071, 0.0, 0.0, 0.7071]
            return httpx.Response(200, json={"status": "ok", "steps": ["above_pick"]})

        client = _make_client(handler)
        result = await client.pick_and_place(
            box_id="box_0001",
            speed_pct=100,
            pick_pose=[0.5, 3.0, -0.2],
            drop_pose=[0.8, -0.1, 0.125],
            drop_quaternion=[0.7071, 0.0, 0.0, 0.7071],
        )
        assert result["status"] == "ok"

    async def test_get_box_images_calls_endpoint(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/sim/boxes/images":
                return httpx.Response(
                    200, json={"images": [{"box_id": "box_0001", "image_b64": "abc"}]}
                )
            return httpx.Response(404)

        client = _make_client(handler)
        result = await client.get_box_images()
        assert len(result) == 1
        assert result[0]["box_id"] == "box_0001"
