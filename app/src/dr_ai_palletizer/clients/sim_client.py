"""Async HTTP client for the sim-server API."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class SimClient:
    """Thin wrapper around httpx.AsyncClient for sim-server communication.

    Parameters
    ----------
    base_url:
        Root URL of the sim server (e.g. ``http://sim-server:8100``).
    timeout:
        Request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        resp = await self._client.get("/sim/health")
        resp.raise_for_status()
        return resp.json()

    async def play(self) -> dict:
        resp = await self._client.post("/sim/play")
        resp.raise_for_status()
        return resp.json()

    async def pause(self) -> dict:
        resp = await self._client.post("/sim/pause")
        resp.raise_for_status()
        return resp.json()

    async def reset(self) -> dict:
        resp = await self._client.post("/sim/reset")
        resp.raise_for_status()
        return resp.json()

    async def step(self, num_steps: int = 1, render: bool = True) -> dict:
        resp = await self._client.post(
            "/sim/step",
            json={"num_steps": num_steps, "render": render},
        )
        resp.raise_for_status()
        return resp.json()

    async def get_state(self) -> dict:
        resp = await self._client.get("/sim/state")
        resp.raise_for_status()
        return resp.json()

    async def move(self, joint_positions: list[float]) -> dict:
        resp = await self._client.post(
            "/sim/robot/move",
            json={"joint_positions": joint_positions},
        )
        resp.raise_for_status()
        return resp.json()

    async def go_home(self) -> dict:
        resp = await self._client.post("/sim/robot/home")
        resp.raise_for_status()
        return resp.json()

    async def move_planned(
        self,
        target: list[float],
        execute: bool = True,
        timeout: float = 120.0,
    ) -> dict:
        """Plan (and optionally execute) joint-space motion via cuRobo on the sim-server."""
        resp = await self._client.post(
            "/sim/robot/move_planned",
            json={"target": target, "execute": execute},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    async def move_cartesian(
        self,
        position: list[float],
        quaternion: list[float],
        execute: bool = True,
        timeout: float = 120.0,
    ) -> dict:
        """Plan (and optionally execute) Cartesian motion via cuRobo on the sim-server."""
        resp = await self._client.post(
            "/sim/robot/move_cartesian",
            json={"position": position, "quaternion": quaternion, "execute": execute},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    async def spawn_box(self) -> dict:
        resp = await self._client.post("/sim/boxes/spawn")
        resp.raise_for_status()
        return resp.json()

    async def fill_buffer(self) -> dict:
        """Fill the conveyor buffer and activate auto-maintain mode."""
        resp = await self._client.post("/sim/boxes/fill_buffer")
        resp.raise_for_status()
        return resp.json()

    async def get_camera(self) -> dict:
        resp = await self._client.get("/sim/camera")
        resp.raise_for_status()
        return resp.json()

    async def auto_pick(self, slot: int, timeout: float = 300.0) -> dict:
        resp = await self._client.post(
            "/sim/robot/auto_pick",
            json={"slot": slot},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    async def get_box_images(self) -> list[dict]:
        """Fetch new box images from the sim. Sim clears its buffer after response."""
        resp = await self._client.get("/sim/boxes/images")
        resp.raise_for_status()
        data = resp.json()
        return data.get("images", [])

    async def get_pick_positions(self) -> list[dict]:
        """Fetch conveyor pickup positions from the sim-server geometry."""
        resp = await self._client.get("/sim/geometry/pick_positions")
        resp.raise_for_status()
        return resp.json().get("positions", [])

    async def get_pallet_centers(self) -> list[dict]:
        """Fetch pallet center positions from the sim-server geometry."""
        resp = await self._client.get("/sim/geometry/pallet_centers")
        resp.raise_for_status()
        return resp.json().get("centers", [])

    async def get_buffer_status(self) -> dict:
        """Fetch conveyor buffer occupancy from the sim-server."""
        resp = await self._client.get("/sim/buffer/status")
        resp.raise_for_status()
        return resp.json()

    async def remove_box(self, box_id: str) -> dict:
        """Remove a single box from the sim by its logical ID."""
        resp = await self._client.post("/sim/boxes/remove", json={"box_id": box_id})
        resp.raise_for_status()
        return resp.json()

    async def pick_and_place(
        self,
        box_id: str,
        speed_pct: int,
        pick_pose: list[float],
        drop_pose: list[float],
        drop_quaternion: list[float] | None = None,
        box_prim: str | None = None,
    ) -> dict:
        """Execute a full pick-and-place sequence via the sim server.

        Parameters
        ----------
        box_id:
            Logical box identifier (e.g. ``box_0001``).
        speed_pct:
            Requested speed percentage (informational; not yet used by sim).
        pick_pose:
            [x, y, z] world-coordinate pick position.
        drop_pose:
            [x, y, z] world-coordinate drop position.
        drop_quaternion:
            Logical [w, x, y, z] rotation for the box at the drop site.
            Identity ``[1,0,0,0]`` means no extra rotation; ``[0.707,0,0,0.707]``
            means 90-degree Z rotation (even-layer interlocking).
            The sim server composes this with the gripper-down base orientation.
        box_prim:
            USD prim path of the box in the sim stage. Required for the sim
            server to snap the box to the end-effector during the carry phase.
            When ``None``, falls back to convention ``/World/{box_id}``.
        """
        prim = box_prim or f"/World/{box_id}"
        quat = drop_quaternion or [1.0, 0.0, 0.0, 0.0]
        logger.info(
            "pick_and_place(%s, speed=%d, prim=%s, quat=%s)",
            box_id,
            speed_pct,
            prim,
            quat,
        )
        body = {
            "box_prim": prim,
            "pick_position": pick_pose,
            "drop_position": drop_pose,
            "drop_quaternion": quat,
            "speed": max(0.0, min(1.0, speed_pct / 100.0)),
        }
        logger.info("pick_and_place REQUEST: %s", body)
        resp = await self._client.post(
            "/sim/robot/pick_place",
            json=body,
            timeout=120.0,
        )
        if resp.status_code != 200:
            logger.error(
                "pick_and_place RESPONSE %d: %s",
                resp.status_code,
                resp.text[:500],
            )
        resp.raise_for_status()
        result = resp.json()
        logger.info("pick_and_place RESPONSE: %s", result)
        return result
