"""FastAPI application for the sim server.

All REST endpoints delegate to ``SimRunner.send_command()`` which enqueues
work for the main-thread sim loop. ``asyncio.wrap_future()`` bridges the
``concurrent.futures.Future`` to an asyncio-awaitable.

The WebSocket ``/sim/camera/stream`` reads JPEG frames directly from the
thread-safe ``FrameBuffer`` without going through the command queue, since
streaming needs to be low-latency and doesn't mutate sim state.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from drp_sim.api_models import (
    AutoPickRequest,
    AutoPickResponse,
    BoxImagesResponse,
    BufferStatusResponse,
    CameraResponse,
    ClearZoneResponse,
    FillBufferResponse,
    HumanCallRequest,
    HumanCallResponse,
    MoveCartesianRequest,
    MoveMotionResponse,
    MovePlannedRequest,
    MoveRequest,
    PalletCentersResponse,
    PickPlaceRequest,
    PickPlaceResponse,
    PickPositionsResponse,
    RemoveBoxRequest,
    RemoveBoxResponse,
    SimHealthResponse,
    SimStateResponse,
    SpawnBoxResponse,
    StepRequest,
)
from drp_sim.sim_runner import SimCommand

if TYPE_CHECKING:
    from drp_sim.sim_runner import SimRunner

logger = logging.getLogger(__name__)

_DEFAULT_STREAM_FPS = 30
_MAX_STREAM_FPS = 60
_MAX_STREAM_CLIENTS = 4


async def _send(runner: SimRunner, cmd: SimCommand, payload: dict | None = None) -> dict:
    """Send a command to the sim runner and await the result."""
    future = runner.send_command(cmd, payload)
    try:
        result = await asyncio.wrap_future(future)
    except Exception as exc:
        logger.exception("_send(%s) raised: %s", cmd.value, exc)
        raise
    if isinstance(result, dict) and "error" in result:
        logger.error("_send(%s) returned error: %s", cmd.value, result["error"])
        raise HTTPException(status_code=500, detail=result["error"])
    return result


def create_app(runner: SimRunner) -> FastAPI:
    """Build a FastAPI app wired to the given SimRunner."""
    app = FastAPI(title="DR Palletizer Sim Server", version="0.1.0")
    app.state.runner = runner

    @app.get("/sim/health", response_model=SimHealthResponse)
    async def health() -> SimHealthResponse:
        return SimHealthResponse(
            status="ok",
            sim_time=runner.sim_time,
            is_playing=runner.is_playing,
        )

    @app.post("/sim/play")
    async def play() -> dict:
        return await _send(runner, SimCommand.PLAY)

    @app.post("/sim/pause")
    async def pause() -> dict:
        return await _send(runner, SimCommand.PAUSE)

    @app.post("/sim/reset")
    async def reset() -> dict:
        return await _send(runner, SimCommand.RESET)

    @app.post("/sim/step")
    async def step(request: StepRequest | None = None) -> dict:
        req = request or StepRequest()
        return await _send(
            runner,
            SimCommand.STEP,
            {"num_steps": req.num_steps, "render": req.render},
        )

    @app.get("/sim/state", response_model=SimStateResponse)
    async def state() -> SimStateResponse:
        result = await _send(runner, SimCommand.GET_STATE)
        return SimStateResponse(**result)

    @app.post("/sim/robot/move")
    async def move(request: MoveRequest) -> dict:
        return await _send(
            runner,
            SimCommand.SET_JOINTS,
            {"joint_positions": request.joint_positions},
        )

    @app.post("/sim/robot/home")
    async def go_home() -> dict:
        return await _send(runner, SimCommand.GO_HOME)

    @app.post("/sim/robot/move_planned", response_model=MoveMotionResponse)
    async def move_planned(request: MovePlannedRequest) -> MoveMotionResponse:
        result = await _send(
            runner,
            SimCommand.MOVE_PLANNED,
            {"target": request.target, "execute": request.execute},
        )
        return MoveMotionResponse(**result)

    @app.post("/sim/robot/move_cartesian", response_model=MoveMotionResponse)
    async def move_cartesian(request: MoveCartesianRequest) -> MoveMotionResponse:
        result = await _send(
            runner,
            SimCommand.MOVE_CARTESIAN,
            {
                "position": request.position,
                "quaternion": request.quaternion,
                "execute": request.execute,
            },
        )
        return MoveMotionResponse(**result)

    @app.post("/sim/boxes/spawn", response_model=SpawnBoxResponse)
    async def spawn_box() -> SpawnBoxResponse:
        result = await _send(runner, SimCommand.SPAWN_BOX)
        return SpawnBoxResponse(**result)

    @app.post("/sim/boxes/fill_buffer", response_model=FillBufferResponse)
    async def fill_buffer() -> FillBufferResponse:
        result = await _send(runner, SimCommand.FILL_BUFFER)
        return FillBufferResponse(**result)

    @app.post("/sim/boxes/clear_conveyor", response_model=ClearZoneResponse)
    async def clear_conveyor() -> ClearZoneResponse:
        """Remove all boxes currently on the conveyor belt (Y in [0, 3.5])."""
        result = await _send(runner, SimCommand.CLEAR_CONVEYOR)
        return ClearZoneResponse(**result)

    @app.post("/sim/boxes/clear_pallet", response_model=ClearZoneResponse)
    async def clear_pallet() -> ClearZoneResponse:
        """Remove all boxes placed on the pallet (Y in [-2, 0])."""
        result = await _send(runner, SimCommand.CLEAR_PALLET)
        return ClearZoneResponse(**result)

    @app.post("/sim/human_call", response_model=HumanCallResponse)
    async def human_call(request: HumanCallRequest) -> HumanCallResponse:
        result = await _send(runner, SimCommand.HUMAN_CALL, {"index": request.index})
        return HumanCallResponse(**result)

    @app.post("/sim/robot/pick_place", response_model=PickPlaceResponse)
    async def pick_place(request: PickPlaceRequest) -> PickPlaceResponse:
        """Execute a pick-and-place sequence and return to home. Blocks until complete."""
        logger.info(
            "pick_place API: box=%s pick=%s drop=%s quat=%s speed=%s",
            request.box_prim,
            request.pick_position,
            request.drop_position,
            request.drop_quaternion,
            request.speed,
        )
        future = runner.send_command(
            SimCommand.PICK_PLACE,
            {
                "box_prim": request.box_prim,
                "pick_position": request.pick_position,
                "drop_position": request.drop_position,
                "drop_quaternion": request.drop_quaternion,
                "speed": request.speed,
            },
        )
        try:
            result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=300.0)
        except TimeoutError as exc:
            logger.error("pick_place TIMEOUT (300s)")
            raise HTTPException(status_code=504, detail="pick_place timed out") from exc
        except Exception as exc:
            logger.exception("pick_place future raised: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if result.get("status") == "error":
            logger.error("pick_place returned error: %s", result)
            raise HTTPException(status_code=500, detail=result.get("message", "motion failed"))
        logger.info("pick_place API SUCCESS: steps=%s", result.get("steps"))
        return PickPlaceResponse(**result)

    @app.post("/sim/robot/auto_pick", response_model=AutoPickResponse)
    async def auto_pick(request: AutoPickRequest) -> AutoPickResponse:
        """Detect the nearest box in the pickup zone and place it on pallet slot 1, 2, or 3.

        Returns 404 if no box is currently in the pickup zone.
        Returns 422 if slot is not 1, 2, or 3.
        """
        future = runner.send_command(SimCommand.AUTO_PICK, {"slot": request.slot.value})
        try:
            result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=300.0)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="auto_pick timed out") from exc
        if result.get("status") == "error":
            msg = result.get("message", "auto-pick failed")
            code = 404 if "no box" in msg else 500
            raise HTTPException(status_code=code, detail=msg)
        return AutoPickResponse(**result)

    @app.get("/sim/boxes/images", response_model=BoxImagesResponse)
    async def box_images() -> BoxImagesResponse:
        result = await _send(runner, SimCommand.GET_BOX_IMAGES)
        return BoxImagesResponse(**result)

    @app.post("/sim/boxes/remove", response_model=RemoveBoxResponse)
    async def remove_box(request: RemoveBoxRequest) -> RemoveBoxResponse:
        """Remove a single box by its logical ID."""
        result = await _send(runner, SimCommand.REMOVE_BOX, {"box_id": request.box_id})
        return RemoveBoxResponse(**result)

    @app.get("/sim/geometry/pick_positions", response_model=PickPositionsResponse)
    async def pick_positions() -> PickPositionsResponse:
        """Return the conveyor pickup positions (derived from buffer geometry)."""
        from drp_sim.box_spawn_config import _DEFAULT_BUFFER_ENDPOINT
        from drp_sim.conveyor_buffer import ConveyorBuffer

        # Pickup positions: one per buffer slot, computed from buffer endpoint
        # and slot stride, matching how ConveyorBuffer._compute_slot_position works.
        ex, ey, ez = _DEFAULT_BUFFER_ENDPOINT
        stride = ConveyorBuffer._SLOT_STRIDE_Y
        positions = [{"x": ex, "y": ey + i * stride, "z": ez} for i in range(3)]
        return PickPositionsResponse(positions=positions)

    @app.get("/sim/geometry/pallet_centers", response_model=PalletCentersResponse)
    async def pallet_centers() -> PalletCentersResponse:
        """Return the pallet center positions (derived from sim geometry)."""
        from drp_sim.sim_runner import _PALLET_SLOTS, _SLOT_Z_C

        centers = [{"x": x, "y": y, "z": _SLOT_Z_C} for x, y in _PALLET_SLOTS]
        return PalletCentersResponse(centers=centers)

    @app.get("/sim/buffer/status", response_model=BufferStatusResponse)
    async def buffer_status() -> BufferStatusResponse:
        """Return the conveyor buffer occupancy (how many boxes are ready)."""
        result = await _send(runner, SimCommand.GET_BUFFER_STATUS)
        return BufferStatusResponse(**result)

    @app.get("/sim/camera", response_model=CameraResponse)
    async def camera() -> CameraResponse:
        result = await _send(runner, SimCommand.GET_CAMERA)
        return CameraResponse(**result)

    async def _stream_camera(websocket: WebSocket, label: str) -> None:
        buf = runner.get_frame_buffer(label)
        if buf.subscriber_count >= _MAX_STREAM_CLIENTS:
            await websocket.close(code=1013, reason="Too many stream clients")
            return
        try:
            fps = min(
                int(websocket.query_params.get("fps", str(_DEFAULT_STREAM_FPS))),
                _MAX_STREAM_FPS,
            )
        except ValueError:
            await websocket.close(code=1008, reason="Invalid fps parameter")
            return
        await websocket.accept()
        buf.subscribe()
        logger.info("Camera stream connected label=%s fps=%d", label, fps)
        last_frame_id = -1
        try:
            while True:
                frame, _w, _h, frame_id = buf.get()
                if frame and frame_id != last_frame_id:
                    await websocket.send_bytes(frame)
                    last_frame_id = frame_id
                await asyncio.sleep(1.0 / fps)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("Camera stream error", exc_info=True)
        finally:
            buf.unsubscribe()
            logger.info("Camera stream disconnected label=%s", label)

    @app.websocket("/sim/camera/stream")
    async def camera_stream(websocket: WebSocket) -> None:
        """Stream JPEG frames — legacy endpoint kept for direct access."""
        await _stream_camera(websocket, "default")

    @app.websocket("/sim/stream/{view}")
    async def camera_stream_view(websocket: WebSocket, view: str) -> None:
        """Stream JPEG frames for the requested view (front/top/persp)."""
        await _stream_camera(websocket, view)

    return app
