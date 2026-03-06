"""FastAPI server: REST API + control loop orchestrator.

The app-server is a pure orchestrator -- it talks to the sim-server over
HTTP (via SimClient) and to the inference-server (via InferenceClient).
Camera streams and sim-level endpoints live on the sim-server; the
app-server only exposes /api/* endpoints and a /api/events WebSocket for
control-loop status broadcasts.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from dr_ai_palletizer.api_models import (
    HealthResponse,
    PalletizeRequest,
    PalletizeResponse,
    PlanRequest,
    PlanResponse,
    ServiceHealth,
    StatusResponse,
)
from dr_ai_palletizer.clients.inference_client import InferenceClient
from dr_ai_palletizer.clients.sim_client import SimClient
from dr_ai_palletizer.config import Settings
from dr_ai_palletizer.control_loop import ControlLoop

# Configure application logging so control-loop INFO messages appear
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("dr_ai_palletizer").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

_control_loop: ControlLoop | None = None
_loop_task: asyncio.Task | None = None

# WebSocket event subscribers (control-loop broadcasts)
_event_subs: set[WebSocket] = set()


async def _broadcast_event(event: dict) -> None:
    """Send a JSON event to all connected /api/events WebSocket clients."""
    import json

    data = json.dumps(event)
    closed: list[WebSocket] = []
    for ws in _event_subs:
        try:
            await ws.send_text(data)
        except Exception:
            closed.append(ws)
    for ws in closed:
        _event_subs.discard(ws)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _control_loop

    settings = Settings()

    app.state.sim_client = SimClient(
        base_url=settings.sim_server_url,
        timeout=settings.request_timeout,
    )
    app.state.inference_client = InferenceClient(
        base_url=settings.inference_server_url,
        model=settings.active_model,
        timeout=settings.request_timeout,
    )
    app.state.settings = settings

    _control_loop = ControlLoop(
        app.state.sim_client,
        app.state.inference_client,
        _broadcast_event,
        max_completion_tokens=settings.max_completion_tokens,
        use_few_shot=not settings.lora_adapter_path,
    )

    prompt_mode = "fine-tuned (LoRA)" if settings.lora_adapter_path else "few-shot"
    logger.info(
        "Server started (sim=%s, inference=%s, model=%s, prompt=%s)",
        settings.sim_server_url,
        settings.inference_server_url,
        settings.active_model,
        prompt_mode,
    )
    yield

    await _stop_loop()
    await app.state.sim_client.close()
    await app.state.inference_client.close()
    _control_loop = None


app = FastAPI(title="DR AI Palletizer", lifespan=_lifespan)


async def _stop_loop() -> None:
    global _loop_task
    if _control_loop is not None:
        _control_loop.reset()
    if _loop_task is not None and not _loop_task.done():
        _loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _loop_task
    _loop_task = None


# ---------------------------------------------------------------------------
# REST API -- health and status
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    loop_state = _control_loop.state if _control_loop else "idle"
    services: list[ServiceHealth] = []

    try:
        sim_health = await app.state.sim_client.health()
        services.append(
            ServiceHealth(
                name="sim-server",
                healthy=sim_health.get("status") == "ok",
                detail=f"sim_time={sim_health.get('sim_time', 0):.1f}",
            )
        )
    except Exception as exc:
        services.append(ServiceHealth(name="sim-server", healthy=False, detail=str(exc)))

    try:
        inference_ok = await app.state.inference_client.health()
        services.append(ServiceHealth(name="inference-server", healthy=inference_ok))
    except Exception as exc:
        services.append(ServiceHealth(name="inference-server", healthy=False, detail=str(exc)))

    all_healthy = all(s.healthy for s in services)
    return StatusResponse(
        status="ok" if all_healthy else "degraded",
        services=services,
        state=loop_state,
        sim_running=all_healthy,
    )


# ---------------------------------------------------------------------------
# REST API -- control loop endpoints
# ---------------------------------------------------------------------------


@app.post("/api/control/start")
async def control_start() -> dict:
    global _loop_task
    if _control_loop is None:
        return {"ok": False, "error": "Control loop not initialized"}
    # Prevent duplicate loops: if already running, just return success
    if _loop_task is not None and not _loop_task.done():
        return {"ok": True, "state": "running"}
    sim: SimClient = app.state.sim_client
    # Fill buffer BEFORE play to prevent extra box spawns during the race
    await sim.fill_buffer()
    await sim.play()

    async def _run_loop() -> None:
        try:
            await _control_loop.start()  # type: ignore[union-attr]
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Control loop crashed")

    _loop_task = asyncio.create_task(_run_loop())
    return {"ok": True, "state": "running"}


@app.post("/api/control/pause")
async def control_pause() -> dict:
    if _control_loop is None:
        return {"ok": False, "error": "Control loop not initialized"}
    _control_loop.pause()
    return {"ok": True, "state": "paused"}


@app.post("/api/control/resume")
async def control_resume() -> dict:
    if _control_loop is None:
        return {"ok": False, "error": "Control loop not initialized"}
    _control_loop.resume()
    return {"ok": True, "state": "running"}


@app.post("/api/control/reset")
async def control_reset() -> dict:
    await _stop_loop()
    sim: SimClient = app.state.sim_client
    await sim.reset()
    return {"ok": True, "state": "idle"}


# ---------------------------------------------------------------------------
# REST API -- planning and palletize
# ---------------------------------------------------------------------------


@app.post("/api/plan", response_model=PlanResponse)
async def plan(request: PlanRequest) -> PlanResponse:
    from dr_ai_palletizer.domain.models import COSMOS2_TASK_PROMPT

    system_prompt = request.system_prompt or COSMOS2_TASK_PROMPT
    try:
        result = await app.state.inference_client.get_plan(
            system_prompt=system_prompt,
            scenario_text=request.scenario_text,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Inference server error: {exc}",
        ) from exc
    return PlanResponse(plan=result, model=app.state.settings.active_model)


@app.post("/api/palletize", response_model=PalletizeResponse)
async def palletize(request: PalletizeRequest) -> PalletizeResponse:
    return PalletizeResponse(
        status="accepted",
        message="Palletize orchestration not yet implemented.",
    )


# ---------------------------------------------------------------------------
# WebSocket: control-loop event stream
# ---------------------------------------------------------------------------


@app.websocket("/api/events")
async def ws_events(websocket: WebSocket) -> None:
    """JSON event stream: reasoning, status, pallet updates from the control loop."""
    await websocket.accept()
    _event_subs.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _event_subs.discard(websocket)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("APP_HOST", "0.0.0.0")
    port = int(os.environ.get("APP_PORT", "8000"))
    uvicorn.run("dr_ai_palletizer.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
