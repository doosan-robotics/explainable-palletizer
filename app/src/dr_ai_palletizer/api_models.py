"""Pydantic request/response schemas for the app server API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ServiceHealth(BaseModel):
    name: str
    healthy: bool
    detail: str = ""


class StatusResponse(BaseModel):
    status: str = "ok"
    services: list[ServiceHealth] = Field(default_factory=list)
    state: str = "idle"
    sim_running: bool = False


class PlanRequest(BaseModel):
    scenario_text: str = Field(..., min_length=1)
    system_prompt: str | None = None


class PlanResponse(BaseModel):
    plan: str
    model: str = ""


# Default system prompt for the /plan endpoint.
SYSTEM_PROMPT: str = (
    "You are a palletizing robot controller. At each step you see 1-3 "
    "individual box images along with their measured weight and dimensions, "
    "plus images of 2 pallets showing their current state.\n\n"
    "Box dimensions use a unit 'd' where d = 25 cm. Each box axis "
    "is either 1d or 2d. Pallets are 4d x 4d x 4d grids (4d per axis). "
    "Valid placement positions are pre-computed and listed for each box on "
    "each pallet.\n\n"
    "Your process:\n"
    "1. OBSERVE: Examine each box image and sensor data -- what do you see "
    "about its condition, shape, and surface?\n"
    "2. CHECK FIRST: Are any boxes damaged, contaminated, or unpickable? "
    "If yes, immediately CALL_A_HUMAN for those boxes. This takes priority "
    "over all other actions.\n"
    "3. ASSESS: For remaining boxes, determine handling parameters (speed, "
    "grip strength) based on observations. Check which boxes have valid "
    "positions on which pallets.\n"
    "4. DECIDE: Choose the safest and most efficient action:\n"
    "   - Prioritize the pallet closer to completion (higher fill %).\n"
    "   - Place heavy boxes at low z-positions (bottom).\n"
    "   - Place fragile boxes at high z-positions (top) with gentle handling.\n"
    "   - Consider lateral support and stacking stability.\n\n"
    "Available actions:\n"
    "- PICK_AND_PLACE: Pick one box and place it at a specific grid position "
    "on a pallet. Adjust speed and grip strength based on box condition.\n"
    "- CALL_A_HUMAN: Route damaged, contaminated, or unpickable boxes for "
    "human inspection. Also used when no pallet has capacity. Check this "
    "FIRST before considering placement.\n"
    "- WAIT: Only valid when fewer than 3 boxes are visible AND no box can "
    "be safely placed AND you cannot call a human AND your last action was "
    "not WAIT. Use when waiting for more boxes could enable a better action "
    "(e.g., only a fragile box available and pallets are empty).\n\n"
    "Each step includes what your last action was. You cannot WAIT twice "
    "in a row.\n\n"
    "Answer in the following format:\n"
    "<think>\nyour reasoning\n</think>\n\n"
    "<answer>\n"
    "For PICK_AND_PLACE:\n"
    '{"action": "PICK_AND_PLACE", "box": "<BOX_ID>", "target_pallet": <1 or 2>, '
    '"position": [<x>, <y>, <z>], "speed_pct": <number>, '
    '"grip_strength": "<standard|gentle|firm>", '
    '"reason": "<brief rationale>"}\n'
    "For CALL_A_HUMAN:\n"
    '{"action": "CALL_A_HUMAN", "boxes": ["<BOX_ID>", ...], '
    '"reason": "<brief rationale>"}\n'
    "For WAIT:\n"
    '{"action": "WAIT", "reason": "<brief rationale>"}\n'
    "</answer>"
)


class PalletizeRequest(BaseModel):
    """Placeholder for a full palletize orchestration loop."""

    scenario_text: str = Field(default="")
    auto_execute: bool = False


class PalletizeResponse(BaseModel):
    status: str = "accepted"
    message: str = "Palletize orchestration not yet implemented."
