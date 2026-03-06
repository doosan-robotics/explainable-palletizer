"""Async control loop state machine for the palletizer application.

Ties together box image polling, LLM inference, and action execution
in a state machine that transitions between idle, initializing,
running, and paused states.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from dr_ai_palletizer.action_parser import ParsedAction, parse_response
from dr_ai_palletizer.domain.models import (
    COSMOS2_SYSTEM_PROMPT,
    D_UNIT_CM,
    PALLET_GRID_SIZE,
    BoxShape,
    select_task_prompt,
)
from dr_ai_palletizer.domain.pallet import (
    PalletState,
    _layer_is_even,
    effective_shape,
    fill_fraction,
    find_valid_positions,
    place_box,
)
from dr_ai_palletizer.prompt_builder import build_scenario_text

logger = logging.getLogger(__name__)

# Cell size in meters (one d-unit)
CELL_SIZE: float = D_UNIT_CM / 100.0

# Max boxes sent to the model per iteration (front of the conveyor queue)
_MAX_PROMPT_BOXES: int = 3

# 90-degree Z-rotation quaternion [w, x, y, z] for even-layer interlocking
_QUAT_IDENTITY: list[float] = [1.0, 0.0, 0.0, 0.0]
_QUAT_90_Z: list[float] = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoxImage:
    """A box image received from the sim, decoded and ready for inference."""

    box_id: str
    image_bytes: bytes
    weight_kg: float
    shape: BoxShape
    width_cm: float
    length_cm: float
    height_cm: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _size_to_shape(size: list[float]) -> BoxShape:
    """Convert [x_m, y_m, z_m] to BoxShape in d-units (each axis min 1)."""
    return BoxShape(
        w=max(1, round(size[0] / CELL_SIZE)),
        ln=max(1, round(size[1] / CELL_SIZE)),
        h=max(1, round(size[2] / CELL_SIZE)),
    )


def grid_to_world(
    gx: int,
    gy: int,
    gz: int,
    shape: BoxShape,
    pallet_center: dict[str, float],
) -> tuple[float, float, float]:
    """Convert grid position + box shape + pallet center to world coordinates.

    Uses the effective (possibly rotated) shape at gz so that the center
    offset matches the physical footprint on even layers.
    """
    eff = effective_shape(shape, gz)
    w, ln, h = eff
    ox = pallet_center["x"] - (PALLET_GRID_SIZE * CELL_SIZE) / 2
    oy = pallet_center["y"] - (PALLET_GRID_SIZE * CELL_SIZE) / 2
    oz = pallet_center["z"]
    return (
        ox + gx * CELL_SIZE + w * CELL_SIZE / 2,
        oy + gy * CELL_SIZE + ln * CELL_SIZE / 2,
        oz + gz * CELL_SIZE + h * CELL_SIZE / 2,
    )


def _pallet_fill_info(pallet: PalletState) -> dict:
    """Return a summary dict for broadcasting pallet state."""
    total_cells = PALLET_GRID_SIZE**3
    occupied = int(pallet.grid.astype(bool).sum())
    return {
        "id": pallet.id + 1,
        "fill_pct": round(fill_fraction(pallet) * 100.0, 1),
        "weight_kg": round(pallet.total_weight_kg, 1),
        "cells_occupied": occupied,
        "cells_total": total_cells,
    }


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------


class ControlLoop:
    """Async state machine that polls for boxes, runs inference, and executes actions.

    States: idle -> initializing -> running <-> paused -> idle (on reset).
    """

    def __init__(
        self,
        sim_client: object,
        inference_client: object,
        broadcast_event: Callable[[dict], Awaitable[None]],
        *,
        max_completion_tokens: int = 2048,
        use_few_shot: bool = False,
    ) -> None:
        self._sim = sim_client
        self._inference = inference_client
        self._broadcast = broadcast_event
        self._max_completion_tokens = max_completion_tokens
        self._use_few_shot = use_few_shot

        self.state: str = "idle"
        self._stop_flag: bool = False
        self._pause_event: asyncio.Event = asyncio.Event()
        self._pause_event.set()

        # Populated during initialization
        self._pick_positions: list[dict] = []
        self._pallet_centers: list[dict] = []
        self._pallets: list[PalletState] = []
        self._box_stack: list[BoxImage] = []
        self._step_number: int = 0
        self._last_action: str | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run the control loop until reset or cancellation."""
        self._stop_flag = False
        self.state = "initializing"
        await self._broadcast({"type": "status", "state": self.state})

        self._pick_positions = await self._sim.get_pick_positions()
        self._pallet_centers = await self._sim.get_pallet_centers()
        self._pallets = [PalletState.empty(i) for i in range(len(self._pallet_centers))]
        self._box_stack = []
        self._step_number = 0
        self._last_action = None

        self.state = "running"
        await self._broadcast({"type": "status", "state": self.state})

        while not self._stop_flag:
            await self._pause_event.wait()
            if self._stop_flag:
                break
            try:
                await self._iteration()
            except Exception:
                logger.exception("Control loop iteration failed, retrying")
                await asyncio.sleep(1.0)

    def pause(self) -> None:
        """Pause the control loop after the current iteration."""
        self.state = "paused"
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume the control loop from paused state."""
        self.state = "running"
        self._pause_event.set()

    def reset(self) -> None:
        """Stop the control loop and return to idle state."""
        self._stop_flag = True
        self._pause_event.set()
        self._box_stack = []
        self._pallets = []
        self._step_number = 0
        self._last_action = None
        self.state = "idle"

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    async def _iteration(self) -> None:
        """Run a single control loop iteration: poll, infer, execute."""
        # 1. Poll for new box images
        raw_images = await self._sim.get_box_images()
        for raw in raw_images:
            size = raw["size"]
            shape = _size_to_shape(size)
            image_bytes = base64.b64decode(raw["image_b64"])
            box = BoxImage(
                box_id=raw["box_id"],
                image_bytes=image_bytes,
                weight_kg=raw["weight"],
                shape=shape,
                width_cm=shape.w * D_UNIT_CM,
                length_cm=shape.ln * D_UNIT_CM,
                height_cm=shape.h * D_UNIT_CM,
            )
            self._box_stack.append(box)

        # 2. Broadcast box stack if new images arrived
        if raw_images:
            await self._broadcast_box_stack()

        # 3. If stack empty, broadcast waiting and sleep
        if not self._box_stack:
            await self._broadcast({"type": "status", "state": "waiting_for_boxes"})
            await asyncio.sleep(0.5)
            return

        # 3b. Wait until at least 2 buffer slots are occupied (boxes physically
        #     present in the pick area) before calling inference.
        try:
            status = await self._sim.get_buffer_status()
            occupied = status.get("occupied", 0)
            if occupied < 2:
                await self._broadcast({"type": "status", "state": "waiting_for_buffer"})
                await asyncio.sleep(0.5)
                return
        except Exception:
            logger.debug("Buffer status check failed, proceeding anyway")

        # 4. Build prompt (only the front of the queue is physically reachable)
        front = self._box_stack[:_MAX_PROMPT_BOXES]
        box_dicts = [
            {
                "box_id": b.box_id,
                "image_bytes": b.image_bytes,
                "weight_kg": b.weight_kg,
                "size": [
                    b.shape.w * CELL_SIZE,
                    b.shape.ln * CELL_SIZE,
                    b.shape.h * CELL_SIZE,
                ],
            }
            for b in front
        ]
        scenario_text = build_scenario_text(
            box_dicts,
            self._pallets,
            step_number=self._step_number,
            last_action=self._last_action,
        )
        task_prompt = select_task_prompt(use_few_shot=False)
        full_text = f"{task_prompt}\n\n{scenario_text}"

        # Log prompt data for debugging
        box_ids_in_stack = [b.box_id for b in self._box_stack]
        box_ids_in_prompt = [b.box_id for b in front]
        logger.info(
            "Step %d: stack=%s, prompt_boxes=%s",
            self._step_number,
            box_ids_in_stack,
            box_ids_in_prompt,
        )
        logger.info("Scenario:\n%s", scenario_text)

        # 5. Broadcast thinking status
        await self._broadcast({"type": "status", "state": "thinking"})

        # 6. Call inference
        image_bytes_list = [b.image_bytes for b in front]
        response = await self._inference.get_action(
            COSMOS2_SYSTEM_PROMPT,
            image_bytes_list,
            full_text,
            max_tokens=self._max_completion_tokens,
        )

        preview = response[:500] if response else "<empty>"
        logger.info("LLM raw response (first 500 chars): %s", preview)

        # 7. Parse response (with continuation retry for truncated think blocks)
        try:
            parsed = parse_response(response)
        except ValueError:
            # If model produced a <think> block but ran out of tokens before
            # the answer, close the think tag and re-prompt for just the JSON.
            if "<think>" in response and "<answer>" not in response:
                logger.info("Truncated think block, requesting continuation")
                continuation = response.rstrip() + "\n</think>\n"
                try:
                    answer_only = await self._inference.continue_response(
                        COSMOS2_SYSTEM_PROMPT,
                        full_text,
                        continuation,
                        max_tokens=512,
                    )
                    parsed = parse_response(continuation + answer_only)
                except (ValueError, Exception) as exc:
                    logger.warning("Continuation also failed: %s", exc)
                    await asyncio.sleep(0.5)
                    return
            else:
                preview = response[:300] if response else "<empty>"
                logger.warning("Failed to parse model response: %s", preview)
                await asyncio.sleep(0.5)
                return

        # 8. Broadcast reasoning + action (matching UI expected format)
        thinking_text = parsed.thinking or parsed.reason
        if thinking_text:
            await self._broadcast({"type": "reasoning", "content": thinking_text})
        action_summary = f"{parsed.action}"
        if parsed.box_id:
            action_summary += f" {parsed.box_id}"
        if parsed.target_pallet is not None:
            action_summary += f" → pallet {parsed.target_pallet}"
        if parsed.position:
            action_summary += f" @ {list(parsed.position)}"
        if parsed.reason:
            action_summary += f" ({parsed.reason})"
        await self._broadcast({"type": "action", "content": action_summary})

        # 9. Execute action
        logger.info(
            "Executing: action=%s box_id=%s pallet=%s position=%s speed=%s",
            parsed.action,
            parsed.box_id,
            parsed.target_pallet,
            parsed.position,
            getattr(parsed, "speed_pct", None),
        )
        acted = await self._execute_action(parsed)

        # 10. Update step tracking (skip if action was a no-op)
        if acted:
            self._step_number += 1
            self._last_action = parsed.action

        # 11. Broadcast pallet update
        await self._broadcast(
            {
                "type": "pallet_update",
                "pallets": [_pallet_fill_info(p) for p in self._pallets],
            }
        )

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def _execute_action(self, parsed: ParsedAction) -> bool:
        """Execute the parsed action. Returns True if an action was taken."""
        if parsed.action == "PICK_AND_PLACE":
            await self._execute_pick_and_place(parsed)
            return True
        elif parsed.action == "CALL_A_HUMAN":
            if not parsed.box_ids:
                logger.warning("CALL_A_HUMAN with empty box list, ignoring")
                return False
            await self._execute_call_human(parsed)
            return True
        elif parsed.action == "WAIT":
            await asyncio.sleep(0.5)
            return True
        return False

    async def _execute_pick_and_place(self, parsed: ParsedAction) -> None:
        """Execute a PICK_AND_PLACE action."""
        if parsed.box_id is None or parsed.target_pallet is None or parsed.position is None:
            logger.warning(
                "PICK_AND_PLACE missing required fields: box_id=%s pallet=%s pos=%s",
                parsed.box_id,
                parsed.target_pallet,
                parsed.position,
            )
            return

        # Find the box in the stack
        box = next((b for b in self._box_stack if b.box_id == parsed.box_id), None)
        if box is None:
            logger.warning(
                "Box %s not found in stack (available: %s)",
                parsed.box_id,
                [b.box_id for b in self._box_stack],
            )
            return

        # Wait for at least one box to be ready in the conveyor buffer.
        # Images are captured at the conveyor start; the box needs time to
        # travel to the buffer slot before the robot can pick it.
        for _ in range(60):  # up to ~30 seconds
            try:
                status = await self._sim.get_buffer_status()
                if status.get("occupied", 0) > 0:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        else:
            logger.warning("Timed out waiting for buffer box, attempting pick anyway")

        # target_pallet is 1-indexed from the model
        pallet_idx = parsed.target_pallet - 1
        if pallet_idx < 0 or pallet_idx >= len(self._pallets):
            logger.warning("Invalid pallet index %d", parsed.target_pallet)
            return

        # Validate position against valid placements. The LLM often picks
        # occupied or invalid positions; fall back to the first valid one.
        valid = find_valid_positions(
            self._pallets[pallet_idx],
            box.shape,
            box.weight_kg,
        )
        # Coerce to int tuple -- model may return floats or nested lists
        raw_pos = parsed.position
        try:
            coerced = [int(v[0]) if isinstance(v, list | tuple) else int(v) for v in raw_pos]
            pos = (coerced[0], coerced[1], coerced[2])
        except (TypeError, ValueError, IndexError):
            logger.warning("Malformed position %s, using first valid", raw_pos)
            pos = valid[0] if valid else (0, 0, 0)
        logger.info(
            "Validation: pallet=%d pos=%s shape=%s valid[:5]=%s",
            pallet_idx + 1,
            pos,
            box.shape,
            valid[:5],
        )
        if pos not in valid:
            if not valid:
                # Try all pallets for a valid position
                for alt_idx, alt_pallet in enumerate(self._pallets):
                    valid = find_valid_positions(alt_pallet, box.shape, box.weight_kg)
                    if valid:
                        pallet_idx = alt_idx
                        logger.warning(
                            "No valid positions on pallet %d, switching to pallet %d",
                            parsed.target_pallet,
                            pallet_idx + 1,
                        )
                        break
            if not valid:
                logger.error(
                    "No valid position on ANY pallet for box=%s shape=%s",
                    parsed.box_id,
                    box.shape,
                )
                return
            old_pos = pos
            pos = valid[0]
            logger.warning(
                "LLM position %s invalid on pallet %d, corrected to %s (valid: %s)",
                old_pos,
                pallet_idx + 1,
                pos,
                valid[:5],
            )
        gx, gy, gz = pos

        # Compute drop pose in world coordinates
        drop_world = grid_to_world(
            gx,
            gy,
            gz,
            box.shape,
            self._pallet_centers[pallet_idx],
        )

        # Pick pose: first available pick position
        pick_pose = [
            self._pick_positions[0]["x"],
            self._pick_positions[0]["y"],
            self._pick_positions[0]["z"],
        ]

        # Even layers rotate 90 degrees for interlocking stability
        drop_quat = _QUAT_90_Z if _layer_is_even(gz) else _QUAT_IDENTITY

        logger.info(
            "PICK_AND_PLACE: box=%s shape=%s grid=(%d,%d,%d) pallet=%d "
            "pick=%s drop_world=%s quat=%s pallet_center=%s",
            parsed.box_id,
            box.shape,
            gx,
            gy,
            gz,
            pallet_idx,
            pick_pose,
            drop_world,
            drop_quat,
            self._pallet_centers[pallet_idx],
        )

        await self._sim.pick_and_place(
            parsed.box_id,
            parsed.speed_pct,
            pick_pose,
            list(drop_world),
            drop_quaternion=drop_quat,
        )

        # Update pallet state (use corrected position, not parsed.position)
        self._pallets[pallet_idx] = place_box(
            self._pallets[pallet_idx],
            box.shape,
            (gx, gy, gz),
            box.weight_kg,
        )
        logger.info(
            "Pallet %d updated: fill=%.1f%% weight=%.1f kg",
            pallet_idx,
            fill_fraction(self._pallets[pallet_idx]) * 100.0,
            self._pallets[pallet_idx].total_weight_kg,
        )

        # Remove box from stack
        self._box_stack = [b for b in self._box_stack if b.box_id != parsed.box_id]
        remaining = [b.box_id for b in self._box_stack]
        logger.info("Box %s removed from stack, remaining: %s", parsed.box_id, remaining)

    async def _execute_call_human(self, parsed: ParsedAction) -> None:
        """Execute a CALL_A_HUMAN action: remove flagged boxes."""
        for box_id in parsed.box_ids:
            await self._sim.remove_box(box_id)
            self._box_stack = [b for b in self._box_stack if b.box_id != box_id]

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def _broadcast_box_stack(self) -> None:
        """Broadcast current box stack as images for the UI."""
        images = [
            {
                "data": f"data:image/png;base64,{base64.b64encode(b.image_bytes).decode('ascii')}",
                "label": (
                    f"{b.box_id} ({b.weight_kg:.1f}kg, "
                    f"{b.width_cm:.0f}x{b.length_cm:.0f}x{b.height_cm:.0f}cm)"
                ),
            }
            for b in self._box_stack
        ]
        await self._broadcast({"type": "box_images", "images": images})
