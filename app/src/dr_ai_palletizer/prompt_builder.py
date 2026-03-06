"""Build multimodal OpenAI messages for the palletizer control loop.

Converts raw box image dicts and PalletState objects into the structured
message format expected by multimodal LLM inference (OpenAI-compatible API).
"""

from __future__ import annotations

import base64

import numpy as np

from dr_ai_palletizer.domain.models import (
    COSMOS2_SYSTEM_PROMPT,
    D_UNIT_CM,
    PALLET_GRID_SIZE,
    Box,
    BoxShape,
    Scenario,
    select_task_prompt,
)
from dr_ai_palletizer.domain.pallet import (
    PalletState,
    fill_fraction,
    find_valid_positions,
)

# d-unit in meters
_D_UNIT_M: float = D_UNIT_CM / 100.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _size_to_shape(size: list[float]) -> BoxShape:
    """Convert [x_m, y_m, z_m] real size to BoxShape in d-units.

    Each axis is rounded to the nearest integer d-unit (1 or 2).
    """
    return BoxShape(
        w=max(1, round(size[0] / _D_UNIT_M)),
        ln=max(1, round(size[1] / _D_UNIT_M)),
        h=max(1, round(size[2] / _D_UNIT_M)),
    )


def _pallet_info(pallet: PalletState) -> dict:
    """Serialize a PalletState to the dict format Scenario expects."""
    total_cells = PALLET_GRID_SIZE**3
    occupied_cells = int(np.count_nonzero(pallet.grid))
    frac = fill_fraction(pallet) * 100.0
    return {
        "id": pallet.id,
        "total_weight_kg": pallet.total_weight_kg,
        "max_weight_kg": pallet.max_weight_kg,
        "occupied_cells": occupied_cells,
        "total_cells": total_cells,
        "fill_pct": frac,
    }


def _box_image_to_box(box_img: dict) -> Box:
    """Convert a box image dict from sim into a Box dataclass."""
    shape = _size_to_shape(box_img["size"])
    return Box(
        id=box_img["box_id"],
        weight_kg=box_img["weight_kg"],
        shape=shape,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_scenario_text(
    box_images: list[dict],
    pallets: list[PalletState],
    *,
    step_number: int,
    last_action: str | None,
) -> str:
    """Build the scenario text prompt from box images and pallet states.

    Args:
        box_images: List of box dicts with keys box_id, image_bytes,
            weight_kg, size.
        pallets: List of PalletState objects.
        step_number: Current step number in the control loop.
        last_action: The last action taken, or None.

    Returns:
        Formatted scenario text suitable for LLM prompt.
    """
    boxes = [_box_image_to_box(img) for img in box_images]
    pallet_infos = [_pallet_info(p) for p in pallets]

    # Compute valid positions: box_id -> {pallet_id -> [(x,y,z), ...]}
    valid_positions: dict[str, dict[int, list[tuple[int, int, int]]]] = {}
    for box in boxes:
        positions_per_pallet: dict[int, list[tuple[int, int, int]]] = {}
        for pallet in pallets:
            positions = find_valid_positions(pallet, box.shape, box.weight_kg)
            positions_per_pallet[pallet.id] = positions
        valid_positions[box.id] = positions_per_pallet

    scenario = Scenario(
        step_number=step_number,
        boxes=boxes,
        pallets=pallet_infos,
        valid_positions=valid_positions,
        boxes_remaining=len(boxes),
        last_action=last_action,
    )
    return scenario.to_text()


def build_messages(
    box_images: list[dict],
    pallets: list[PalletState],
    *,
    step_number: int,
    last_action: str | None,
    use_few_shot: bool = False,
) -> list[dict]:
    """Build OpenAI multimodal messages for inference.

    Returns a two-element list: [system_message, user_message].
    The user message contains base64-encoded images followed by the
    scenario text.

    Args:
        box_images: List of box dicts with keys box_id, image_bytes,
            weight_kg, size.
        pallets: List of PalletState objects.
        step_number: Current step number in the control loop.
        last_action: The last action taken, or None.
        use_few_shot: When True, use the few-shot prompt with an example
            for base models without a LoRA adapter.

    Returns:
        List of two message dicts in OpenAI chat format.
    """
    task_prompt = select_task_prompt(use_few_shot=use_few_shot)

    system_msg: dict = {
        "role": "system",
        "content": COSMOS2_SYSTEM_PROMPT,
    }

    # Build multimodal content parts: images first, then text
    content_parts: list[dict] = []

    for box_img in box_images:
        b64 = base64.b64encode(box_img["image_bytes"]).decode("ascii")
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    scenario_text = build_scenario_text(
        box_images,
        pallets,
        step_number=step_number,
        last_action=last_action,
    )
    text_content = f"{task_prompt}\n\n{scenario_text}"
    content_parts.append({"type": "text", "text": text_content})

    user_msg: dict = {
        "role": "user",
        "content": content_parts,
    }

    return [system_msg, user_msg]
