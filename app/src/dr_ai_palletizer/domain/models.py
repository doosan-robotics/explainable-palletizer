"""Domain models and constants for palletizer control.

Runtime types only. At inference time Isaac Sim provides: image, size, weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

D_UNIT_CM: float = 25.0
"""Side length of the base unit d in centimeters. All box axes are d or 2d."""

PALLET_GRID_SIZE: int = 4
"""Each pallet is a PALLET_GRID_SIZE * d cube per axis."""


class BoxShape(NamedTuple):
    """Discrete box dimensions in d-units (each axis is 1 or 2)."""

    w: int  # width in d-units
    ln: int  # length in d-units
    h: int  # height in d-units


ALL_BOX_SHAPES: tuple[BoxShape, ...] = tuple(
    BoxShape(w, ln, h) for w in (1, 2) for ln in (1, 2) for h in (1, 2)
)


class RobotConfig(NamedTuple):
    max_payload_kg: float = 30.0


class PalletConfig(NamedTuple):
    num_pallets: int = 2
    grid_size: int = 4
    max_weight_kg: float = 500.0


ROBOT = RobotConfig()
PALLET = PalletConfig()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Box:
    """A box on the conveyor as seen by the runtime control loop.

    Fields come directly from sim: id, weight, and shape (derived from size).
    """

    id: str
    weight_kg: float
    shape: BoxShape
    observations: list[str] = field(default_factory=list)

    @property
    def width_cm(self) -> float:
        return self.shape.w * D_UNIT_CM

    @property
    def length_cm(self) -> float:
        return self.shape.ln * D_UNIT_CM

    @property
    def height_cm(self) -> float:
        return self.shape.h * D_UNIT_CM

    def to_text(self) -> str:
        w, ln, h = self.shape
        lines = [
            f"[{self.id}] {self.weight_kg:.1f} kg "
            f"| {w}d x {ln}d x {h}d "
            f"({self.width_cm:.0f} x {self.length_cm:.0f} x {self.height_cm:.0f} cm)"
        ]
        for obs in self.observations:
            lines.append(f"    - {obs}")
        return "\n".join(lines)


@dataclass
class Scenario:
    step_number: int
    boxes: list[Box]
    pallets: list[dict]
    valid_positions: dict[str, dict[int, list[tuple[int, int, int]]]]
    boxes_remaining: int
    last_action: str | None = None

    def to_text(self) -> str:
        last_str = self.last_action or "none"
        lines = [f"=== STEP {self.step_number} === (last action: {last_str})", ""]
        lines.append("BOXES:")
        for i, box in enumerate(self.boxes):
            lines.append(f"  BOX {i + 1}: {box.to_text()}")
            lines.append("")

        lines.append("PALLETS:")
        for p_info in self.pallets:
            pid = p_info["id"]
            weight = p_info["total_weight_kg"]
            max_w = p_info["max_weight_kg"]
            occupied = p_info["occupied_cells"]
            total = p_info["total_cells"]
            frac = p_info["fill_pct"]
            lines.append(
                f"  Pallet {pid + 1}: {weight:.1f}/{max_w:.0f} kg "
                f"| {occupied}/{total} cells occupied | fill: {frac:.0f}%"
            )
            for box in self.boxes:
                positions = self.valid_positions.get(box.id, {}).get(pid, [])
                if positions:
                    pos_str = ", ".join(f"({x},{y},{z})" for x, y, z in positions[:8])
                    if len(positions) > 8:
                        pos_str += f" ... (+{len(positions) - 8} more)"
                    lines.append(f"    {box.id} positions: {pos_str}")
                else:
                    lines.append(f"    {box.id} positions: (none)")
            lines.append("")

        lines.append(f"ROBOT: payload {ROBOT.max_payload_kg:.0f} kg | single pick")
        lines.append("")
        lines.append("Decide the next action. Check for damaged/unpickable boxes first.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cosmos-Reason2 prompt format
# ---------------------------------------------------------------------------

COSMOS2_SYSTEM_PROMPT: str = "You are a helpful assistant."

_TASK_INSTRUCTIONS: str = (
    "You are controlling a palletizing robot. At each step you see 1-3 "
    "individual box images along with their measured weight and dimensions, "
    "plus the current state of 2 pallets.\n\n"
    f"Box dimensions use a unit 'd' where d = {D_UNIT_CM:.0f} cm. Each box axis "
    f"is either 1d or 2d. Pallets are 4d x 4d x 4d grids ({PALLET_GRID_SIZE}d per axis). "
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
    "not WAIT.\n\n"
    "Each step includes what your last action was. You cannot WAIT twice "
    "in a row.\n\n"
)

_ANSWER_FORMAT: str = (
    "Answer the question using the following format:\n\n"
    "<think>\nYour reasoning.\n</think>\n\n"
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

# Prompt for LoRA-tuned models (no example needed -- learned from training data)
COSMOS2_TASK_PROMPT: str = _TASK_INSTRUCTIONS + _ANSWER_FORMAT

# ---------------------------------------------------------------------------
# Few-shot example for base models (no LoRA adapter)
# ---------------------------------------------------------------------------
# When running without a fine-tuned adapter the base model has no exposure to
# observation-based reasoning about box conditions.  This example teaches
# the model to: (1) identify box contents from images (glass, cans, etc.),
# (2) infer handling requirements from content type (fragile, heavy, normal),
# (3) detect packaging failures (unsealed, torn), (4) prioritize safety.

_FEW_SHOT_EXAMPLE: str = (
    "Here is an example of how you should reason about box conditions.\n\n"
    "--- EXAMPLE ---\n"
    "=== STEP 5 === (last action: PICK_AND_PLACE)\n\n"
    "BOXES:\n"
    "  BOX 1: [BOX_0005_0] 16.2 kg | 2d x 1d x 1d (50 x 25 x 25 cm)\n"
    "  BOX 2: [BOX_0005_1] 4.8 kg | 1d x 1d x 1d (25 x 25 x 25 cm)\n"
    "  BOX 3: [BOX_0005_2] 9.1 kg | 2d x 2d x 1d (50 x 50 x 25 cm)\n\n"
    "PALLETS:\n"
    "  Pallet 1: 45.0/500 kg | 12/64 cells occupied | fill: 19%\n"
    "    BOX_0005_0 positions: (0,0,1), (2,0,1), (0,2,1)\n"
    "    BOX_0005_1 positions: (0,0,2), (1,0,2), (0,1,2)\n"
    "    BOX_0005_2 positions: (0,0,1)\n"
    "  Pallet 2: 120.3/500 kg | 28/64 cells occupied | fill: 44%\n"
    "    BOX_0005_0 positions: (0,2,2), (2,0,2)\n"
    "    BOX_0005_1 positions: (0,0,3), (1,0,3), (0,2,3)\n"
    "    BOX_0005_2 positions: (none)\n\n"
    "ROBOT: payload 30 kg | single pick\n\n"
    "Decide the next action. Check for damaged/unpickable boxes first.\n\n"
    "<think>\n"
    "Assessing each visible box:\n\n"
    "BOX 1 (BOX_0005_0): 16.2 kg, 2d x 1d x 1d. The image shows a box "
    "containing 24 rigid aluminum cans of carbonated drink, tightly packed "
    "in a standard corrugated case. Sealing tape is intact along the center "
    "seam with no dents, tears, or deformation. Aluminum cans are sturdy "
    "and tolerant of compression. Weight is moderate -- standard grip and "
    "full speed are appropriate.\n\n"
    "BOX 2 (BOX_0005_1): 4.8 kg, 1d x 1d x 1d. I can see glass beer "
    "bottles through the packaging -- six dark glass bottles with crown "
    "caps visible through the cutout handle. Glass is fragile and can "
    "crack from impact or excessive grip pressure. The box is lightweight "
    "at 4.8 kg which is consistent with glass bottles. Needs gentle grip "
    "and reduced speed to avoid breakage. Should be placed at a higher "
    "z-position so heavier boxes do not crush the glass.\n\n"
    "BOX 3 (BOX_0005_2): 9.1 kg, 2d x 2d x 1d. The box is not taped "
    "and remains folded flat without any sealing -- the flaps are open and "
    "the contents are partially exposed. The structural integrity of the "
    "packaging is compromised: the gripper cannot safely lift an unsealed "
    "box without risk of contents spilling. This box is unpickable.\n\n"
    "Priority check: BOX 3 has compromised packaging (unsealed, flaps "
    "open) -- must route to human inspection immediately.\n\n"
    "Remaining boxes: BOX 1 is normal (aluminum cans, sturdy), BOX 2 is "
    "fragile (glass bottles, needs gentle handling). After routing BOX 3 "
    "to human, CALL_A_HUMAN takes priority this step.\n"
    "</think>\n\n"
    "<answer>\n"
    '{"action": "CALL_A_HUMAN", "boxes": ["BOX_0005_2"], '
    '"reason": "BOX_0005_2: box is unsealed with flaps open and contents '
    'exposed -- packaging integrity compromised, cannot safely grip"}\n'
    "</answer>\n"
    "--- END EXAMPLE ---\n\n"
    "Now handle the current scenario below.\n\n"
)

COSMOS2_TASK_PROMPT_FEW_SHOT: str = _TASK_INSTRUCTIONS + _FEW_SHOT_EXAMPLE + _ANSWER_FORMAT


def select_task_prompt(*, use_few_shot: bool) -> str:
    """Return the task prompt appropriate for the current adapter config."""
    return COSMOS2_TASK_PROMPT_FEW_SHOT if use_few_shot else COSMOS2_TASK_PROMPT
