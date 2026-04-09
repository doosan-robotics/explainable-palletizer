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

COSMOS2_SYSTEM_PROMPT: str = (
    "You are an expert warehouse logistics AI controlling a palletizing "
    "robot. You have deep knowledge of packaging materials, product types, "
    "and safe handling practices. You reason carefully about what each box "
    "contains based on visible text, branding, imagery, and handling symbols, "
    "then use that knowledge to make safe stacking decisions."
)

_TASK_INSTRUCTIONS: str = (
    "You are controlling a palletizing robot. At each step you see 1-3 "
    "individual box images along with their measured weight and dimensions, "
    "plus the current state of 2 pallets.\n\n"
    f"Box dimensions use a unit 'd' where d = {D_UNIT_CM:.0f} cm. Each box axis "
    f"is either 1d or 2d. Pallets are 4d x 4d x 4d grids ({PALLET_GRID_SIZE}d per axis). "
    "Valid placement positions are pre-computed and listed for each box on "
    "each pallet.\n\n"
    "CRITICAL: You MUST base your reasoning on what you actually SEE in "
    "each box image. Describe specific visual details -- colors, brand "
    "names, product imagery, label text, tape condition, flap position. "
    "Do NOT assume or infer box condition from the scenario text alone. "
    "The images are your primary source of truth.\n\n"
    "Your process:\n"
    "1. OBSERVE: For each box image, describe what you SEE:\n"
    "   a) READ all visible text: brand names, product names, descriptions, "
    'handling instructions ("Fragile", "This Side Up", "Handle With Care", '
    '"Keep Dry"), weight markings, and warning labels.\n'
    "   b) IDENTIFY the contents: use branding, product imagery, and text to "
    'determine what is inside (e.g. "Coca-Cola" = glass bottles, '
    '"Samsung TV" = electronics with screen, "Pampers" = lightweight diapers).\n'
    "   c) INSPECT the packaging: check tape integrity, flap closure, "
    "structural condition, wet spots, crushing, and deformation.\n"
    "2. CHECK FIRST: Immediately CALL_A_HUMAN for ANY box showing ACTUAL "
    "packaging damage -- this takes absolute priority:\n"
    "   - Flaps or lid partially or fully open (not sealed shut)\n"
    "   - Flaps resting closed but NO tape securing them (they will open "
    "when the gripper lifts the box)\n"
    "   - Tape torn, peeling, missing, or not sealing the box\n"
    "   - Contents spilling out or falling through a hole or tear\n"
    "   - Box crushed, deformed, or structurally compromised\n"
    "   - Wet, stained, or contaminated packaging\n"
    "   IMPORTANT: Being able to see product branding, images, or text on "
    "the outside of a box is NORMAL -- that is printing, not damage. "
    "Cutout handles or display windows designed into the packaging are also "
    "normal. Only flag a box if the packaging is actually BROKEN, TORN, or "
    "UNSEALED and contents could fall out during lifting.\n"
    "3. ASSESS: For each intact box, reason step by step:\n"
    "   a) CONTENTS: What is inside? (inferred from labels, branding, imagery)\n"
    "   b) MATERIAL: What are the contents made of? "
    "(glass, electronics, metal cans, plastic, paper, liquid, etc.)\n"
    "   c) FRAGILITY: How breakable are the contents? "
    "(glass bottles = very fragile, cans = sturdy, electronics = fragile, "
    "paper goods = tolerant)\n"
    "   d) HANDLING: Set speed_pct and grip_strength:\n"
    "      - Fragile contents (glass, electronics): speed 30-50%, gentle grip\n"
    "      - Normal contents (cans, plastic): speed 70-100%, standard grip\n"
    "      - Heavy/sturdy contents (metal, tools): speed 70-100%, firm grip\n"
    "   e) PLACEMENT: Choose position from the VALID POSITIONS list only:\n"
    "      - You MUST pick a position from the valid positions listed for "
    "that box on that pallet. These are pre-computed to ensure physical "
    "support. You cannot invent positions.\n"
    "      - PREFER low z for heavy boxes (stability) and high z for "
    "fragile boxes (avoid crushing) -- but only if those z-levels appear "
    "in the valid positions.\n"
    "      - If a pallet is empty, only z=0 positions exist. Place sturdy/"
    "heavy boxes first to build a base, then place fragile boxes on top "
    "in later steps.\n"
    "      - If a box is both heavy AND fragile (e.g. glass bottles), "
    "prefer low z but use gentle handling.\n"
    "4. DECIDE: Choose the safest and most efficient action:\n"
    "   - Prioritize the pallet closer to completion (higher fill %).\n"
    "   - Apply the handling and placement reasoning from step 3.\n"
    "   - You can only place ONE box per step. Pick the best single action.\n"
    "   - Consider lateral support and stacking stability.\n\n"
    "Available actions:\n"
    "- PICK_AND_PLACE: Pick one box and place it at a specific grid position "
    "on a pallet. Adjust speed and grip strength based on box contents.\n"
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
    "Answer using this EXACT format. In the <think> block you MUST include "
    "a separate visual inspection for EACH box -- do not make blanket "
    "statements about all boxes at once:\n\n"
    "<think>\n"
    "BOX 1 (<BOX_ID>):\n"
    "  I SEE: [describe specific visual details from the image -- colors, "
    "text, brand, product imagery]\n"
    "  PACKAGING: [describe tape, flaps, sealing -- is it sealed shut? "
    "are flaps open? is tape intact?]\n"
    "  CONTENTS: [what is inside based on what you read/see]\n"
    "  MATERIAL: [glass/metal/plastic/electronics/etc]\n"
    "  VERDICT: [pickable or unpickable, and why]\n\n"
    "(repeat for each box)\n\n"
    "DECISION: [which action to take and why]\n"
    "</think>\n\n"
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
    "Here are two examples showing how to reason and act.\n\n"
    "--- EXAMPLE 1: CALL_A_HUMAN (damaged box) ---\n"
    "BOXES:\n"
    "  BOX 1: [BOX_A] 9.1 kg | 2d x 2d x 1d (50 x 50 x 25 cm)\n\n"
    "PALLETS:\n"
    "  Pallet 1: 45.0/500 kg | 12/64 cells occupied | fill: 19%\n"
    "    BOX_A positions: (0,0,1), (2,0,1)\n"
    "  Pallet 2: 0.0/500 kg | 0/64 cells occupied | fill: 0%\n"
    "    BOX_A positions: (0,0,0)\n\n"
    "ROBOT: payload 30 kg | single pick\n\n"
    "Decide the next action. Check for damaged/unpickable boxes first.\n\n"
    "<think>\n"
    "BOX 1 (BOX_A): 9.1 kg, 2d x 2d x 1d.\n"
    "  I SEE: Beige cardboard box with small labels and stickers on the "
    "side. The top flaps are slightly ajar -- there is a visible gap "
    "between the flaps. Tape appears partially detached along one edge.\n"
    "  PACKAGING: The flaps are not fully closed. Tape is peeling off "
    "and not properly sealing the box. There is a gap at the top where "
    "I can see a sliver of the contents inside. The box is not securely "
    "sealed.\n"
    "  CONTENTS: Cannot confirm -- packaging is not secure.\n"
    "  MATERIAL: Unknown.\n"
    "  VERDICT: UNPICKABLE. Flaps slightly open, tape partially "
    "detached. Even a small gap means the gripper could cause contents "
    "to shift or spill during lifting.\n\n"
    "DECISION: BOX_A has partially open flaps and failing tape -- "
    "CALL_A_HUMAN.\n"
    "</think>\n\n"
    "<answer>\n"
    '{"action": "CALL_A_HUMAN", "boxes": ["BOX_A"], '
    '"reason": "flaps slightly open and tape partially detached -- '
    'not securely sealed, risk of contents shifting during lift"}\n'
    "</answer>\n"
    "--- END EXAMPLE 1 ---\n\n"
    "--- EXAMPLE 2: PICK_AND_PLACE (intact boxes) ---\n"
    "BOXES:\n"
    "  BOX 1: [BOX_B] 16.2 kg | 2d x 1d x 1d (50 x 25 x 25 cm)\n"
    "  BOX 2: [BOX_C] 4.8 kg | 1d x 1d x 1d (25 x 25 x 25 cm)\n\n"
    "PALLETS:\n"
    "  Pallet 1: 45.0/500 kg | 12/64 cells occupied | fill: 19%\n"
    "    BOX_B positions: (0,0,1), (2,0,1)\n"
    "    BOX_C positions: (0,0,2), (1,0,2)\n"
    "  Pallet 2: 120.3/500 kg | 28/64 cells occupied | fill: 44%\n"
    "    BOX_B positions: (0,2,2)\n"
    "    BOX_C positions: (0,0,3), (1,0,3)\n\n"
    "ROBOT: payload 30 kg | single pick\n\n"
    "Decide the next action. Check for damaged/unpickable boxes first.\n\n"
    "<think>\n"
    "BOX 1 (BOX_B): 16.2 kg, 2d x 1d x 1d.\n"
    '  I SEE: Blue box with "Pepsi" logo and imagery of aluminum cans.\n'
    "  PACKAGING: Tape intact along center seam. All flaps sealed shut. "
    "No dents.\n"
    "  CONTENTS: 24-pack of Pepsi cans.\n"
    "  MATERIAL: Aluminum cans -- metal, rigid, sturdy.\n"
    "  VERDICT: Pickable. speed 100%, standard grip.\n\n"
    "BOX 2 (BOX_C): 4.8 kg, 1d x 1d x 1d.\n"
    '  I SEE: Green box with "Heineken" branding. Glass bottles visible '
    "through cutout handle.\n"
    "  PACKAGING: Intact. Cutout handle is a designed feature, not damage.\n"
    "  CONTENTS: 6-pack of Heineken glass beer bottles.\n"
    "  MATERIAL: Glass -- very fragile.\n"
    "  VERDICT: Pickable. speed 40%, gentle grip.\n\n"
    "DECISION: No damaged boxes. Pallet 2 is closer to completion (44% "
    "vs 19%). BOX_B is heavy and sturdy -- place at (0,2,2) on Pallet 2 "
    "(from valid positions). Standard handling for aluminum cans.\n"
    "</think>\n\n"
    "<answer>\n"
    '{"action": "PICK_AND_PLACE", "box": "BOX_B", "target_pallet": 2, '
    '"position": [0, 2, 2], "speed_pct": 100, '
    '"grip_strength": "standard", '
    '"reason": "Pepsi cans are sturdy, Pallet 2 closer to completion"}\n'
    "</answer>\n"
    "--- END EXAMPLE 2 ---\n\n"
    "Now handle the current scenario below.\n\n"
)

COSMOS2_TASK_PROMPT_FEW_SHOT: str = _TASK_INSTRUCTIONS + _FEW_SHOT_EXAMPLE + _ANSWER_FORMAT


def select_task_prompt(*, use_few_shot: bool) -> str:
    """Return the task prompt appropriate for the current adapter config."""
    return COSMOS2_TASK_PROMPT_FEW_SHOT if use_few_shot else COSMOS2_TASK_PROMPT
