"""Parse structured actions from model responses.

Supports multiple response formats:
  1. ``<answer>JSON</answer>`` blocks (preferred, from fine-tuned models)
  2. Markdown fenced JSON (```json ... ```)
  3. Raw JSON objects (``{...}``)

An optional ``<think>`` block is extracted as reasoning context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_RAW_JSON_RE = re.compile(r"\{[^{}]*\"action\"[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class ParsedAction:
    """Structured representation of a model action response."""

    thinking: str = ""
    action: str = ""
    box_id: str | None = None
    box_ids: list[str] = field(default_factory=list)
    target_pallet: int | None = None
    position: tuple[int, int, int] | None = None
    speed_pct: int = 100
    grip_strength: str = "standard"
    reason: str = ""


def _extract_json(response: str) -> str:
    """Try multiple strategies to find a JSON action object in the response."""
    # Strategy 1: <answer> block — search for raw JSON within the block
    # because the LLM sometimes prefixes with "For PICK_AND_PLACE:\n{...}"
    m = _ANSWER_RE.search(response)
    if m:
        answer_content = m.group(1).strip()
        # Try raw JSON search within the answer block first
        m2 = _RAW_JSON_RE.search(answer_content)
        if m2:
            return m2.group(0).strip()
        return answer_content

    # Strategy 2: markdown fenced JSON block
    m = _FENCED_JSON_RE.search(response)
    if m:
        return m.group(1).strip()

    # Strategy 3: raw JSON object containing "action"
    m = _RAW_JSON_RE.search(response)
    if m:
        return m.group(0).strip()

    msg = "No JSON action object found in response"
    raise ValueError(msg)


def parse_response(response: str, *, max_thinking_chars: int = 4000) -> ParsedAction:
    """Extract a ``ParsedAction`` from a model response string.

    Raises
    ------
    ValueError
        If no JSON action object is found, the JSON is invalid, or
        the JSON object lacks an ``"action"`` key.
    """
    # -- thinking (optional) --------------------------------------------------
    think_match = _THINK_RE.search(response)
    thinking = think_match.group(1).strip() if think_match else ""

    if thinking and len(thinking) > max_thinking_chars:
        thinking = thinking[:max_thinking_chars] + "..."

    # -- action JSON (multiple format strategies) -----------------------------
    raw_json = _extract_json(response)

    try:
        data: dict = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in action block: {exc}"
        raise ValueError(msg) from exc

    if "action" not in data:
        msg = "JSON object missing required 'action' field"
        raise ValueError(msg)

    action: str = data["action"]

    # -- action-specific fields -----------------------------------------------
    box_id: str | None = None
    box_ids: list[str] = []
    target_pallet: int | None = None
    position: tuple[int, int, int] | None = None
    speed_pct: int = 100
    grip_strength: str = "standard"
    reason: str = data.get("reason", "")

    if action == "PICK_AND_PLACE":
        box_id = data.get("box")
        target_pallet = data.get("target_pallet")
        raw_pos = data.get("position")
        if raw_pos is not None:
            position = tuple(raw_pos)  # type: ignore[arg-type]
        speed_pct = data.get("speed_pct", 100)
        grip_strength = data.get("grip_strength", "standard")

    elif action == "CALL_A_HUMAN":
        box_ids = data.get("boxes", [])

    return ParsedAction(
        thinking=thinking,
        action=action,
        box_id=box_id,
        box_ids=box_ids,
        target_pallet=target_pallet,
        position=position,
        speed_pct=speed_pct,
        grip_strength=grip_strength,
        reason=reason,
    )
