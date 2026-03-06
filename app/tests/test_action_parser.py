"""Tests for model response action parser."""

from __future__ import annotations

import textwrap

import pytest
from dr_ai_palletizer.action_parser import ParsedAction, parse_response


class TestParseResponse:
    def test_extracts_think_and_action(self) -> None:
        response = textwrap.dedent("""\
            <think>
            Box is heavy at 22 kg.
            </think>

            <answer>
            {
              "action": "PICK_AND_PLACE",
              "box": "box_0001",
              "target_pallet": 1,
              "position": [2, 0, 0],
              "speed_pct": 70,
              "grip_strength": "standard",
              "reason": "heavy box at low z"
            }
            </answer>
        """)
        result = parse_response(response)

        assert isinstance(result, ParsedAction)
        assert result.thinking == "Box is heavy at 22 kg."
        assert result.action == "PICK_AND_PLACE"
        assert result.box_id == "box_0001"
        assert result.target_pallet == 1
        assert result.position == (2, 0, 0)
        assert result.speed_pct == 70
        assert result.grip_strength == "standard"
        assert result.reason == "heavy box at low z"
        assert result.box_ids == []

    def test_strips_whitespace_in_think(self) -> None:
        response = textwrap.dedent("""\
            <think>
                  Lots of padding here.
            </think>

            <answer>
            {"action": "WAIT", "reason": "idle"}
            </answer>
        """)
        result = parse_response(response)
        assert result.thinking == "Lots of padding here."

    def test_extracts_box_ids(self) -> None:
        response = textwrap.dedent("""\
            <think>
            Multiple damaged boxes.
            </think>

            <answer>
            {
              "action": "CALL_A_HUMAN",
              "boxes": ["box_0010", "box_0011", "box_0012"],
              "reason": "damaged"
            }
            </answer>
        """)
        result = parse_response(response)

        assert result.action == "CALL_A_HUMAN"
        assert result.box_ids == ["box_0010", "box_0011", "box_0012"]
        assert result.reason == "damaged"
        assert result.box_id is None
        assert result.target_pallet is None
        assert result.position is None

    def test_extracts_wait(self) -> None:
        response = textwrap.dedent("""\
            <answer>
            {"action": "WAIT", "reason": "conveyor empty"}
            </answer>
        """)
        result = parse_response(response)

        assert result.action == "WAIT"
        assert result.reason == "conveyor empty"
        assert result.thinking == ""
        assert result.box_id is None
        assert result.box_ids == []
        assert result.target_pallet is None
        assert result.position is None
        assert result.speed_pct == 100
        assert result.grip_strength == "standard"

    def test_missing_answer_tag_raises(self) -> None:
        response = textwrap.dedent("""\
            <think>
            Some thinking here.
            </think>

            No answer block at all.
        """)
        with pytest.raises(ValueError, match="No JSON action object found"):
            parse_response(response)

    def test_invalid_json_raises(self) -> None:
        response = textwrap.dedent("""\
            <answer>
            {not valid json at all}
            </answer>
        """)
        with pytest.raises(ValueError, match="JSON"):
            parse_response(response)

    def test_missing_action_field_raises(self) -> None:
        response = textwrap.dedent("""\
            <answer>
            {"box": "box_0001", "target_pallet": 1}
            </answer>
        """)
        with pytest.raises(ValueError, match="action"):
            parse_response(response)

    def test_long_thinking_is_truncated(self) -> None:
        long_text = "x" * 5000
        answer_json = '{"action": "WAIT", "reason": "idle"}'
        response = f"<think>\n{long_text}\n</think>\n\n<answer>\n{answer_json}\n</answer>"

        result = parse_response(response, max_thinking_chars=100)
        assert len(result.thinking) == 103  # 100 chars + "..."
        assert result.thinking.endswith("...")
        assert result.thinking[:100] == "x" * 100
