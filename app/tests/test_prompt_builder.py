"""Tests for prompt_builder -- multimodal message construction."""

from __future__ import annotations

from dr_ai_palletizer.domain.pallet import PalletState
from dr_ai_palletizer.prompt_builder import build_messages, build_scenario_text


def _make_box_image(box_id: str = "box_0001", weight: float = 10.0) -> dict:
    """Create a minimal box image dict as returned by sim."""
    return {
        "box_id": box_id,
        "image_bytes": b"\x89PNG_FAKE",
        "weight_kg": weight,
        "size": [0.50, 0.25, 0.25],  # 2d x 1d x 1d
    }


class TestBuildScenarioText:
    def test_contains_step_number(self) -> None:
        pallets = [PalletState.empty(0), PalletState.empty(1)]
        boxes = [_make_box_image()]
        text = build_scenario_text(boxes, pallets, step_number=3, last_action="WAIT")
        assert "STEP 3" in text
        assert "last action: WAIT" in text

    def test_contains_box_metadata(self) -> None:
        pallets = [PalletState.empty(0), PalletState.empty(1)]
        boxes = [_make_box_image("box_0042", 15.5)]
        text = build_scenario_text(boxes, pallets, step_number=1, last_action=None)
        assert "box_0042" in text
        assert "15.5" in text

    def test_contains_pallet_fill(self) -> None:
        pallets = [PalletState.empty(0), PalletState.empty(1)]
        boxes = [_make_box_image()]
        text = build_scenario_text(boxes, pallets, step_number=1, last_action=None)
        assert "fill: 0%" in text


class TestBuildMessages:
    def test_returns_system_and_user(self) -> None:
        pallets = [PalletState.empty(0), PalletState.empty(1)]
        boxes = [_make_box_image()]
        msgs = build_messages(boxes, pallets, step_number=1, last_action=None)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_content_has_image_and_text(self) -> None:
        pallets = [PalletState.empty(0), PalletState.empty(1)]
        boxes = [_make_box_image(), _make_box_image("box_0002", 5.0)]
        msgs = build_messages(boxes, pallets, step_number=1, last_action=None)
        content = msgs[1]["content"]
        # 2 images + 1 text part
        assert len(content) == 3
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"

    def test_images_are_base64_data_urls(self) -> None:
        pallets = [PalletState.empty(0), PalletState.empty(1)]
        boxes = [_make_box_image()]
        msgs = build_messages(boxes, pallets, step_number=1, last_action=None)
        img_part = msgs[1]["content"][0]
        url = img_part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_default_prompt_has_no_example(self) -> None:
        pallets = [PalletState.empty(0), PalletState.empty(1)]
        boxes = [_make_box_image()]
        msgs = build_messages(boxes, pallets, step_number=1, last_action=None)
        text = msgs[1]["content"][-1]["text"]
        assert "--- EXAMPLE ---" not in text

    def test_few_shot_prompt_includes_example(self) -> None:
        pallets = [PalletState.empty(0), PalletState.empty(1)]
        boxes = [_make_box_image()]
        msgs = build_messages(boxes, pallets, step_number=1, last_action=None, use_few_shot=True)
        text = msgs[1]["content"][-1]["text"]
        assert "--- EXAMPLE ---" in text
        assert "aluminum cans" in text
