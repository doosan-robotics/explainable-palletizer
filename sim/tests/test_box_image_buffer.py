"""Tests for BoxImageCapture in-memory buffer and drain."""

from __future__ import annotations

from drp_sim.box_image_capture import BoxImageCapture


class TestBoxImageBuffer:
    def test_drain_returns_empty_initially(self) -> None:
        cap = BoxImageCapture.__new__(BoxImageCapture)
        cap._buffer = []
        assert cap.drain() == []

    def test_drain_returns_and_clears(self) -> None:
        cap = BoxImageCapture.__new__(BoxImageCapture)
        cap._buffer = [{"box_id": "box_0000", "image_b64": "abc"}]
        result = cap.drain()
        assert len(result) == 1
        assert result[0]["box_id"] == "box_0000"
        assert cap.drain() == []  # cleared

    def test_buffer_mode_init(self) -> None:
        """BoxImageCapture with output_dir=None enters buffer-only mode."""
        cap = BoxImageCapture(output_dir=None)
        assert cap._buffer == []
        assert cap._output_dir is None
