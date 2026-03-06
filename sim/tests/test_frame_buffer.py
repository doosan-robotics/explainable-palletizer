"""Tests for the thread-safe FrameBuffer."""

from __future__ import annotations

from drp_sim.frame_buffer import FrameBuffer


class TestFrameBuffer:
    def test_initially_empty(self) -> None:
        buf = FrameBuffer()
        frame, w, h, fid = buf.get()
        assert frame == b""
        assert w == 0
        assert h == 0
        assert fid == 0

    def test_put_and_get(self) -> None:
        buf = FrameBuffer()
        buf.put(b"\xff\xd8jpeg", 640, 480)
        frame, w, h, fid = buf.get()
        assert frame == b"\xff\xd8jpeg"
        assert w == 640
        assert h == 480
        assert fid == 1

    def test_overwrites_previous(self) -> None:
        buf = FrameBuffer()
        buf.put(b"frame1", 100, 100)
        buf.put(b"frame2", 200, 200)
        frame, w, _h, fid = buf.get()
        assert frame == b"frame2"
        assert w == 200
        assert fid == 2

    def test_subscriber_tracking(self) -> None:
        buf = FrameBuffer()
        assert buf.active is False

        buf.subscribe()
        assert buf.active is True

        buf.subscribe()
        assert buf.active is True

        buf.unsubscribe()
        assert buf.active is True

        buf.unsubscribe()
        assert buf.active is False

    def test_unsubscribe_below_zero(self) -> None:
        buf = FrameBuffer()
        buf.unsubscribe()
        buf.unsubscribe()
        assert buf.active is False
        assert buf.subscriber_count == 0

    def test_subscriber_count_property(self) -> None:
        buf = FrameBuffer()
        assert buf.subscriber_count == 0
        buf.subscribe()
        buf.subscribe()
        assert buf.subscriber_count == 2
        buf.unsubscribe()
        assert buf.subscriber_count == 1
