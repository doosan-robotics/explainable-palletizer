"""Thread-safe single-frame buffer for camera streaming.

The main-thread sim loop writes JPEG frames via ``put()``.
WebSocket handlers on the uvicorn thread read via ``get()``.
Capture is skipped when ``active`` is False (no subscribers).
"""

from __future__ import annotations

import threading


class FrameBuffer:
    """Lock-protected buffer holding the most recent JPEG frame.

    Subscribers register via ``subscribe()`` / ``unsubscribe()`` so the
    sim loop can skip expensive JPEG encoding when nobody is watching.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: bytes = b""
        self._width: int = 0
        self._height: int = 0
        self._frame_id: int = 0
        self._subscriber_count: int = 0

    @property
    def active(self) -> bool:
        """True when at least one WebSocket client is streaming."""
        return self._subscriber_count > 0

    @property
    def subscriber_count(self) -> int:
        """Current number of streaming subscribers."""
        return self._subscriber_count

    def subscribe(self) -> None:
        with self._lock:
            self._subscriber_count += 1

    def unsubscribe(self) -> None:
        with self._lock:
            self._subscriber_count = max(0, self._subscriber_count - 1)

    def put(self, frame: bytes, width: int, height: int) -> None:
        """Store a new JPEG frame (called from main thread)."""
        with self._lock:
            self._frame = frame
            self._width = width
            self._height = height
            self._frame_id += 1

    def get(self) -> tuple[bytes, int, int, int]:
        """Read the latest frame. Returns (jpeg_bytes, width, height, frame_id)."""
        with self._lock:
            return self._frame, self._width, self._height, self._frame_id
