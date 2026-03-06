"""Box image capture using Omni Replicator annotators.

Captures RGB images of spawned boxes (with stickers) on the conveyor
belt.  A camera is placed diagonally above and to the +X side of the
spawn point at ~45 degrees, capturing both the top face and the +X side
face where stickers are attached.  One image is saved per box shortly
after it appears on the belt.

Uses an RGB annotator attached to a render product so that pixel data
is read directly from the render buffer after ``world.step(render=True)``
without calling ``rep.orchestrator.step()`` (which can interfere with the
physics timeline).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

_DEFAULT_CAMERA_OFFSET_Z = 3.5  # metres from spawn point (diagonal)
_DEFAULT_RESOLUTION = (960, 704)
_CAPTURE_DELAY_STEPS = 10  # frames after spawn before capturing


class BoxImageCapture:
    """Captures RGB images of conveyor boxes via Omni Replicator annotator.

    Places a camera diagonally above and to the +X side of the conveyor
    spawn zone at ~45 degrees, and saves one RGB PNG image per box
    shortly after it spawns.

    Parameters
    ----------
    output_dir:
        Directory where captured images are saved.
    spawn_position:
        (x, y, z) world position of the box spawn point (camera reference).
    camera_offset_z:
        Distance from *spawn_position* along +X for the camera.
    resolution:
        (width, height) in pixels.
    capture_delay_steps:
        Physics steps to wait after a new spawn before capturing.
    """

    def __init__(
        self,
        output_dir: str | Path | None,
        spawn_position: tuple[float, float, float] = (0.59, 3.01, -0.20),
        camera_offset_z: float = _DEFAULT_CAMERA_OFFSET_Z,
        resolution: tuple[int, int] = _DEFAULT_RESOLUTION,
        capture_delay_steps: int = _CAPTURE_DELAY_STEPS,
    ) -> None:
        self._output_dir: Path | None = Path(output_dir) if output_dir is not None else None
        self._spawn_position = spawn_position
        self._camera_offset_z = camera_offset_z
        self._resolution = resolution
        self._capture_delay_steps = capture_delay_steps

        self._last_box_count = 0
        self._countdown: int | None = None
        self._capture_count = 0
        self._annotator: object | None = None
        self._setup_done = False
        self._pending_metadata: dict[str, object] | None = None
        self._buffer: list[dict] = []

    def setup(self) -> None:
        """Create camera, render product, and RGB annotator.

        Must be called after ``World.reset()`` so the USD stage is loaded.
        Called lazily on first capture to avoid creating a second render
        product that conflicts with the streaming camera's overscan state.
        """
        if self._setup_done:
            return
        self._setup_done = True

        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()

        cam_path = "/World/BoxCaptureCam"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.GetFocalLengthAttr().Set(90.0)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
        sx, sy, sz = self._spawn_position
        d = self._camera_offset_z
        xf.AddTranslateOp().Set(Gf.Vec3d(sx + d, sy, sz + d * 0.62))
        # Rz(90) aligns image-up with world +Z; Rx(60) gives 30-deg downward tilt.
        xf.AddRotateXYZOp().Set(Gf.Vec3f(60, 0, 90))

        rp = rep.create.render_product(cam_path, self._resolution)
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotator.attach([rp])

        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._capture_count = self._next_index(self._output_dir)
            start = self._capture_count
            print(f"[BoxImageCapture] Saving to {self._output_dir} (start={start})")
        else:
            print("[BoxImageCapture] Buffer-only mode (no disk output)")

    def step(
        self,
        box_count: int,
        box_metadata: list[dict[str, object]] | None = None,
    ) -> None:
        """Advance capture state; call once per simulation step.

        Parameters
        ----------
        box_count:
            Current total number of spawned boxes (from ``BoxSpawner``).
        box_metadata:
            Per-box metadata list from ``BoxSpawner.box_metadata``.  The
            latest entry is saved when a new spawn is detected and written
            to ``metadata.json`` alongside the captured image.
        """
        if not self._setup_done:
            self.setup()
        if self._annotator is None:
            return

        # Detect new box spawn
        if box_count > self._last_box_count:
            self._countdown = self._capture_delay_steps
            self._last_box_count = box_count
            if box_metadata:
                self._pending_metadata = box_metadata[-1]

        # Countdown then capture
        if self._countdown is not None:
            if self._countdown <= 0:
                self._capture()
                self._countdown = None
            else:
                self._countdown -= 1

    @property
    def capture_count(self) -> int:
        """Number of images captured so far."""
        return self._capture_count

    def drain(self) -> list[dict]:
        """Return and clear the buffer."""
        result = list(self._buffer)
        self._buffer.clear()
        return result

    @staticmethod
    def _next_index(directory: Path) -> int:
        """Return the next available index by scanning existing ``box_*.png`` files."""
        max_idx = -1
        for p in directory.glob("box_*.png"):
            stem = p.stem  # e.g. "box_0042"
            parts = stem.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
        return max_idx + 1

    def _capture(self) -> None:
        from PIL import Image

        try:
            data = self._annotator.get_data()
        except Exception:
            # Replicator can raise during early render initialisation when
            # viewport overscan parameters are not yet populated.
            return
        if data is None:
            return
        # Some Isaac Sim versions wrap the array in a dict
        if isinstance(data, dict):
            data = data["data"]
        if data.size == 0 or data.ndim < 3:
            return

        # Annotator returns RGBA uint8 — drop alpha channel
        img = Image.fromarray(data[:, :, :3])

        # Encode to PNG bytes for the in-memory buffer
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        entry: dict[str, object] = {
            "box_id": f"box_{self._capture_count:04d}",
            "image_b64": base64.b64encode(png_bytes).decode(),
        }
        if self._pending_metadata is not None:
            for key in ("size", "weight", "type", "visual"):
                if key in self._pending_metadata:
                    entry[key] = self._pending_metadata[key]
        self._buffer.append(entry)

        # Write to disk only when an output directory is configured
        if self._output_dir is not None:
            image_name = f"box_{self._capture_count:04d}.png"
            path = self._output_dir / image_name
            img.save(str(path))

            if self._pending_metadata is not None:
                self._append_metadata({"image": image_name, **self._pending_metadata})

        self._pending_metadata = None
        self._capture_count += 1
        print(f"[BoxImageCapture] Captured box_{self._capture_count - 1:04d}")

    def _append_metadata(self, entry: dict[str, object]) -> None:
        """Append *entry* to ``metadata.json`` (read-append-write for crash safety)."""
        if self._output_dir is None:
            return
        meta_path = self._output_dir / "metadata.json"
        entries: list[dict[str, object]] = []
        if meta_path.exists():
            entries = json.loads(meta_path.read_text())
        entries.append(entry)
        meta_path.write_text(json.dumps(entries, indent=2) + "\n")
