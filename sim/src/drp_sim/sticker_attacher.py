"""Sticker attacher for spawned box prims.

Attaches a textured sticker quad as a USD child prim of a box so the
sticker inherits the box transform and follows it automatically.

The sticker is placed on the +X side face of the box, centered and
scaled to fill the face as much as possible while preserving its
aspect ratio.  Geometry is pre-compensated by the parent box scale
so the sticker appears undistorted.

Sticker data is loaded from a ``metadata.json`` manifest that maps each
image to a box type, optional weight, and optional visual description.
"""

from __future__ import annotations

import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path

# Native half-extents of the default box mesh (before parent scale).
# SM_CardBoxD_05 mesh (scale 0.01, extent X[-19,19] Y[-12.5,12.5] Z[0,14.875]).
# BoxSpawner overrides _box_half at runtime per variant.
_DEFAULT_BOX_HALF: tuple[float, float, float] = (0.19, 0.125, 0.14875)

_STICKER_Z_OFFSET: float = 0.003
_MIN_FACE_RATIO: float = 1 / 3


# ------------------------------------------------------------------
# Public helpers for BoxPool embedded stickers
# ------------------------------------------------------------------


def build_sticker_material(stage: object, mat_path: str) -> object:
    """Create a blank UsdPreviewSurface material (no texture yet)."""
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)

    surf = UsdShade.Shader.Define(stage, f"{mat_path}/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")

    uv_rd = UsdShade.Shader.Define(stage, f"{mat_path}/UVReader")
    uv_rd.CreateIdAttr("UsdPrimvarReader_float2")
    uv_rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader.Define(stage, f"{mat_path}/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        uv_rd.ConnectableAPI(), "result"
    )
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    return mat


def create_sticker_mesh(stage: object, sticker_path: str, mat_path: str) -> None:
    """Create a default unit-sized sticker mesh quad, bound to *mat_path*."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    points = [
        Gf.Vec3f(0, -0.5, -0.5),
        Gf.Vec3f(0, 0.5, -0.5),
        Gf.Vec3f(0, 0.5, 0.5),
        Gf.Vec3f(0, -0.5, 0.5),
    ]
    mesh = UsdGeom.Mesh.Define(stage, sticker_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(1, 0, 0)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    pvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = pvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)


def update_sticker_geometry(
    stage: object,
    mesh_path: str,
    aspect: float,
    box_half: tuple[float, float, float],
    parent_scale: tuple[float, float, float],
) -> None:
    """Rewrite sticker mesh points to fit the box face."""
    from pxr import Gf, UsdGeom

    sx, sy, sz = parent_scale
    face_w = 2 * box_half[1] * sy
    face_h = box_half[2] * sz
    face_aspect = face_w / face_h if face_h > 0 else 1.0

    if aspect >= face_aspect:
        w = face_w
        h = face_w / aspect
    else:
        h = face_h
        w = h * aspect
    hw, hh = w / 2, h / 2

    points = [
        Gf.Vec3f(0, -hw / sy, -hh / sz),
        Gf.Vec3f(0, hw / sy, -hh / sz),
        Gf.Vec3f(0, hw / sy, hh / sz),
        Gf.Vec3f(0, -hw / sy, hh / sz),
    ]
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    mesh.GetPointsAttr().Set(points)

    x_local = box_half[0] + _STICKER_Z_OFFSET / sx
    z_local = box_half[2] / 2
    xf = UsdGeom.Xformable(mesh.GetPrim())
    for op in xf.GetOrderedXformOps():
        if "translate" in op.GetOpName():
            op.Set(Gf.Vec3d(x_local, 0.0, z_local))
            return
    xf.AddTranslateOp().Set(Gf.Vec3d(x_local, 0.0, z_local))


def update_sticker_texture(stage: object, mat_path: str, image_path: str) -> None:
    """Update the texture file on an existing sticker material."""
    from pxr import Sdf, UsdShade

    tex_path = f"{mat_path}/Texture"
    tex = UsdShade.Shader(stage.GetPrimAtPath(tex_path))
    if tex:
        tex.GetInput("file").Set(Sdf.AssetPath(image_path))


def _png_aspect_ratio(path: Path) -> float:
    """Read width/height from a PNG IHDR chunk (no PIL dependency)."""
    with open(path, "rb") as f:
        f.seek(16)
        w, h = struct.unpack(">II", f.read(8))
    return w / h


@dataclass(frozen=True, slots=True)
class _StickerEntry:
    """Internal catalog entry loaded from metadata.json."""

    image_path: str
    aspect_ratio: float
    box_type: str
    weight: float | None = None
    visual: str | None = None


@dataclass(frozen=True, slots=True)
class StickerInfo:
    """Result of attaching a sticker to a box."""

    sticker_path: str
    image_path: str | None = None
    weight: float | None = None
    visual: str | None = None


@dataclass(frozen=True, slots=True)
class StickerSelection:
    """Selected sticker data without USD prim creation.

    Used by the object pool path where the sticker mesh already exists
    and only the geometry/texture need updating.
    """

    image_path: str
    aspect_ratio: float
    weight: float | None = None
    visual: str | None = None


class StickerAttacher:
    """Attaches textured sticker prims as children of box prims.

    Sticker entries are loaded from a ``metadata.json`` file.  Each entry
    specifies an image path, box type, and optional weight/visual fields.
    The ``attach()`` method selects a random sticker matching the given
    box type.

    Parameters
    ----------
    metadata_path:
        Path to a JSON manifest file.  Each entry must have ``image``
        and ``type`` fields.  Optional: ``weight``, ``visual``.
    box_half_extents:
        Native local half-extents (x, y, z_top) of the box mesh.
    """

    def __init__(
        self,
        metadata_path: str | Path,
        box_half_extents: tuple[float, float, float] = _DEFAULT_BOX_HALF,
    ) -> None:
        self._box_half = box_half_extents
        ordered, by_type = self._load_metadata(Path(metadata_path))
        self._entries_ordered = ordered
        self._entries_by_type = by_type
        self._cursor: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def attach(
        self,
        stage: object,
        box_prim_path: str,
        parent_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        box_type: str = "normal",
    ) -> StickerInfo | None:
        """Attach a randomly chosen sticker for the given box type.

        Returns ``None`` if the type has no sticker entries.
        """
        if box_type not in self._entries_by_type:
            return None
        entry = random.choice(self._entries_by_type[box_type])
        return self._attach_entry(stage, box_prim_path, parent_scale, entry)

    def attach_next(
        self,
        stage: object,
        box_prim_path: str,
        parent_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> StickerInfo | None:
        """Attach the next sticker in metadata order.

        Advances the internal cursor.  Returns ``None`` when all entries
        have been consumed.
        """
        if self._cursor >= len(self._entries_ordered):
            return None
        entry = self._entries_ordered[self._cursor]
        self._cursor += 1
        return self._attach_entry(stage, box_prim_path, parent_scale, entry)

    def peek_type(self) -> str | None:
        """Return the box type of the next sequential entry without consuming it."""
        if self._cursor >= len(self._entries_ordered):
            return None
        return self._entries_ordered[self._cursor].box_type

    @property
    def done(self) -> bool:
        """True when all sequential entries have been consumed."""
        return self._cursor >= len(self._entries_ordered)

    def pick_random(self, box_type: str) -> StickerSelection | None:
        """Select a random sticker for the given box type without creating prims.

        Used by the pooled spawn path where the sticker mesh already exists.
        """
        if box_type not in self._entries_by_type:
            return None
        entry = random.choice(self._entries_by_type[box_type])
        return StickerSelection(
            image_path=entry.image_path,
            aspect_ratio=entry.aspect_ratio,
            weight=entry.weight,
            visual=entry.visual,
        )

    def pick_next(self) -> StickerSelection | None:
        """Select the next sequential sticker without creating prims.

        Advances the internal cursor. Returns ``None`` when exhausted.
        """
        if self._cursor >= len(self._entries_ordered):
            return None
        entry = self._entries_ordered[self._cursor]
        self._cursor += 1
        return StickerSelection(
            image_path=entry.image_path,
            aspect_ratio=entry.aspect_ratio,
            weight=entry.weight,
            visual=entry.visual,
        )

    def __len__(self) -> int:
        """Total number of sticker entries."""
        return len(self._entries_ordered)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attach_entry(
        self,
        stage: object,
        box_prim_path: str,
        parent_scale: tuple[float, float, float],
        entry: _StickerEntry,
    ) -> StickerInfo:
        sticker_path = self._create_sticker(
            stage, box_prim_path, entry.image_path, entry.aspect_ratio, parent_scale
        )
        return StickerInfo(
            sticker_path=sticker_path,
            image_path=entry.image_path,
            weight=entry.weight,
            visual=entry.visual,
        )

    @staticmethod
    def _load_metadata(
        metadata_path: Path,
    ) -> tuple[list[_StickerEntry], dict[str, list[_StickerEntry]]]:
        """Load sticker entries from a JSON manifest.

        Returns ``(ordered_list, by_type_dict)`` where *ordered_list*
        preserves the JSON array order and *by_type_dict* groups entries
        by their ``type`` field.
        """
        data = json.loads(metadata_path.read_text())
        base_dir = metadata_path.parent
        ordered: list[_StickerEntry] = []
        by_type: dict[str, list[_StickerEntry]] = {}
        for entry in data:
            img_path = base_dir / entry["image"]
            if not img_path.exists():
                continue
            se = _StickerEntry(
                image_path=str(img_path),
                aspect_ratio=_png_aspect_ratio(img_path),
                box_type=entry["type"],
                weight=entry.get("weight"),
                visual=entry.get("visual"),
            )
            ordered.append(se)
            by_type.setdefault(entry["type"], []).append(se)
        if not ordered:
            raise FileNotFoundError(f"No valid sticker entries in {metadata_path}")
        return ordered, by_type

    def _create_sticker(
        self,
        stage: object,
        box_prim_path: str,
        image: str,
        aspect: float,
        parent_scale: tuple[float, float, float],
    ) -> str:
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        sx, sy, sz = parent_scale
        sticker_path = f"{box_prim_path}/Sticker"
        mat_path = f"{box_prim_path}/Looks/StickerMat"

        mat = self._build_material(stage, mat_path, image)

        # +X face dimensions in world space
        face_w = 2 * self._box_half[1] * sy
        face_h = self._box_half[2] * sz
        face_aspect = face_w / face_h if face_h > 0 else 1.0

        # Fit sticker to fill face while preserving aspect ratio
        if aspect >= face_aspect:
            w = face_w
            h = face_w / aspect
        else:
            h = face_h
            w = h * aspect
        hw, hh = w / 2, h / 2

        # YZ-plane quad with +X normals, pre-compensated by parent scale
        points = [
            Gf.Vec3f(0, -hw / sy, -hh / sz),
            Gf.Vec3f(0, hw / sy, -hh / sz),
            Gf.Vec3f(0, hw / sy, hh / sz),
            Gf.Vec3f(0, -hw / sy, hh / sz),
        ]

        mesh = UsdGeom.Mesh.Define(stage, sticker_path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(1, 0, 0)] * 4)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        pvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = pvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)

        # Center on the +X face
        x_local = self._box_half[0] + _STICKER_Z_OFFSET / sx
        y_local = 0.0
        z_local = self._box_half[2] / 2

        xf = UsdGeom.Xformable(mesh.GetPrim())
        # Clear any existing xformOps (handles recycled pool boxes that
        # already have a translate op from a previous sticker).
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x_local, y_local, z_local))

        return sticker_path

    @staticmethod
    def _build_material(stage: object, mat_path: str, image: str) -> object:
        """UsdPreviewSurface with PrimvarReader -> UsdUVTexture -> diffuseColor."""
        from pxr import Sdf, UsdShade

        mat = UsdShade.Material.Define(stage, mat_path)

        surf = UsdShade.Shader.Define(stage, f"{mat_path}/Surface")
        surf.CreateIdAttr("UsdPreviewSurface")
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        surf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")

        uv_rd = UsdShade.Shader.Define(stage, f"{mat_path}/UVReader")
        uv_rd.CreateIdAttr("UsdPrimvarReader_float2")
        uv_rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        tex = UsdShade.Shader.Define(stage, f"{mat_path}/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(image))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv_rd.ConnectableAPI(), "result"
        )
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        return mat
