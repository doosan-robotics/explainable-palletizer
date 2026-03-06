"""Fragile sticker domain randomization test.

Randomly places sticker_<N>.png textures on the top face of a box,
varying position and rotation each frame via Omni Replicator.

Usage:
    uv run python sim/scripts/test_fragile_sticker.py
    uv run python sim/scripts/test_fragile_sticker.py --headless --num-frames 20
    uv run python sim/scripts/test_fragile_sticker.py --output-dir /tmp/sticker_data
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Scene config — edit these constants to adapt to a different setup
# ---------------------------------------------------------------------------

# Box centre in world space (m).  Bottom sits on the ground plane (z=0)
# when center_z == box_half_height.
BOX_CENTER: tuple[float, float, float] = (0.0, 0.0, 0.25)

# Full extents of the box (m)
BOX_SIZE: tuple[float, float, float] = (0.5, 0.5, 0.5)

# Sticker full extents (m) — typical fragile label
STICKER_SIZE: tuple[float, float] = (0.08, 0.06)

# Small z-offset added to the box top face to prevent z-fighting
Z_EPSILON: float = 0.001

# Where to look for sticker_*.png images
_TEXTURE_DIR = Path(__file__).resolve().parent.parent / "usd" / "assets" / "textures"

# Prim path for the texture node (used for per-frame texture switching)
_TEX_PRIM_PATH = "/World/Looks/StickerMat/Texture"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_sticker_images() -> list[str]:
    """Return sorted absolute paths for all sticker_*.png images."""
    images = sorted(_TEXTURE_DIR.glob("sticker_*.png"))
    if not images:
        raise FileNotFoundError(
            f"No sticker_*.png images found in:\n  {_TEXTURE_DIR}\n"
            "Name your images sticker_0.png, sticker_1.png, ..."
        )
    return [str(p) for p in images]


def _build_material(stage: object) -> object:
    """Build UsdPreviewSurface with a switchable texture input.

    Shader graph:
        PrimvarReader("st") -> UsdUVTexture(file) -> UsdPreviewSurface

    The 'file' input is left empty here and updated per-frame via USD API.
    """
    from pxr import Sdf, UsdShade

    mat_path = "/World/Looks/StickerMat"
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

    tex = UsdShade.Shader.Define(stage, _TEX_PRIM_PATH)
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset)  # filled per-frame
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


def _build_sticker_mesh(stage: object, mat: object) -> str:
    """Create a textured quad mesh; return its prim path."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    prim_path = "/World/Sticker"
    mesh = UsdGeom.Mesh.Define(stage, prim_path)

    hw, hh = STICKER_SIZE[0] / 2, STICKER_SIZE[1] / 2
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-hw, -hh, 0),
            Gf.Vec3f(hw, -hh, 0),
            Gf.Vec3f(hw, hh, 0),
            Gf.Vec3f(-hw, hh, 0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    pvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = pvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)
    return prim_path


def _set_texture(stage: object, image_path: str) -> None:
    """Switch the active texture on the sticker material (per-frame call)."""
    from pxr import Sdf, UsdShade

    tex = UsdShade.Shader(stage.GetPrimAtPath(_TEX_PRIM_PATH))
    tex.GetInput("file").Set(Sdf.AssetPath(image_path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fragile sticker randomization test")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--num-frames", type=int, default=10, help="Randomized frames")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Save RGB renders here via Replicator BasicWriter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sticker_images = _find_sticker_images()
    print(f"Found {len(sticker_images)} sticker(s): {[Path(p).name for p in sticker_images]}")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    # Post-bootstrap imports (Kit runtime must be alive before any omni.* import)
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()

    # Scene
    GroundPlane(prim_path="/World/GroundPlane", z_position=0)
    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
    light.CreateIntensityAttr(500)

    # Box — center at BOX_CENTER, size BOX_SIZE
    bx, by, bz = BOX_CENTER
    DynamicCuboid(
        prim_path="/World/Box",
        name="box",
        position=np.array([bx, by, bz]),
        size=BOX_SIZE[0],  # DynamicCuboid takes full edge length (assumes cube)
        color=np.array([0.6, 0.4, 0.2]),
    )

    # Sticker mesh + material
    mat = _build_material(stage)
    sticker_prim_path = _build_sticker_mesh(stage, mat)

    # Placement bounds — sticker stays fully within the box top face
    sticker_z = bz + BOX_SIZE[2] / 2 + Z_EPSILON
    x_half = (BOX_SIZE[0] - STICKER_SIZE[0]) / 2  # max x offset from box centre
    y_half = (BOX_SIZE[1] - STICKER_SIZE[1]) / 2  # max y offset from box centre

    # Camera: top-down view centred on the box
    cam = UsdGeom.Camera.Define(stage, "/World/TopCam")
    cam_xform = UsdGeom.Xformable(cam.GetPrim())
    cam_xform.AddTranslateOp().Set(Gf.Vec3d(bx, by, bz + 1.0))
    cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))

    # Replicator graph: pose randomization only
    with rep.new_layer():
        sticker_rep = rep.get.prim_at_path(sticker_prim_path)

        def randomize_sticker():
            with sticker_rep:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (bx - x_half, by - y_half, sticker_z),
                        (bx + x_half, by + y_half, sticker_z),
                    ),
                    rotation=rep.distribution.uniform((0, 0, -180), (0, 0, 180)),
                )
            return sticker_rep.node

        rep.randomizer.register(randomize_sticker)

        with rep.trigger.on_frame(num_frames=args.num_frames):
            rep.randomizer.randomize_sticker()

    # Optional image output
    if args.output_dir:
        render_product = rep.create.render_product("/World/TopCam", (640, 480))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=args.output_dir, rgb=True)
        writer.attach(render_product)
        print(f"Saving renders to {args.output_dir}")

    # Simulation loop: texture is switched per-frame via USD, pose via Replicator
    print(f"Running {args.num_frames} randomized frames...")
    for frame in range(args.num_frames):
        chosen = random.choice(sticker_images)
        _set_texture(stage, chosen)
        rep.orchestrator.step()
        simulation_app.update()
        print(f"  [{frame + 1}/{args.num_frames}] {Path(chosen).name}")

    print("Done. Scene stays open (Ctrl+C or close window to exit).")
    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    main()
