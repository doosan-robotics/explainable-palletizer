"""Inspect native bounding box dimensions of a USD asset.

Usage:
    python3.10 sim/scripts/inspect_usd_bbox.py [path_to_usd]

Defaults to SM_CardBoxD_05.usd if no argument is provided.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pxr import Usd, UsdGeom


def inspect_bbox(usd_path: str) -> None:
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print(f"ERROR: Cannot open {usd_path}")
        return

    root = stage.GetDefaultPrim()
    if not root.IsValid():
        print("ERROR: No default prim found")
        return

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"File: {usd_path}")
    print(f"Default prim: {root.GetPath()}")
    print(f"Up axis: {UsdGeom.GetStageUpAxis(stage)}")
    print(f"Meters per unit: {mpu}")
    print()

    # Print xform ops on all Xformable prims (first 10)
    print("=== Xform ops (first 10 prims) ===")
    count = 0
    for prim in Usd.PrimRange(root):
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        if ops:
            print(f"  {prim.GetPath()}:")
            for op in ops:
                print(f"    {op.GetOpName()} = {op.Get()}")
            count += 1
            if count >= 10:
                break
    print()

    # Print mesh extents
    print("=== Mesh extents ===")
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            ext = mesh.GetExtentAttr().Get()
            if ext:
                print(f"  {prim.GetPath()}: extent = {ext}")
    print()

    # BBox in stage units
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = cache.ComputeWorldBound(root)
    rng = bbox.ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()

    nx = hi[0] - lo[0]
    ny = hi[1] - lo[1]
    nz = hi[2] - lo[2]
    print("BBox (stage units):")
    print(f"  min: ({lo[0]:.6f}, {lo[1]:.6f}, {lo[2]:.6f})")
    print(f"  max: ({hi[0]:.6f}, {hi[1]:.6f}, {hi[2]:.6f})")
    print(f"  extent: ({nx:.6f}, {ny:.6f}, {nz:.6f})")
    print()

    # Convert to metres
    mx, my, mz = nx * mpu, ny * mpu, nz * mpu
    print(f"Real-world size (metres): ({mx:.6f}, {my:.6f}, {mz:.6f})")
    print()

    # Show what scale is needed for common target sizes
    for target in [(0.5, 0.5, 0.5), (0.5, 0.5, 1.0), (0.3, 0.3, 0.4)]:
        sx = target[0] / mx if mx > 0 else 0
        sy = target[1] / my if my > 0 else 0
        sz = target[2] / mz if mz > 0 else 0
        print(f"  Target {target}m -> scale ({sx:.4f}, {sy:.4f}, {sz:.4f})")


if __name__ == "__main__":
    default = str(Path(__file__).parent.parent / "usd" / "SM_CardBoxD_05.usd")
    path = sys.argv[1] if len(sys.argv) > 1 else default
    inspect_bbox(path)
