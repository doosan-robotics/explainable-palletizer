"""Continuous P3020 pick-and-place loop with cuRobo.

Extends pick_place_once.py to place N boxes across a 2x2 pallet layout.
Uses the same verified coordinate system (PICKUP_X=0.35, PICKUP_Y=-0.80).

    uv run drp-pick [--num-boxes N] [--headless]
"""

from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Motion constants  (identical to pick_place_once.py -- all verified PASS)
# ---------------------------------------------------------------------------

PICKUP_X: float = 0.35
PICKUP_Y: float = -0.80
CONVEYOR_H: float = 1.10
BOX_H: float = 0.20
BOX_W: float = 0.35
BOX_D: float = 0.25
VGC10_LEN: float = 0.23
BOX_ATTACH_Z: float = VGC10_LEN + BOX_H / 2  # 0.33 m: link_6-to-box-centre offset

ABOVE_BOX: list[float] = [PICKUP_X, PICKUP_Y, CONVEYOR_H + BOX_H + VGC10_LEN + 0.05]
LIFT: list[float] = [PICKUP_X, PICKUP_Y, ABOVE_BOX[2] + 0.15]

PALLET_Z: float = 0.70  # shelf surface
SLOT_Z: float = PALLET_Z + BOX_H / 2  # box centre z when placed (0.80)

# 2 cols x 2 rows = 4 slots.  Column spacing = 0.40 m > BOX_W = 0.35 m (no overlap).
# Row spacing = 0.30 m > BOX_D = 0.25 m.  All slots outside conveyor obstacle (y > -0.45).
PALLET_SLOTS: list[tuple[float, float]] = [
    (0.60, -0.20),  # col 0, row 0
    (1.00, -0.20),  # col 1, row 0
    (0.60, 0.10),  # col 0, row 1
    (1.00, 0.10),  # col 1, row 1
]


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _say(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def _above_slot(x: float, y: float) -> list[float]:
    """EE target 0.07 m above the place pose."""
    return [x, y, SLOT_Z + BOX_ATTACH_Z + 0.07]  # z ≈ 1.20


def _place_ee(x: float, y: float) -> list[float]:
    """EE target at placement height."""
    return [x, y, SLOT_Z + BOX_ATTACH_Z]  # z ≈ 1.13


def _build_scene(stage) -> None:
    """Add conveyor surface and pallet shelf (collision geometry)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    c = UsdGeom.Cube.Define(stage, "/World/Conv/surf")
    c.GetSizeAttr().Set(1.0)
    cx = UsdGeom.Xformable(c.GetPrim())
    cx.AddTranslateOp().Set(Gf.Vec3d(1.55, PICKUP_Y, CONVEYOR_H - 0.025))
    cx.AddScaleOp().Set(Gf.Vec3f(2.90, 0.55, 0.05))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())

    # Pallet shelf — wide enough to cover all 4 slots (x=[0.40,1.20], y=[-0.35,0.25])
    s = UsdGeom.Cube.Define(stage, "/World/Pallet/shelf")
    s.GetSizeAttr().Set(1.0)
    sx = UsdGeom.Xformable(s.GetPrim())
    sx.AddTranslateOp().Set(Gf.Vec3d(0.80, -0.05, PALLET_Z - 0.02))
    sx.AddScaleOp().Set(Gf.Vec3f(0.80, 0.60, 0.04))
    UsdPhysics.CollisionAPI.Apply(s.GetPrim())


def _make_box(stage, idx: int) -> str:
    """Create a non-physics cardboard box at the pickup position. Return prim path."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    path = f"/World/Box_{idx}"
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(PICKUP_X, PICKUP_Y, CONVEYOR_H + BOX_H / 2))

    cb = UsdGeom.Cube.Define(stage, f"{path}/mesh")
    cb.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(cb.GetPrim()).AddScaleOp().Set(Gf.Vec3f(BOX_W, BOX_D, BOX_H))

    mat = UsdShade.Material.Define(stage, f"{path}/mat")
    sh = UsdShade.Shader.Define(stage, f"{path}/mat/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.60, 0.40, 0.20))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.90)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cb.GetPrim()).Bind(mat)
    return path


def _get_link6_pos(stage, robot_prim: str) -> tuple[float, float, float]:
    from pxr import UsdGeom

    base = robot_prim.rsplit("/", 1)[0]
    # Try robot_prim/link_6 first (drp-sim: /p3020/root_joint/link_6),
    # then base/link_6 (standalone setup: /p3020/link_6).
    for cand in (f"{robot_prim}/link_6", f"{base}/link_6"):
        p = stage.GetPrimAtPath(cand)
        if p.IsValid():
            t = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0).ExtractTranslation()
            return float(t[0]), float(t[1]), float(t[2])
    raise RuntimeError("link_6 not found in stage")


def _snap(stage, path: str, pos: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    for op in xf.GetOrderedXformOps():
        if "translate" in op.GetOpName():
            op.Set(Gf.Vec3d(*pos))
            return


# ---------------------------------------------------------------------------
# Per-box pick-and-place
# ---------------------------------------------------------------------------


def _pick_and_place(
    robot,
    stage,
    box_path: str,
    slot_xy: tuple[float, float],
    render: bool,
) -> bool:
    slot_x, slot_y = slot_xy
    step_results: list[bool] = []

    # 1. Approach above box
    ok = robot.move_to_pose(ABOVE_BOX, render=render)
    step_results.append(ok)
    _say("OK" if ok else "ERR", f"above_box  z={ABOVE_BOX[2]:.3f}")

    # 2. Attach: teleport box to EE
    ee = _get_link6_pos(stage, robot.prim_path)
    _snap(stage, box_path, (ee[0], ee[1], ee[2] - BOX_ATTACH_Z))

    def _track(_b: str = box_path) -> None:
        ep = _get_link6_pos(stage, robot.prim_path)
        _snap(stage, _b, (ep[0], ep[1], ep[2] - BOX_ATTACH_Z))

    # 3. Lift (base MotionGen — no payload spheres)
    ok = robot.move_to_pose(LIFT, render=render, step_callback=_track)
    step_results.append(ok)
    _say("OK" if ok else "ERR", f"lift  z={LIFT[2]:.3f}")

    # 4. Carry to above pallet slot
    above = _above_slot(slot_x, slot_y)
    ok = robot.move_to_pose(above, render=render, step_callback=_track)
    step_results.append(ok)
    _say("OK" if ok else "ERR", f"above_slot  ({slot_x:.2f}, {slot_y:.2f})")
    robot.step(8, render=render)

    # 5. Descend to place height
    place = _place_ee(slot_x, slot_y)
    ok = robot.move_to_pose(place, render=render, step_callback=_track)
    step_results.append(ok)
    _say("OK" if ok else "ERR", f"place  ({slot_x:.2f}, {slot_y:.2f})")
    robot.step(5, render=render)

    # 6. Detach: snap box to final resting position
    _snap(stage, box_path, (slot_x, slot_y, SLOT_Z))
    return all(step_results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="drp-pick",
        description="P3020 + cuRobo pick-and-place loop (2x2 pallet, 4 boxes max).",
    )
    parser.add_argument(
        "--num-boxes",
        type=int,
        default=4,
        metavar="N",
        help="Number of boxes to place (default: 4, max: 4).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window.",
    )
    args = parser.parse_args()
    num_boxes = min(args.num_boxes, len(PALLET_SLOTS))
    render = not args.headless

    from isaacsim import SimulationApp

    _app = SimulationApp({"headless": args.headless})

    import omni.usd
    from curobo.geom.types import Cuboid
    from isaacsim.core.api import World

    from drp_sim.robot import P3020Robot

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    _build_scene(stage)

    obstacles = [
        Cuboid(
            name="conveyor",
            pose=[1.55, PICKUP_Y, 0.625, 1.0, 0.0, 0.0, 0.0],
            dims=[2.90, 0.70, 1.25],
        ),
    ]
    robot = P3020Robot(world, ghost=False, curobo=True, world_obstacles=obstacles)
    robot.setup()
    robot.step(20, render=render)

    _say("INFO", f"placing {num_boxes} box(es) on 2x2 pallet")

    results: dict[str, bool] = {}
    for idx in range(num_boxes):
        slot_xy = PALLET_SLOTS[idx]
        _say("BOX", f"{idx + 1}/{num_boxes}  slot=({slot_xy[0]:.2f}, {slot_xy[1]:.2f})")

        box_path = _make_box(stage, idx)
        robot.step(5, render=render)

        ok = _pick_and_place(robot, stage, box_path, slot_xy, render)
        results[f"box_{idx}"] = ok
        _say("PASS" if ok else "FAIL", f"box_{idx}")

        robot.go_home(render=render)
        robot.step(10, render=render)

    # Hold final state so GUI users can inspect the result
    robot.step(180, render=render)

    print("\n=== Pick-and-place loop ===")
    for name, passed in results.items():
        print(f"  {name:<10}: {'PASS' if passed else 'FAIL'}")
    overall = all(results.values())
    print(f"  {'overall':<10}: {'PASS' if overall else 'FAIL'}")

    _app.close()
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
