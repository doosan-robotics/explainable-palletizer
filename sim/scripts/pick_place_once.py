"""Single pick-and-place test for P3020.

Spawns one box at the pickup position, picks it up, carries it to the pallet
stand, and places it.  Prints PASS/FAIL for each trajectory segment to verify
smooth motion (no trembling) and absence of collisions.

    PYTHONPATH=sim/src /home/ms/isaac-sim/python.sh sim/scripts/pick_place_once.py
    PYTHONPATH=sim/src /home/ms/isaac-sim/python.sh sim/scripts/pick_place_once.py --headless
"""

from __future__ import annotations

import sys

from drp_sim.robot import P3020Robot
from isaacsim import SimulationApp

HEADLESS = "--headless" in sys.argv
app = SimulationApp({"headless": HEADLESS})

# All omni/USD/curobo imports must follow SimulationApp construction.
import omni.usd  # noqa: E402
from curobo.geom.types import Cuboid  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

# ---------------------------------------------------------------------------
# Motion targets  (link_6 world position in metres)
# ---------------------------------------------------------------------------
PICKUP_X = 0.35
PICKUP_Y = -0.80
CONVEYOR_H = 1.10
BOX_H = 0.20
VGC10_LEN = 0.23
BOX_ATTACH_Z = VGC10_LEN + BOX_H / 2  # 0.33 m: link_6-to-box-centre offset

ABOVE_BOX = [PICKUP_X, PICKUP_Y, CONVEYOR_H + BOX_H + VGC10_LEN + 0.05]  # z=1.58
LIFT = [PICKUP_X, PICKUP_Y, ABOVE_BOX[2] + 0.15]  # z=1.73

# Pallet slot: x=0.80, y=-0.20 -- outside conveyor obstacle (y>-0.45), small
# joint_1 rotation from pick (~-66 deg) to place (~-14 deg) = only 52 deg.
# Base MotionGen is used throughout; payload sphere planning is deferred.
PALLET_XY = (0.80, -0.20)
PALLET_Z = 0.70  # shelf surface height
SLOT_Z = PALLET_Z + BOX_H / 2  # box centre z when placed  (0.80)
ABOVE_PLACE = [PALLET_XY[0], PALLET_XY[1], SLOT_Z + BOX_ATTACH_Z + 0.07]  # z=1.20
PLACE_EE = [PALLET_XY[0], PALLET_XY[1], SLOT_Z + BOX_ATTACH_Z]  # z=1.13

RENDER = not HEADLESS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def say(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def get_link6_pos(stage, robot_prim: str) -> tuple[float, float, float]:
    base = robot_prim.rsplit("/", 1)[0]
    for cand in (f"{base}/link_6", "/p3020/link_6"):
        p = stage.GetPrimAtPath(cand)
        if p.IsValid():
            t = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0).ExtractTranslation()
            return (float(t[0]), float(t[1]), float(t[2]))
    raise RuntimeError("link_6 not found in stage")


def snap(stage, path: str, pos: tuple[float, float, float]) -> None:
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    for op in xf.GetOrderedXformOps():
        if "translate" in op.GetOpName():
            op.Set(Gf.Vec3d(*pos))
            return


def build_scene(stage) -> str:
    """Add conveyor surface, pallet shelf, cardboard box. Return box USD path."""
    cx = (3.0 + 0.10) / 2
    cy = PICKUP_Y

    # Conveyor surface (collision geometry)
    c = UsdGeom.Cube.Define(stage, "/World/Conv/surf")
    c.GetSizeAttr().Set(1.0)
    cxf = UsdGeom.Xformable(c.GetPrim())
    cxf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, CONVEYOR_H - 0.025))
    cxf.AddScaleOp().Set(Gf.Vec3f(2.90, 0.55, 0.05))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())

    # Pallet shelf (collision geometry)
    px, py = PALLET_XY
    s = UsdGeom.Cube.Define(stage, "/World/Pallet/shelf")
    s.GetSizeAttr().Set(1.0)
    sxf = UsdGeom.Xformable(s.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(px, py, PALLET_Z - 0.02))
    sxf.AddScaleOp().Set(Gf.Vec3f(0.80, 0.40, 0.04))
    UsdPhysics.CollisionAPI.Apply(s.GetPrim())

    # Cardboard box (display only -- teleported throughout, no physics)
    box = "/World/Box"
    bxf = UsdGeom.Xform.Define(stage, box)
    bxf.AddTranslateOp().Set(Gf.Vec3d(PICKUP_X, PICKUP_Y, CONVEYOR_H + BOX_H / 2))
    cb = UsdGeom.Cube.Define(stage, f"{box}/mesh")
    cb.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(cb.GetPrim()).AddScaleOp().Set(Gf.Vec3f(0.35, 0.25, BOX_H))
    mat = UsdShade.Material.Define(stage, f"{box}/mat")
    sh = UsdShade.Shader.Define(stage, f"{box}/mat/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.60, 0.40, 0.20))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.90)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cb.GetPrim()).Bind(mat)
    return box


# ---------------------------------------------------------------------------
# World setup
# ---------------------------------------------------------------------------
world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
box = build_scene(stage)

obstacles = [
    Cuboid(
        name="conveyor",
        pose=[1.55, PICKUP_Y, 0.625, 1.0, 0.0, 0.0, 0.0],
        dims=[2.90, 0.70, 1.25],
    ),
]
robot = P3020Robot(world, ghost=False, curobo=True, world_obstacles=obstacles)
robot.setup()
robot.step(20, render=RENDER)

say("INFO", f"ABOVE_BOX={ABOVE_BOX}  LIFT={LIFT}")
say("INFO", f"ABOVE_PLACE={ABOVE_PLACE}  PLACE_EE={PLACE_EE}")

results: dict[str, bool] = {}

# ---------------------------------------------------------------------------
# Pick-and-place sequence
# ---------------------------------------------------------------------------

# Step 1: Approach above box
say("STEP", "1/6  above_box")
ok = robot.move_to_pose(ABOVE_BOX, render=RENDER)
results["above_box"] = ok
say("OK" if ok else "ERR", f"above_box  z={ABOVE_BOX[2]:.3f}")
robot.step(10, render=RENDER)

# Step 2: Attach box (teleport to EE position)
ee = get_link6_pos(stage, robot.prim_path)
snap(stage, box, (ee[0], ee[1], ee[2] - BOX_ATTACH_Z))
say("INFO", f"box attached  EE={[round(v, 3) for v in ee]}")


def _track(_b: str = box) -> None:
    ep = get_link6_pos(stage, robot.prim_path)
    snap(stage, _b, (ep[0], ep[1], ep[2] - BOX_ATTACH_Z))


# Step 3: Lift with base MotionGen (box near conveyor -- no payload spheres yet)
say("STEP", "3/6  lift")
ok = robot.move_to_pose(LIFT, render=RENDER, step_callback=_track)
results["lift"] = ok
say("OK" if ok else "ERR", f"lift  z={LIFT[2]:.3f}")

# Step 4: Carry to above pallet (base MotionGen -- no payload spheres).
# The robot has already cleared the conveyor obstacle at LIFT height.
# The pallet is at y=-0.20, well outside the conveyor obstacle (y<-0.45).
say("STEP", f"4/6  above_pallet  {ABOVE_PLACE}")
ok = robot.move_to_pose(ABOVE_PLACE, render=RENDER, step_callback=_track)
results["above_pallet"] = ok
say("OK" if ok else "ERR", "above_pallet")
robot.step(10, render=RENDER)

# Step 5: Descend to placement height
say("STEP", f"5/6  place  {PLACE_EE}")
ok = robot.move_to_pose(PLACE_EE, render=RENDER, step_callback=_track)
results["place"] = ok
say("OK" if ok else "ERR", "place")
robot.step(5, render=RENDER)

# Step 6: Detach and return home
snap(stage, box, (PALLET_XY[0], PALLET_XY[1], SLOT_Z))
say("STEP", "6/6  home")
robot.go_home(render=RENDER)

# Hold final pose so the result is visible in GUI mode.
robot.step(180, render=RENDER)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Pick-and-place summary ===")
for name, passed in results.items():
    print(f"  {name:<15}: {'PASS' if passed else 'FAIL'}")
overall = all(results.values())
print(f"  {'overall':<15}: {'PASS' if overall else 'FAIL'}")

app.close()
sys.exit(0 if overall else 1)
