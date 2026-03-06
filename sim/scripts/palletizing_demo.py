"""P3020 palletizing demo: conveyor -> pick & place -> pallet stack.

Industrial-style conveyor (legs, rails, rollers) at z=1.10m keeps pick
poses near the verified IK range [0.8, 0.2, 1.4] m.

    uv run python sim/scripts/palletizing_demo.py [--headless] [--num-boxes N]
"""

from __future__ import annotations

import argparse
import time

from drp_sim.robot import P3020Robot

# Bootstrap must come first -- all omni.* imports follow below.
from isaacsim import SimulationApp

# --- Constants ---

LOG_FILE = "/tmp/palletizing_demo.log"

CONVEYOR_START_X = 3.0
CONVEYOR_END_X = 0.10
PICKUP_X = 0.35
PICKUP_Y = -0.80
CONVEYOR_HEIGHT = 1.10
CONVEYOR_WIDTH = 0.55
CONVEYOR_SPEED = 0.20  # m/s, -X direction

BOX_SIZE = (0.35, 0.25, 0.20)  # L x W x H
BOX_HALF = (BOX_SIZE[0] / 2, BOX_SIZE[1] / 2, BOX_SIZE[2] / 2)
BOX_SPAWN_Z = CONVEYOR_HEIGHT + BOX_HALF[2]
BOX_MASS_KG = 1.5

# Pallet origins chosen to avoid payload-MotionGen IK failures.
# Both positions require small joint_1 rotation (<50 deg) from the lift
# pose, and lie outside the conveyor obstacle Y-range (y > -0.45).
PALLET_1_ORIGIN = (0.60, -0.40, 0.70)
PALLET_2_ORIGIN = (0.60, 0.10, 0.70)
PALLET_COLS = 2
PALLET_ROWS = 1
PALLET_LAYERS = 2

VGC10_LENGTH = 0.23
BOX_ATTACH_Z_OFFSET = VGC10_LENGTH + BOX_HALF[2]
ABOVE_OFFSET_Z = VGC10_LENGTH + 0.05
ABOVE_PLACE_Z = BOX_ATTACH_Z_OFFSET + 0.07
LIFT_OFFSET_Z = 0.15

MAX_CYCLES = 60

# --- Logging ---

_log_entries: list[str] = []


def _log(msg: str) -> None:
    _log_entries.append(msg)
    with open(LOG_FILE, "a") as fh:
        fh.write(msg + "\n")
        fh.flush()


def log_info(msg: str) -> None:
    line = f"[INFO]  {msg}"
    print(line, flush=True)
    _log(line)


def log_ok(msg: str) -> None:
    line = f"[OK]    {msg}"
    print(line, flush=True)
    _log(line)


def log_err(msg: str) -> None:
    line = f"[ERROR] {msg}"
    print(line, flush=True)
    _log(line)


def log_warn(msg: str) -> None:
    line = f"[WARN]  {msg}"
    print(line, flush=True)
    _log(line)


# --- Materials ---


def _apply_material(
    stage, prim_path: str, rgb: tuple[float, float, float], roughness: float = 0.6
) -> None:
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, f"{prim_path}_mat")
    shader = UsdShade.Shader.Define(stage, f"{prim_path}_mat/s")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    prim = stage.GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(mat)


_STEEL_GRAY = (0.35, 0.35, 0.38)
_RUBBER_BLACK = (0.12, 0.12, 0.12)
_ORANGE = (0.85, 0.40, 0.05)

_VGC10_DARK = (0.15, 0.15, 0.18)
_VGC10_RUBBER = (0.10, 0.10, 0.10)
_VGC10_ORANGE = (0.85, 0.35, 0.05)


# --- OnRobot VGC10 vacuum gripper visual ---


def _build_vgc10_visual(stage, root_path: str) -> None:
    """Build OnRobot VGC10 visual geometry under root_path (child of link_6 xform)."""
    from pxr import Gf, UsdGeom

    def _cube(name: str, tx: float, ty: float, tz: float, sx: float, sy: float, sz: float) -> str:
        p = f"{root_path}/{name}"
        c = UsdGeom.Cube.Define(stage, p)
        c.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(c.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
        xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        return p

    def _cyl(name: str, tx: float, ty: float, tz: float, r: float, h: float) -> str:
        p = f"{root_path}/{name}"
        c = UsdGeom.Cylinder.Define(stage, p)
        c.GetRadiusAttr().Set(r)
        c.GetHeightAttr().Set(h)
        UsdGeom.Xformable(c.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
        return p

    p = _cube("flange", 0, 0, -0.015, 0.116, 0.116, 0.030)
    _apply_material(stage, p, _VGC10_ORANGE, 0.5)
    p = _cube("body", 0, 0, -0.090, 0.110, 0.110, 0.120)
    _apply_material(stage, p, _VGC10_DARK, 0.4)
    p = _cube("cup_housing", 0, 0, -0.175, 0.095, 0.095, 0.050)
    _apply_material(stage, p, _VGC10_DARK, 0.5)

    cup_offsets = ((0.030, 0.030), (-0.030, 0.030), (0.030, -0.030), (-0.030, -0.030))
    for k, (cx, cy) in enumerate(cup_offsets):
        p = _cyl(f"cup_{k}", cx, cy, -0.220, 0.014, 0.020)
        _apply_material(stage, p, _VGC10_RUBBER, 0.95)

    log_ok(f"VGC10 visual built under {root_path}")


def attach_vgc10_to_robot(stage, robot_prim_path: str) -> str:
    from pxr import UsdGeom

    robot_base = robot_prim_path.rsplit("/", 1)[0]

    ee_prim = None
    for candidate in [f"{robot_base}/link_6"]:
        p = stage.GetPrimAtPath(candidate)
        if p.IsValid():
            ee_prim = p
            break
    if ee_prim is None:
        for p in stage.GetPrimAtPath(robot_base).GetAllDescendants():
            if p.GetName() == "link_6":
                ee_prim = p
                break

    if ee_prim is None:
        log_err("link_6 not found -- VGC10 not attached")
        return ""

    vgc10_path = str(ee_prim.GetPath()) + "/VGC10"
    UsdGeom.Xform.Define(stage, vgc10_path)
    _build_vgc10_visual(stage, vgc10_path)
    return vgc10_path


# --- Scene geometry ---


class ConveyorBelt:
    ROOT = "/World/Conveyor"

    def __init__(self, stage) -> None:
        from pxr import Gf, UsdGeom, UsdPhysics

        length = CONVEYOR_START_X - CONVEYOR_END_X
        cx = (CONVEYOR_START_X + CONVEYOR_END_X) / 2.0
        cy = PICKUP_Y
        leg_h = CONVEYOR_HEIGHT - 0.05
        leg_w = 0.06

        root = UsdGeom.Xform.Define(stage, self.ROOT)
        root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        surf_path = f"{self.ROOT}/surface"
        surf = UsdGeom.Cube.Define(stage, surf_path)
        surf.GetSizeAttr().Set(1.0)
        surf_xf = UsdGeom.Xformable(surf.GetPrim())
        surf_xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, CONVEYOR_HEIGHT - 0.025))
        surf_xf.AddScaleOp().Set(Gf.Vec3f(length, CONVEYOR_WIDTH, 0.05))
        UsdPhysics.CollisionAPI.Apply(surf.GetPrim())
        _apply_material(stage, surf_path, _RUBBER_BLACK, roughness=0.95)

        rail_thick = 0.04
        rail_h = 0.12
        for side, sy in (
            ("left", cy + CONVEYOR_WIDTH / 2 + rail_thick / 2),
            ("right", cy - CONVEYOR_WIDTH / 2 - rail_thick / 2),
        ):
            rp = f"{self.ROOT}/{side}_rail"
            r = UsdGeom.Cube.Define(stage, rp)
            r.GetSizeAttr().Set(1.0)
            rxf = UsdGeom.Xformable(r.GetPrim())
            rxf.AddTranslateOp().Set(Gf.Vec3d(cx, sy, CONVEYOR_HEIGHT - 0.025 + rail_h / 2))
            rxf.AddScaleOp().Set(Gf.Vec3f(length, rail_thick, rail_h))
            _apply_material(stage, rp, _STEEL_GRAY, roughness=0.4)

        leg_positions = [
            (CONVEYOR_END_X + 0.15, cy - 0.18),
            (CONVEYOR_END_X + 0.15, cy + 0.18),
            (CONVEYOR_START_X - 0.15, cy - 0.18),
            (CONVEYOR_START_X - 0.15, cy + 0.18),
        ]
        for i, (lx, ly) in enumerate(leg_positions):
            lp = f"{self.ROOT}/leg_{i}"
            leg = UsdGeom.Cube.Define(stage, lp)
            leg.GetSizeAttr().Set(1.0)
            lxf = UsdGeom.Xformable(leg.GetPrim())
            lxf.AddTranslateOp().Set(Gf.Vec3d(lx, ly, leg_h / 2))
            lxf.AddScaleOp().Set(Gf.Vec3f(leg_w, leg_w, leg_h))
            _apply_material(stage, lp, _STEEL_GRAY, roughness=0.4)

        for i, bx in enumerate([CONVEYOR_END_X + 0.15, CONVEYOR_START_X - 0.15]):
            bp = f"{self.ROOT}/brace_{i}"
            brace = UsdGeom.Cube.Define(stage, bp)
            brace.GetSizeAttr().Set(1.0)
            bxf = UsdGeom.Xformable(brace.GetPrim())
            bxf.AddTranslateOp().Set(Gf.Vec3d(bx, cy, leg_h * 0.45))
            bxf.AddScaleOp().Set(Gf.Vec3f(leg_w, CONVEYOR_WIDTH + 0.10, leg_w))
            _apply_material(stage, bp, _STEEL_GRAY, roughness=0.4)

        n_rollers = 12
        roller_spacing = length / (n_rollers + 1)
        UsdGeom.Xform.Define(stage, f"{self.ROOT}/rollers")
        for k in range(n_rollers):
            rx = CONVEYOR_END_X + roller_spacing * (k + 1)
            rpath = f"{self.ROOT}/rollers/r{k}"
            cyl = UsdGeom.Cylinder.Define(stage, rpath)
            cyl.GetRadiusAttr().Set(0.035)
            cyl.GetHeightAttr().Set(CONVEYOR_WIDTH + 0.08)
            cxf = UsdGeom.Xformable(cyl.GetPrim())
            cxf.AddTranslateOp().Set(Gf.Vec3d(rx, cy, CONVEYOR_HEIGHT - 0.025))
            cxf.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))
            _apply_material(stage, rpath, _STEEL_GRAY, roughness=0.3)

        ep = f"{self.ROOT}/end_stripe"
        end_b = UsdGeom.Cube.Define(stage, ep)
        end_b.GetSizeAttr().Set(1.0)
        exf = UsdGeom.Xformable(end_b.GetPrim())
        exf.AddTranslateOp().Set(Gf.Vec3d(CONVEYOR_END_X + 0.03, cy, CONVEYOR_HEIGHT - 0.02))
        exf.AddScaleOp().Set(Gf.Vec3f(0.06, CONVEYOR_WIDTH + 0.12, 0.08))
        _apply_material(stage, ep, _ORANGE, roughness=0.5)

        log_ok(f"ConveyorBelt built (length={length:.2f}m, surface z={CONVEYOR_HEIGHT}m)")


def _build_pallet_stand(stage, root: str, origin: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    px, py, pz = origin
    shelf_thick = 0.04
    shelf_w = PALLET_COLS * BOX_SIZE[0] + 0.10
    shelf_d = PALLET_ROWS * BOX_SIZE[1] + 0.10
    stand_h = pz - shelf_thick / 2

    UsdGeom.Xform.Define(stage, root)

    sp = f"{root}/shelf"
    shelf = UsdGeom.Cube.Define(stage, sp)
    shelf.GetSizeAttr().Set(1.0)
    sxf = UsdGeom.Xformable(shelf.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(px + shelf_w / 2 - BOX_HALF[0], py, pz - shelf_thick / 2))
    sxf.AddScaleOp().Set(Gf.Vec3f(shelf_w, shelf_d, shelf_thick))
    UsdPhysics.CollisionAPI.Apply(shelf.GetPrim())
    _apply_material(stage, sp, (0.55, 0.45, 0.30), roughness=0.7)

    for i, (lx, ly) in enumerate(
        [
            (px - 0.04, py - shelf_d / 2 + 0.05),
            (px + shelf_w - 0.04, py - shelf_d / 2 + 0.05),
            (px - 0.04, py + shelf_d / 2 - 0.05),
            (px + shelf_w - 0.04, py + shelf_d / 2 - 0.05),
        ]
    ):
        lp = f"{root}/leg_{i}"
        leg = UsdGeom.Cube.Define(stage, lp)
        leg.GetSizeAttr().Set(1.0)
        lxf = UsdGeom.Xformable(leg.GetPrim())
        lxf.AddTranslateOp().Set(Gf.Vec3d(lx, ly, stand_h / 2))
        lxf.AddScaleOp().Set(Gf.Vec3f(0.05, 0.05, stand_h))
        _apply_material(stage, lp, _STEEL_GRAY, roughness=0.4)

    log_ok(f"PalletStand {root} at ({px:.2f}, {py:.2f}), z={pz:.2f}m")


# --- Box and pallet logic ---


class BoxSpawner:
    def __init__(self, stage) -> None:
        self._stage = stage
        self._box_count = 0

    def spawn(self) -> str:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        idx = self._box_count
        self._box_count += 1
        path = f"/World/Box_{idx}"

        xform = UsdGeom.Xform.Define(self._stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(CONVEYOR_START_X, PICKUP_Y, BOX_SPAWN_Z))

        cube = UsdGeom.Cube.Define(self._stage, f"{path}/mesh")
        cube.GetSizeAttr().Set(1.0)
        UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
            Gf.Vec3f(BOX_SIZE[0], BOX_SIZE[1], BOX_SIZE[2])
        )

        mat = UsdShade.Material.Define(self._stage, f"{path}/mat")
        shader = UsdShade.Shader.Define(self._stage, f"{path}/mat/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.60, 0.40, 0.20)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        log_ok(f"Spawned box {idx} at x={CONVEYOR_START_X}")
        return path

    def advance(self, dt: float, box_path: str) -> None:
        from pxr import Gf, UsdGeom

        prim = self._stage.GetPrimAtPath(box_path)
        if not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        pos = xf.ComputeLocalToWorldTransform(0).ExtractTranslation()
        new_x = pos[0] - CONVEYOR_SPEED * dt
        for op in xf.GetOrderedXformOps():
            if "translate" in op.GetOpName():
                op.Set(Gf.Vec3d(new_x, pos[1], pos[2]))
                break

    def get_position(self, box_path: str) -> tuple[float, float, float]:
        from pxr import UsdGeom

        prim = self._stage.GetPrimAtPath(box_path)
        t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
        return (float(t[0]), float(t[1]), float(t[2]))

    def remove(self, box_path: str) -> None:
        if self._stage.GetPrimAtPath(box_path).IsValid():
            self._stage.RemovePrim(box_path)


class PalletStack:
    def __init__(self, origin: tuple[float, float, float], label: str = "Pallet") -> None:
        self._origin = origin
        self._label = label
        self._count = 0
        self._total = PALLET_ROWS * PALLET_COLS * PALLET_LAYERS

    @property
    def full(self) -> bool:
        return self._count >= self._total

    def next_position(self) -> tuple[float, float, float]:
        idx = self._count
        layer = idx // (PALLET_ROWS * PALLET_COLS)
        rem = idx % (PALLET_ROWS * PALLET_COLS)
        row = rem // PALLET_COLS
        col = rem % PALLET_COLS
        x = self._origin[0] + col * BOX_SIZE[0]
        y = self._origin[1] + row * BOX_SIZE[1]
        z = self._origin[2] + layer * BOX_SIZE[2] + BOX_HALF[2]
        return (x, y, z)

    def confirm_placed(self) -> None:
        self._count += 1
        log_ok(f"{self._label}: {self._count}/{self._total} boxes placed")


class DualPalletStack:
    def __init__(self) -> None:
        self._stacks = [
            PalletStack(PALLET_1_ORIGIN, "Pallet1"),
            PalletStack(PALLET_2_ORIGIN, "Pallet2"),
        ]
        self._idx = 0

    @property
    def full(self) -> bool:
        return all(s.full for s in self._stacks)

    def next_position(self) -> tuple[float, float, float]:
        for _ in range(2):
            if not self._stacks[self._idx].full:
                return self._stacks[self._idx].next_position()
            self._idx = (self._idx + 1) % 2
        raise RuntimeError("Both pallets are full")

    def confirm_placed(self) -> None:
        self._stacks[self._idx].confirm_placed()
        self._idx = (self._idx + 1) % 2


# --- Scene helpers ---


def get_ee_world_pos(stage, robot_prim_path: str) -> tuple[float, float, float]:
    from pxr import UsdGeom

    robot_base = robot_prim_path.rsplit("/", 1)[0]
    for candidate in [f"{robot_base}/link_6", "/p3020/link_6"]:
        prim = stage.GetPrimAtPath(candidate)
        if prim.IsValid():
            t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
            return (float(t[0]), float(t[1]), float(t[2]))

    root_prim = stage.GetPrimAtPath(robot_base)
    if root_prim.IsValid():
        for prim in root_prim.GetAllDescendants():
            if prim.GetName() == "link_6":
                t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
                return (float(t[0]), float(t[1]), float(t[2]))

    raise RuntimeError("Could not locate link_6 EE prim")


def teleport_box(stage, box_path: str, position: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(box_path)
    if not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if "translate" in op.GetOpName():
            op.Set(Gf.Vec3d(*position))
            return
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))


def _make_world_obstacles() -> list:
    """Build cuRobo collision cuboids for the scene.

    Only the conveyor belt is registered as a collision object.  Pallet stand
    obstacles were removed because:
    - The *_body* cuboids (floor-to-shelf-top) blocked IK/planning for
      above-pallet approach positions, causing ``plan failed`` failures.
    - The *_shelf* cuboids triggered false self-collision with the payload
      spheres (box extends below link_6 and touches the shelf height).
    - The stand legs are 50 x 50 mm; the arm clears them naturally on
      cuRobo-planned paths without explicit collision geometry.
    Previous trembling was caused by drive instability (underdamped joint_1
    with d=150), not by the arm hitting the stand -- now fixed by teleporting
    joints directly via set_joint_positions().
    """
    from curobo.geom.types import Cuboid

    conv_h = CONVEYOR_HEIGHT + 0.15
    conv_cx = (CONVEYOR_START_X + CONVEYOR_END_X) / 2.0
    return [
        Cuboid(
            name="conveyor",
            pose=[conv_cx, PICKUP_Y, conv_h / 2, 1.0, 0.0, 0.0, 0.0],
            dims=[CONVEYOR_START_X - CONVEYOR_END_X + 0.10, CONVEYOR_WIDTH + 0.15, conv_h],
        ),
    ]


# --- Main palletizing loop ---


def palletizing_loop(
    app: SimulationApp,
    robot: P3020Robot,
    num_boxes: int,
    *,
    render: bool = True,
) -> None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    spawner = BoxSpawner(stage)
    pallet = DualPalletStack()

    placed = 0
    total_cycles = 0
    dt = 1.0 / 60.0

    while placed < num_boxes and not pallet.full and total_cycles < MAX_CYCLES:
        total_cycles += 1
        log_info(f"--- Cycle {total_cycles} | placed {placed}/{num_boxes} ---")

        box_path = spawner.spawn()

        # Run conveyor until box reaches pickup zone
        max_steps = int((CONVEYOR_START_X - PICKUP_X + 0.8) / CONVEYOR_SPEED / dt) + 60
        reached = False
        for _ in range(max_steps):
            bx, _by, _bz = spawner.get_position(box_path)
            if bx < PICKUP_X:
                reached = True
                break
            spawner.advance(dt, box_path)
            robot.step(render=render)

        if not reached:
            log_err("Box did not reach pickup -- skipping")
            spawner.remove(box_path)
            continue

        bx, by, _bz = spawner.get_position(box_path)
        box_top_z = CONVEYOR_HEIGHT + BOX_SIZE[2]
        log_ok(f"Box at pickup: x={bx:.3f}, box_top_z={box_top_z:.3f}")

        # Approach above box
        above_pos = [bx, by, box_top_z + ABOVE_OFFSET_Z]
        log_info(f"Approach above box: {above_pos}")
        if not robot.move_to_pose(above_pos, render=render):
            log_err("Could not approach above box -- skipping")
            spawner.remove(box_path)
            continue

        robot.step(20, render=render)

        # Teleport-attach: snap box under VGC10 cup face
        ee = get_ee_world_pos(stage, robot.prim_path)
        teleport_box(stage, box_path, (ee[0], ee[1], ee[2] - BOX_ATTACH_Z_OFFSET))
        log_ok(f"Box attached at EE {[f'{v:.3f}' for v in ee]}")

        # Track box position each physics step while it is attached to the EE.
        # Bind box_path explicitly to avoid B023 (loop variable capture).
        def _track_box(_bp: str = box_path) -> None:
            ep = get_ee_world_pos(stage, robot.prim_path)
            teleport_box(stage, _bp, (ep[0], ep[1], ep[2] - BOX_ATTACH_Z_OFFSET))

        # Lift first with base MotionGen: start EE is still near the conveyor so
        # the box payload spheres would overlap the conveyor obstacle.  A straight
        # vertical lift clears this without needing payload-aware planning.
        lift_pos = [above_pos[0], above_pos[1], above_pos[2] + LIFT_OFFSET_Z]
        log_info("Lifting...")
        if not robot.move_to_pose(lift_pos, render=render, step_callback=_track_box):
            log_err("Lift failed -- dropping")
            spawner.remove(box_path)
            continue

        # Move to above pallet slot (base MotionGen -- pallet origins chosen to
        # avoid payload-MotionGen IK failures caused by the conveyor obstacle
        # boundary coinciding with the lift start pose of payload spheres).
        place_pos = pallet.next_position()
        above_place = [place_pos[0], place_pos[1], place_pos[2] + ABOVE_PLACE_Z]
        log_info(f"Above pallet slot: {above_place}")
        if not robot.move_to_pose(above_place, render=render, step_callback=_track_box):
            log_err("Could not reach above pallet -- dropping")
            spawner.remove(box_path)
            continue

        robot.step(10, render=render)

        # Descend from above_place to the exact placement height so the box
        # lands on the shelf without a visible jump.
        # EE target: box centre at place_pos.z  →  EE_z = place_pos.z + BOX_ATTACH_Z_OFFSET
        place_ee_z = place_pos[2] + BOX_ATTACH_Z_OFFSET
        descend_pos = [place_pos[0], place_pos[1], place_ee_z]
        log_info(f"Descending to place: EE z={place_ee_z:.3f}")
        if not robot.move_to_pose(
            descend_pos, max_attempts=5, render=render, step_callback=_track_box
        ):
            log_warn("Descent IK failed -- box will snap from above")

        robot.step(5, render=render)

        teleport_box(stage, box_path, place_pos)
        log_ok(f"Box placed at {[f'{v:.3f}' for v in place_pos]}")
        pallet.confirm_placed()
        placed += 1

        log_info("Returning home...")
        robot.go_home(render=render)

        if not app.is_running():
            break

    log_ok(f"Done: {placed} boxes placed in {total_cycles} cycles")


# --- CLI + main ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P3020 palletizing demo")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--num-boxes", type=int, default=6)
    p.add_argument(
        "--no-curobo", action="store_true", help="Conveyor-only test (no motion planning)"
    )
    return p.parse_args()


def main() -> None:
    with open(LOG_FILE, "w") as fh:
        fh.write(f"palletizing_demo started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    args = parse_args()
    app = SimulationApp({"headless": args.headless})

    import omni.usd
    from isaacsim.core.api import World

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    ConveyorBelt(stage)
    _build_pallet_stand(stage, "/World/PalletStand1", PALLET_1_ORIGIN)
    _build_pallet_stand(stage, "/World/PalletStand2", PALLET_2_ORIGIN)

    obstacles = [] if args.no_curobo else _make_world_obstacles()
    robot = P3020Robot(world, ghost=True, curobo=not args.no_curobo, world_obstacles=obstacles)
    robot.setup()
    log_ok(f"World ready, num_dof={robot.robot.num_dof}")

    render = not args.headless
    attach_vgc10_to_robot(stage, robot.prim_path)
    robot.step(1, render=render)  # flush renderer so VGC10 geometry is visible

    if args.no_curobo:
        log_info("--no-curobo: running conveyor-only visual test")
        spawner = BoxSpawner(stage)
        box = spawner.spawn()
        for _ in range(800):
            spawner.advance(1.0 / 60.0, box)
            robot.step(render=render)
        if not args.headless:
            while app.is_running():
                robot.step(render=True)
        app.close()
        return

    palletizing_loop(app, robot, args.num_boxes, render=render)

    if not args.headless:
        log_info("GUI mode -- close window to exit")
        while app.is_running():
            robot.step(render=True)

    app.close()


if __name__ == "__main__":
    main()
