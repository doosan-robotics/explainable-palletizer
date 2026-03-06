"""CLI entry point for drp-sim.

Usage:
    uv run drp-sim [options]
    uv run drp-sim --load-robot [--headless] [--spawn-interval SECONDS]
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import queue
import sys
import threading


# ---------------------------------------------------------------------------
# Pre-load libnvrtc-builtins before any curobo or Isaac Sim import.
# torch's cu130 wheel installs nvrtc into site-packages/nvidia/cu13/lib/ but
# does not add that directory to LD_LIBRARY_PATH.  When Isaac Sim's CUDA 11.8
# runtime is loaded first, dlopen cannot find the cu13 builtins and curobo
# JIT compilation fails.  Pre-loading via ctypes pins the correct library into
# the process before either runtime claims the NVRTC slot.
# ---------------------------------------------------------------------------
def _preload_nvrtc() -> None:
    import re

    best: tuple[tuple[int, ...], str] | None = None
    for path_entry in sys.path:
        for hit in glob.glob(f"{path_entry}/nvidia/*/lib/libnvrtc-builtins.so.*"):
            m = re.search(r"libnvrtc-builtins\.so\.(\d+(?:\.\d+)*)", hit)
            if not m:
                continue
            ver = tuple(int(x) for x in m.group(1).split("."))
            if best is None or ver > best[0]:
                best = (ver, hit)
    if best is None:
        return
    import contextlib

    with contextlib.suppress(OSError):
        ctypes.CDLL(best[1], mode=ctypes.RTLD_GLOBAL)


_preload_nvrtc()

# ---------------------------------------------------------------------------
# Pick-and-place constants — mixed_palletizing_scene.usd coordinate system
# Robot base stays at world (0, 0, 0); cuRobo targets are world coordinates.
#
# NOTE: these constants are intentionally different from pick_loop.py which
# uses a standalone fresh scene (PICKUP_X=0.35, PICKUP_Y=-0.80, robot at
# world z=0).  mixed_palletizing_scene.usd has the conveyor at a different
# position (box arrives at x=0.59, y=0.30) and the pallet on the other side
# of the robot.  The two files serve different purposes and must not share
# constants.
# ---------------------------------------------------------------------------

# Conveyor / pickup zone (world space, from scene diagnostics)
_CONV_X: float = 0.59  # box x on conveyor
_PICKUP_Y: float = 0.30  # world y to wait for box (boxes travel from y=3 → y≈0)
_PICKUP_TOL: float = 0.12  # pickup zone tolerance (m)

# Robot / gripper geometry
_VGC10_LEN: float = 0.23

# After lifting, swing to this waypoint before moving to the pallet.
# Boxes arrive from +Y; rotating toward -Y first keeps the held box out of the
# conveyor path and prevents collision with the next incoming box.
_SWING_Z: float = 0.377  # safe transit height (above any box on conveyor)
_SWING_VIA: list[float] = [0.0, -0.40, _SWING_Z]


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _say(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def _get_link6_pos(stage, robot_prim: str) -> tuple[float, float, float]:
    from pxr import UsdGeom

    # robot_prim = "/p3020/root_joint" → link_6 at "/p3020/root_joint/link_6"
    base = robot_prim.rsplit("/", 1)[0]
    for cand in (f"{robot_prim}/link_6", f"{base}/link_6"):
        p = stage.GetPrimAtPath(cand)
        if p.IsValid():
            t = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0).ExtractTranslation()
            return float(t[0]), float(t[1]), float(t[2])
    raise RuntimeError("link_6 not found in stage")


def _snap_rigid(box_prim, pos: tuple[float, float, float]) -> None:
    """Teleport a RigidPrim to pos and zero all velocities."""
    import numpy as np

    box_prim.set_world_poses(positions=np.array([pos]))
    box_prim.set_linear_velocities(np.array([[0.0, 0.0, 0.0]]))
    box_prim.set_angular_velocities(np.array([[0.0, 0.0, 0.0]]))


def _find_pickup_box(spawner, picked: set[str]):
    """Return (path, prim) for the first un-picked box in the pickup zone."""
    if spawner is None:
        return None, None
    for path, box in zip(spawner.box_paths, spawner.boxes, strict=False):
        if path in picked:
            continue
        try:
            positions, _ = box.get_world_poses()
            x = float(positions[0, 0])
            y = float(positions[0, 1])
            if abs(x - _CONV_X) < _PICKUP_TOL and abs(y - _PICKUP_Y) < _PICKUP_TOL:
                return path, box
        except Exception:
            continue
    return None, None


def _pick_and_place(
    robot,
    stage,
    box_prim,
    slot_xy: tuple[float, float],
    render: bool,
    pick_pos: tuple[float, float, float] = (_CONV_X, _PICKUP_Y, -0.206),
    box_half_h: float = 0.103,
    slot_z: float = -0.251,
    pallet_high_z: float = 0.70,
    on_step=None,
) -> bool:
    """Kinematic pick-and-place using known positions only (no USD queries).

    Parameters
    ----------
    pick_pos:
        Known (x, y, z_center) of the box — from buffer slot or spawner.
    box_half_h:
        Half the box height — from spawner metadata.
    on_step:
        Optional callback invoked every physics step to pin remaining
        buffer boxes in place while the robot is moving.
    """
    slot_x, slot_y = slot_xy
    results: list[bool] = []

    px, py, box_center_z = pick_pos
    box_top_z = box_center_z + box_half_h
    box_attach_z = _VGC10_LEN + box_half_h

    above_z = box_top_z + _VGC10_LEN + 0.10
    grasp_z = box_top_z + _VGC10_LEN + 0.07
    lift_z = above_z + 0.15

    _snap_rigid(box_prim, pick_pos)

    approach = [px, py, lift_z]
    above = [px, py, above_z]
    grasp = [px, py, grasp_z]
    lift = [px, py, lift_z]

    def _pin() -> None:
        """Hold the box at the pickup position during pre-grasp approach."""
        _snap_rigid(box_prim, pick_pos)
        if on_step is not None:
            on_step()

    ok = robot.move_to_pose(approach, render=render, step_callback=_pin, pre_step_callback=_pin)
    results.append(ok)
    _say("OK" if ok else "ERR", f"pickup_approach z={approach[2]:.3f}")

    ok = robot.move_to_pose(above, render=render, step_callback=_pin, pre_step_callback=_pin)
    results.append(ok)
    _say("OK" if ok else "ERR", f"above_box z={above[2]:.3f}")

    # Descend to cup-on-box-top contact then snap (box already at correct position).
    ok = robot.move_to_pose(grasp, render=render, step_callback=_pin, pre_step_callback=_pin)
    results.append(ok)
    _say("OK" if ok else "ERR", f"grasp z={grasp[2]:.3f}")

    ee = _get_link6_pos(stage, robot.prim_path)
    _snap_rigid(box_prim, (ee[0], ee[1], ee[2] - box_attach_z))

    def _track() -> None:
        ep = _get_link6_pos(stage, robot.prim_path)
        _snap_rigid(box_prim, (ep[0], ep[1], ep[2] - box_attach_z))
        if on_step is not None:
            on_step()

    ok = robot.move_to_pose(lift, render=render, step_callback=_track, pre_step_callback=_track)
    results.append(ok)
    _say("OK" if ok else "ERR", f"lift z={lift[2]:.3f}")

    # Swing toward -Y before moving to the pallet so the arm never sweeps
    # back through the conveyor (+Y) while holding the box.
    ok = robot.move_to_pose(
        _SWING_VIA,
        render=render,
        step_callback=_track,
        pre_step_callback=_track,
    )
    results.append(ok)
    _say("OK" if ok else "ERR", f"swing_via y={_SWING_VIA[1]:.2f}")

    # Move to directly above the target slot at lift height first, then descend
    # vertically.  Going straight from SWING_VIA to a low above_slot position
    # causes cuRobo to plan a diagonal path that sweeps through already-placed
    # boxes and knocks them off the pallet.
    pallet_high = [slot_x, slot_y, pallet_high_z]
    ok = robot.move_to_pose(
        pallet_high,
        render=render,
        step_callback=_track,
        pre_step_callback=_track,
    )
    results.append(ok)
    _say("OK" if ok else "ERR", f"pallet_high ({slot_x:.2f},{slot_y:.2f}) z={pallet_high_z:.3f}")

    above = [slot_x, slot_y, slot_z + box_attach_z + 0.15]
    ok = robot.move_to_pose(above, render=render, step_callback=_track, pre_step_callback=_track)
    results.append(ok)
    _say("OK" if ok else "ERR", f"above_slot ({slot_x:.2f},{slot_y:.2f})")
    robot.step(8, render=render, step_callback=_track)

    place = [slot_x, slot_y, slot_z + box_attach_z]
    ok = robot.move_to_pose(place, render=render, step_callback=_track, pre_step_callback=_track)
    results.append(ok)
    _say("OK" if ok else "ERR", f"place ({slot_x:.2f},{slot_y:.2f})")
    robot.step(5, render=render, step_callback=_track)

    _snap_rigid(box_prim, (slot_x, slot_y, slot_z))
    return all(results)


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------


def _run_view_mode(args) -> None:
    """Passive viewing: load the pre-built scene and step."""
    from drp_sim import PalletizerEnv

    env = PalletizerEnv(
        headless=args.headless,
        load_robot=False,
        spawn_boxes=not args.no_spawn_boxes,
        spawn_interval=args.spawn_interval,
    )
    env.reset()
    try:
        while env._app.is_running():
            env.step(render=not args.headless)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def _run_pick_mode(args) -> None:
    """P3020 + cuRobo interactive mode with SimRunner-like command dispatch.

    The scene already contains /p3020/root_joint from mixed_palletizing_scene_physics.usd.
    Re-importing the URDF creates a second articulation that splits visual mesh from physics
    (visual stays at home, only physics moves).  Loading with load_robot=False and wrapping
    the existing articulation keeps the scene visual mesh correctly linked to physics.
    """
    # Pre-warm curobo CUDA JIT before Isaac Sim loads its own CUDA 11.8 libraries.
    # MotionGen class-body evaluation triggers normalize_quaternion which compiles CUDA
    # kernels via nvrtc.  If Isaac Sim initialises first, its nvrtc 11.8 is already
    # resident in the process and the cu130 nvrtc-builtins.so.13.0 cannot be opened.
    from curobo.geom.types import Cuboid
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig  # noqa: F401

    from drp_sim import PalletizerEnv
    from drp_sim._constants import _PROCESSED_URDF, _URDF_PATH, preprocess_urdf
    from drp_sim.robot import P3020Robot

    render = not args.headless
    obstacles = [
        Cuboid(
            name="conveyor_floor",
            pose=[0.59, _PICKUP_Y, -0.934, 1.0, 0.0, 0.0, 0.0],
            dims=[2.90, 0.70, 1.25],
        ),
    ]

    env = PalletizerEnv(
        headless=args.headless,
        load_robot=False,
        spawn_boxes=True,
        spawn_interval=args.spawn_interval,
    )
    env.reset()

    import omni.usd
    from isaacsim.core.api.robots import Robot

    preprocess_urdf(_URDF_PATH, _PROCESSED_URDF)
    prim_path = "/p3020/root_joint"
    robot_art = env._world.scene.add(Robot(prim_path=prim_path, name="p3020"))
    env._world.reset()

    robot = P3020Robot.from_existing(
        env._world,
        robot_art,
        prim_path,
        world_obstacles=obstacles,
    )

    from drp_sim.sim_runner import attach_vgc10_gripper

    stage = omni.usd.get_context().get_stage()
    attach_vgc10_gripper(prim_path)
    robot.step(20, render=render)
    robot._teleport_to_home()
    robot.step(10, render=render)
    env.fill_buffer()

    from drp_sim.pallet_state import PalletConfig, PalletManager, hide_boxes

    pallet_mgr = PalletManager(
        [
            PalletConfig(
                "pallet_3",
                [(-0.657, -0.471), (-0.397, -0.471), (-0.657, -0.211), (-0.397, -0.211)],
                slot_z=-0.251,
                high_z=0.70,
            ),
        ]
    )
    _say("INFO", f"pick zone x={_CONV_X:.2f} y={_PICKUP_Y:.2f}  pallets={pallet_mgr.pallet_count}")
    _say("INFO", "Interactive mode. Type 'help' for commands.")

    cmd_queue: queue.Queue[str] = queue.Queue()
    playing = False
    shutdown = False
    picked: set[str] = set()

    def _read_stdin() -> None:
        while True:
            try:
                line = input("sim> ").strip()
                if line:
                    cmd_queue.put(line)
            except EOFError:
                cmd_queue.put("quit")
                break

    threading.Thread(target=_read_stdin, daemon=True).start()

    try:
        while env._app.is_running() and not shutdown:
            while not cmd_queue.empty():
                try:
                    line = cmd_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    parts = line.split()
                    cmd = parts[0].lower()

                    if cmd in ("quit", "exit", "q"):
                        shutdown = True
                    elif cmd == "help":
                        _say(
                            "CMD",
                            "fill spawn step [N] play pause state pick"
                            " human [N] pallet [N] reset home quit",
                        )
                    elif cmd == "fill":
                        result = env.fill_buffer()
                        _say("FILL", json.dumps(result))
                    elif cmd == "spawn":
                        path = env.spawn_box()
                        _say("SPAWN", path)
                    elif cmd == "step":
                        n = int(parts[1]) if len(parts) > 1 else 1
                        for _ in range(n):
                            env.step(render=render)
                        _say("STEP", f"{n} steps")
                    elif cmd == "play":
                        playing = True
                        _say("PLAY", "running")
                    elif cmd == "pause":
                        playing = False
                        _say("PAUSE", "stopped")
                    elif cmd == "state":
                        _print_sim_state(env, pallet_mgr)
                    elif cmd.startswith("pick"):
                        pick_parts = line.split()
                        pick_idx = int(pick_parts[1]) if len(pick_parts) > 1 else 0
                        path, prim = None, None
                        pick_pos = None
                        box_half_h = 0.103
                        if env._buffer is not None and env._buffer.occupied_count > 0:
                            # Read slot info BEFORE popping (position + dimensions)
                            slot_obj = (
                                env._buffer._slots[pick_idx]
                                if pick_idx < env._buffer.slot_count
                                else None
                            )
                            if slot_obj is not None:
                                pick_pos = slot_obj.assigned_position
                                box_half_h = slot_obj.box_half_h
                            result = env._buffer.pop_box_at(pick_idx)
                            if result is not None:
                                prim, path = result
                        if prim is None:
                            path, prim = _find_pickup_box(env._spawner, picked)
                        if prim is None:
                            _say("PICK", "no box in buffer")
                        else:
                            xy = pallet_mgr.current_slot_xy()
                            kwargs = {
                                "slot_z": pallet_mgr.slot_z(),
                                "pallet_high_z": pallet_mgr.high_z(),
                            }
                            if pick_pos is not None:
                                kwargs["pick_pos"] = pick_pos
                                kwargs["box_half_h"] = box_half_h
                            slot_s = pallet_mgr.active_pallet.slot_idx
                            pin_fn = (
                                env._buffer._enforce_slot_positions
                                if env._buffer is not None
                                else None
                            )
                            kwargs["on_step"] = pin_fn
                            ok = _pick_and_place(robot, stage, prim, xy, render, **kwargs)
                            # Release kinematic AFTER pick-and-place so Fabric
                            # stays in sync with cuRobo during motion planning.
                            if env._buffer is not None:
                                env._buffer.release_box(prim)
                            pallet_mgr.place_box(path, prim)
                            picked.add(path)
                            n_slots = len(pallet_mgr.active_pallet.config.slots)
                            _say("PICK", f"slot {slot_s % n_slots} {'OK' if ok else 'FAIL'}")
                            robot.go_home(render=render, step_callback=pin_fn)
                            robot.step(10, render=render, step_callback=pin_fn)
                    elif cmd == "human":
                        idx = int(parts[1]) if len(parts) > 1 else 0
                        result = env.remove_buffer_box(idx)
                        if result is None:
                            _say("HUMAN", f"no box at slot {idx}")
                        else:
                            path, _ = result
                            picked.add(path)
                            _say("HUMAN", f"slot {idx} removed ({path})")
                            fill_result = env.fill_buffer()
                            _say("FILL", json.dumps(fill_result))
                    elif cmd == "home":
                        robot.go_home(render=render)
                        _say("HOME", "done")
                    elif cmd == "reset":
                        hidden = env.reset_boxes()
                        pallet_boxes = pallet_mgr.reset_all()
                        hidden += hide_boxes(pallet_boxes)
                        picked.clear()
                        robot.go_home(render=render)
                        _say("RESET", f"{hidden} boxes cleared, all state reset")
                        result = env.fill_buffer()
                        _say("FILL", json.dumps(result))
                    elif cmd == "pallet":
                        p_idx = int(parts[1]) - 1 if len(parts) > 1 else None
                        if p_idx is not None and not 0 <= p_idx < pallet_mgr.pallet_count:
                            _say("ERR", f"pallet index must be 1..{pallet_mgr.pallet_count}")
                        else:
                            boxes = pallet_mgr.reset_pallet(p_idx)
                            hidden = hide_boxes(boxes)
                            for pb in boxes:
                                picked.discard(pb.prim_path)
                            target = (p_idx if p_idx is not None else pallet_mgr.active_idx) + 1
                            _say("PALLET", f"pallet {target} reset: {hidden} boxes hidden")
                    else:
                        _say("ERR", f"unknown: {cmd}. Type 'help'.")
                except Exception as exc:
                    _say("ERR", f"{type(exc).__name__}: {exc}")

            if playing:
                env.step(render=render)
            else:
                env._world.step(render=render)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def _print_sim_state(env, pallet_mgr=None) -> None:
    """Print current simulation state to stdout."""
    try:
        joints = env.get_joint_positions()
        _say("STATE", f"joints=[{', '.join(f'{j:.3f}' for j in joints)}]")
    except RuntimeError:
        pass
    if env._buffer is not None:
        states = env._buffer.slot_states
        tag = ",".join(str(s) for s in states)
        _say("STATE", f"buffer=[{tag}] ({env._buffer.occupied_count}/{env._buffer.slot_count})")
    for i, box in enumerate(env.boxes):
        try:
            pos, _ = box.get_world_poses()
            _say(f"BOX{i}", f"({pos[0, 0]:.2f}, {pos[0, 1]:.2f}, {pos[0, 2]:.2f})")
        except Exception:
            pass
    if pallet_mgr is not None:
        for i in range(pallet_mgr.pallet_count):
            p = pallet_mgr._pallets[i]
            active = " *" if i == pallet_mgr.active_idx else ""
            msg = f"{p.config.name} slot={p.slot_idx} placed={p.placed_count}{active}"
            _say(f"PALLET{i + 1}", msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="drp-sim",
        description="Run the palletizer simulation.",
    )
    parser.add_argument(
        "--spawn-interval",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Time in seconds between box spawns (default: 10.0).",
    )
    parser.add_argument(
        "--no-spawn-boxes",
        action="store_true",
        default=False,
        help="Disable automatic box spawning.",
    )
    parser.add_argument(
        "--load-robot",
        action="store_true",
        default=False,
        help="Import the P3020 URDF and run cuRobo pick-and-place.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI window.",
    )
    parser.add_argument(
        "--generate-box-images",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help="Capture top-down RGB images of each spawned box and save to OUTPUT_DIR.",
    )
    parser.add_argument(
        "--box-num",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N boxes are spawned (and N images if --generate-box-images is set).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for reproducibility (default: 42). Use -1 to disable.",
    )
    parser.add_argument(
        "--type-weights",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "Box type sampling weights as a JSON string, e.g. "
            """'{"normal":0.4,"fragile":0.2,"heavy":0.2,"damaged":0.2}'. """
            "Defaults to equal mix of all four types."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to YAML box spawn config file.",
    )
    parser.add_argument(
        "--generate-pallet-pattern",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help="Generate top-down images of solved pallet patterns and save to OUTPUT_DIR.",
    )
    parser.add_argument(
        "--num-patterns",
        type=int,
        default=50,
        metavar="N",
        help="Number of pallet patterns to generate (default: 50).",
    )
    parser.add_argument(
        "--min-boxes",
        type=int,
        default=3,
        metavar="N",
        help="Minimum boxes per pallet pattern (default: 3).",
    )
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=12,
        metavar="N",
        help="Maximum boxes per pallet pattern (default: 12).",
    )
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="Pause between pallet patterns for visual inspection (default: 3.0). 0 to disable.",
    )

    args = parser.parse_args()

    type_weights: dict[str, float] | None = None
    if args.type_weights is not None:
        type_weights = json.loads(args.type_weights)

    if args.generate_pallet_pattern is not None:
        _run_pallet_pattern(args, type_weights)
    elif args.load_robot:
        _run_pick_mode(args)
    else:
        _run_env(args, type_weights)


def _run_pallet_pattern(args: argparse.Namespace, type_weights: dict[str, float] | None) -> None:
    """Generate pallet pattern images using the solver + Isaac Sim."""
    from pathlib import Path

    from drp_sim.box_spawn_config import load_box_spawn_config
    from drp_sim.env import PalletizerEnv
    from drp_sim.pallet_pattern_generator import PalletPatternGenerator
    from drp_sim.sticker_attacher import StickerAttacher

    seed = args.seed if args.seed >= 0 else None

    # Load box type configs
    configs, weights, spawn = load_box_spawn_config(args.config)
    if type_weights is not None:
        weights = type_weights

    # Create a minimal env just for the World and USD stage
    env = PalletizerEnv(
        headless=False,
        spawn_boxes=False,
        seed=seed,
        config_path=args.config,
    )
    env.reset()

    try:
        sticker_attacher: StickerAttacher | None = None
        if spawn.sticker_metadata:
            p = Path(spawn.sticker_metadata)
            if not p.is_absolute():
                textures_dir = Path(__file__).parent.parent.parent / "usd" / "assets" / "textures"
                p = textures_dir / spawn.sticker_metadata
            sticker_attacher = StickerAttacher(metadata_path=p)

        generator = PalletPatternGenerator(
            world=env._world,
            type_configs=configs,
            type_weights=weights,
            output_dir=args.generate_pallet_pattern,
            sticker_attacher=sticker_attacher,
            min_boxes=args.min_boxes,
            max_boxes=args.max_boxes,
            preview_seconds=args.preview_seconds,
        )
        generator.setup()
        generator.run(num_patterns=args.num_patterns, seed=seed)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[drp-sim] {type(e).__name__}: {e}")
    finally:
        env.close()


def _run_env(args: argparse.Namespace, type_weights: dict[str, float] | None) -> None:
    """Run the standard palletizer simulation loop."""
    from drp_sim import PalletizerEnv

    env = PalletizerEnv(
        headless=False,
        load_robot=False,
        spawn_boxes=not args.no_spawn_boxes,
        spawn_interval=args.spawn_interval,
        seed=args.seed if args.seed >= 0 else None,
        generate_box_images=args.generate_box_images,
        box_num=args.box_num,
        type_weights=type_weights,
        config_path=args.config,
    )
    env.reset()
    env.fill_buffer()

    try:
        while env._app.is_running() and not env.done:
            env.step(render=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[drp-sim] {type(e).__name__}: {e}")
    finally:
        env.close()
