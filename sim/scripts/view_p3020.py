"""Isaac Sim GUI viewer for P3020 robot.

* Solid robot    -- physics simulation, follows cuRobo trajectory
* Ghost robot    -- transparent blue overlay, snaps to commanded target each step
* Joint Monitor  -- floating panel: Cmd / Actual / Error per joint

Run:
    uv run python sim/scripts/view_p3020.py
    uv run python sim/scripts/view_p3020.py --no-curobo
    uv run python sim/scripts/view_p3020.py --loop
"""

from __future__ import annotations

import argparse
import math

from drp_sim.robot import JOINT_NAMES, P3020Robot
from isaacsim import SimulationApp

JOINT_LIMITS_DEG = [(-360, 360), (-125, 125), (-160, 160), (-360, 360), (-360, 360)]

POSES: list[tuple[str, list[float]]] = [
    ("zero / home", [0.000, 0.000, 0.000, 0.000, 0.000]),
    ("ready", [0.000, -0.524, 1.047, 0.000, 0.000]),
    ("reach forward", [0.000, -0.785, 1.309, 0.000, 0.000]),
    ("turn left 45°", [0.785, -0.524, 1.047, 0.000, 0.000]),
    ("turn right 45°", [-0.785, -0.524, 1.047, 0.000, 0.000]),
    ("wrist up (j5=90°)", [0.000, -0.524, 1.047, 1.571, 0.000]),
    ("wrist roll (j6=90°)", [0.000, -0.524, 1.047, 0.000, 1.571]),
    ("full extend", [0.000, -0.785, 1.571, 0.000, 0.000]),
    ("back to ready", [0.000, -0.524, 1.047, 0.000, 0.000]),
]

HOLD_STEPS = 120  # ~2 s at 60 Hz


# ── Joint Monitor UI ────────────────────────────────────────────────────────


class JointMonitorUI:
    """Floating omni.ui panel: Commanded / Actual / Error per joint."""

    _WHITE = 0xFFFFFFFF
    _CYAN = 0xFFFFFF00
    _YELLOW = 0xFF00FFFF
    _GREEN = 0xFF00FF00
    _ORANGE = 0xFF00AAFF
    _RED = 0xFF0000FF
    _GREY = 0xFFAAAAAA

    def __init__(self) -> None:
        import omni.ui as ui

        self._ui = ui
        self._window = ui.Window(
            "P3020 Joint Monitor",
            width=520,
            height=310,
            flags=ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_RESIZE,
        )
        self._rows: list[tuple] = []
        self._build()

    def _lbl(self, text: str, width: int, color: int = 0xFFFFFFFF, size: int = 13) -> None:
        self._ui.Label(text, width=width, style={"font_size": size, "color": color})

    def _build(self) -> None:
        ui = self._ui
        cw = [110, 95, 95, 85, 115]

        with self._window.frame, ui.VStack(spacing=4):
            with ui.HStack(height=22):
                self._pose_lbl = ui.Label("Pose: --", style={"font_size": 16, "color": self._WHITE})
            with ui.HStack(height=18):
                self._status_lbl = ui.Label(
                    "Status: initializing", style={"font_size": 13, "color": self._CYAN}
                )
            ui.Separator(height=2)

            headers = ["Joint", "Cmd (°)", "Actual (°)", "Error (°)", "Limit (°)"]
            with ui.HStack(height=18):
                for txt, w in zip(headers, cw, strict=True):
                    self._lbl(txt, w, color=self._GREY)
            ui.Separator(height=2)

            for i, jname in enumerate(JOINT_NAMES):
                lo, hi = JOINT_LIMITS_DEG[i]
                with ui.HStack(height=20):
                    self._lbl(jname, cw[0])
                    cmd_l = ui.Label(
                        "   0.00", width=cw[1], style={"font_size": 13, "color": self._YELLOW}
                    )
                    act_l = ui.Label(
                        "   0.00", width=cw[2], style={"font_size": 13, "color": self._WHITE}
                    )
                    err_l = ui.Label(
                        "  0.00", width=cw[3], style={"font_size": 13, "color": self._GREEN}
                    )
                    self._lbl(f"[{lo}, {hi}]", cw[4], color=self._GREY, size=12)
                self._rows.append((cmd_l, act_l, err_l))

            ui.Separator(height=2)
            with ui.HStack(height=18):
                self._rms_lbl = ui.Label(
                    "RMS error: 0.000°", style={"font_size": 13, "color": self._GREEN}
                )

    def update(
        self, pose_name: str, cmd: list[float], actual: list[float], status: str = "moving"
    ) -> None:
        self._pose_lbl.text = f"Pose: {pose_name}"
        self._status_lbl.text = f"Status: {status}"

        errs_sq: list[float] = []
        for i, (cmd_l, act_l, err_l) in enumerate(self._rows):
            c = math.degrees(cmd[i]) if i < len(cmd) else 0.0
            a = math.degrees(actual[i]) if i < len(actual) else 0.0
            e = a - c
            cmd_l.text = f"{c:+8.2f}"
            act_l.text = f"{a:+8.2f}"
            err_l.text = f"{e:+7.2f}"
            errs_sq.append(e**2)
            abs_e = abs(e)
            col = self._GREEN if abs_e < 1.0 else (self._ORANGE if abs_e < 3.0 else self._RED)
            err_l.set_style({"font_size": 13, "color": col})

        rms = math.sqrt(sum(errs_sq) / len(errs_sq)) if errs_sq else 0.0
        rms_col = self._GREEN if rms < 1.0 else (self._ORANGE if rms < 3.0 else self._RED)
        self._rms_lbl.text = f"RMS error: {rms:.3f}°"
        self._rms_lbl.set_style({"font_size": 13, "color": rms_col})


# ── main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--no-curobo", action="store_true")
    p.add_argument("--loop", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = SimulationApp({"headless": False})

    from isaacsim.core.api import World

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()

    robot = P3020Robot(world, ghost=True, curobo=not args.no_curobo)
    monitor = JointMonitorUI()

    monitor._status_lbl.text = "Status: loading robot..."
    world.step(render=True)

    if not args.no_curobo:
        monitor._status_lbl.text = "Status: building cuRobo (~20s)..."
        world.step(render=True)

    try:
        robot.setup()
    except Exception as exc:
        print(f"[view] setup failed: {exc}", flush=True)
        app.close()
        return

    def move_to(pose_name: str, goal: list[float]) -> None:
        print(f"[view] -> {pose_name}", flush=True)

        # Ghost jumps to goal right away so the user can see the target
        if robot.ghost:
            robot.ghost.update(goal)
        monitor.update(pose_name, goal, robot.get_joint_positions(), "planning")
        world.step(render=True)

        def _tick() -> None:
            monitor.update(pose_name, goal, robot.get_joint_positions(), "moving")

        robot.move_to_joints(goal, steps_per_wp=3, render=True, step_callback=_tick)
        robot.hold(goal, HOLD_STEPS, render=True)

        monitor.update(pose_name, goal, robot.get_joint_positions(), "holding")

    cycle = 0
    while app.is_running():
        cycle += 1
        print(f"[view] === Cycle {cycle} ===", flush=True)
        for name, goal in POSES:
            if not app.is_running():
                break
            move_to(name, goal)
        if not args.loop:
            monitor._status_lbl.text = "Status: done -- close window to exit"
            while app.is_running():
                world.step(render=True)
            break

    app.close()


if __name__ == "__main__":
    main()
