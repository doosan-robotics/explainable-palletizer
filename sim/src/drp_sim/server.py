"""Sim server entrypoint: SimulationApp on main thread, uvicorn in daemon thread.

Isaac Sim requires ``SimulationApp`` to be created on the main thread before
any ``omni.*`` imports. This module:

1. Creates ``SimulationApp({"headless": True})``
2. Imports drp_sim modules (safe after SimulationApp exists)
3. Starts uvicorn in a daemon thread
4. Runs ``SimRunner.run()`` on the main thread (blocking)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # Isaac Sim's GUI extensions (loaded even in headless mode via [all]) call
    # asyncio.ensure_future() during USD stage initialisation. uvloop raises
    # RuntimeError if no event loop has been set on the current thread, so we
    # set one explicitly before SimulationApp is created.
    asyncio.set_event_loop(asyncio.new_event_loop())

    logger.info("Creating SimulationApp (headless=True)")
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": True})

    # SimulationApp.__init__ clears the asyncio event loop as part of its own
    # extension initialisation. Re-set it so that omni.kit.scripting's
    # asyncio.ensure_future() calls (triggered by any USD stage mutation) do
    # not raise "There is no current event loop in thread 'MainThread'".
    asyncio.set_event_loop(asyncio.new_event_loop())

    # Safe to import drp_sim modules now that SimulationApp exists
    from drp_sim.api import create_app
    from drp_sim.sim_runner import SimCommand, SimRunner

    load_robot = os.environ.get("SIM_LOAD_ROBOT", "true").lower() == "true"
    spawn_boxes = os.environ.get("SIM_SPAWN_BOXES", "true").lower() == "true"

    runner = SimRunner(load_robot=load_robot, spawn_boxes=spawn_boxes)
    app = create_app(runner)

    host = os.environ.get("SIM_HOST", "0.0.0.0")
    port = int(os.environ.get("SIM_PORT", "8100"))

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %d, requesting graceful shutdown", signum)
        runner.send_command(SimCommand.SHUTDOWN)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    def _run_uvicorn() -> None:
        import uvicorn

        logger.info("Starting uvicorn on %s:%d", host, port)
        uvicorn.run(app, host=host, port=port, log_level="info", loop="asyncio")

    server_thread = threading.Thread(target=_run_uvicorn, daemon=True)
    server_thread.start()

    logger.info("Entering sim main loop")
    runner.run(sim_app)


if __name__ == "__main__":
    main()
