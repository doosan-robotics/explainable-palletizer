"""Minimal Isaac Sim standalone test: ground plane + dynamic cube + physics steps."""

import numpy as np
from isaacsim import SimulationApp


def main() -> None:
    simulation_app = SimulationApp({"headless": False})

    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from pxr import Sdf, UsdLux

    # Scene setup
    GroundPlane(prim_path="/World/GroundPlane", z_position=0)

    stage = omni.usd.get_context().get_stage()
    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    light.CreateIntensityAttr(300)

    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=np.array([0, 0, 1.0]),
        size=0.3,
        color=np.array([0.0, 0.5, 1.0]),
    )

    # Run simulation
    world = World(stage_units_in_meters=1.0)
    world.reset()

    for _step in range(200):
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
