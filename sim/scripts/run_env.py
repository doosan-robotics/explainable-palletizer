"""Run the PalletizerEnv with the mixed palletizing USD scene."""

from drp_sim import PalletizerEnv


def main() -> None:
    # TODO:Need to load yaml
    env = PalletizerEnv(
        headless=False,
        spawn_boxes=True,
        spawn_position=(0.59, 3.01, -0.20),
        box_scale=(0.6, 0.6, 0.6),
        spawn_interval=3.0,
    )
    env.reset()

    try:
        while env._app.is_running():
            env.step(render=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[run_env] Exception: {type(e).__name__}: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
