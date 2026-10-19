"""Drive MetaDrive yourself with the keyboard (3D window, macOS-ready).

Keys (MetaDrive built-ins):
  W/S throttle/brake, A/D steer, R reset, [ / ] previous/next map,
  Q third-person camera, B top-down camera, mouse wheel camera height,
  F toggle 30 fps limit, H help overlay, Esc quit.

Usage (from this folder, using the MetaDrive venv):
    .venv-md/bin/python md_play.py                 # random 7-block maps with traffic
    .venv-md/bin/python md_play.py --map SCXCO     # S straight, C curve, X crossroad, O roundabout, T, r/R ramps
    .venv-md/bin/python md_play.py --traffic 0.3 --topdown
"""
import argparse

import md_mac  # noqa: F401  (must come before MetaDrive creates its window)
from metadrive.envs import MetaDriveEnv


def parse_map(s):
    return int(s) if s.isdigit() else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=parse_map, default=7, help="block count or block string, e.g. SCXCO")
    ap.add_argument("--traffic", type=float, default=0.15, help="traffic density 0..1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topdown", action="store_true", help="also open a 2D top-down window")
    ap.add_argument("--steps", type=int, default=0, help="exit after N steps (for testing)")
    ap.add_argument("--screenshot", default=None, help="save the 3D window here on exit (for testing)")
    args = ap.parse_args()

    env = MetaDriveEnv(dict(
        use_render=True,
        manual_control=True,
        map=args.map,
        traffic_density=args.traffic,
        num_scenarios=100,
        start_seed=args.seed,
        window_size=(1280, 720),
        show_interface=True,
        log_level=50,
    ))
    try:
        env.reset()
        print(__doc__.split("Usage")[0])
        step = 0
        while True:
            _, _, terminated, truncated, info = env.step([0.0, 0.0])  # keyboard overrides the action
            if args.topdown:
                env.render(mode="topdown", window=True, screen_size=(600, 600), scaling=4)
            if terminated or truncated:
                reason = [k for k in ("arrive_dest", "crash_vehicle", "out_of_road", "max_step") if info.get(k)]
                print("episode ended:", ", ".join(reason) or "done")
                env.reset()
            step += 1
            if args.steps and step >= args.steps:
                break
        if args.screenshot:
            from panda3d.core import Filename
            env.engine.win.save_screenshot(Filename.from_os_specific(args.screenshot))
    finally:
        env.close()


if __name__ == "__main__":
    main()
