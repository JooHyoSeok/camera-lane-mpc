"""Grade camera lane perception in MetaDrive, separately on straights and curves.

The car is driven by the oracle MPC (so every model sees the same states), and
each frame's perceived lane offset / heading / curvature is compared with the
simulator's ground truth at the camera point.

Usage (MetaDrive venv):
    .venv-md/bin/python md_eval_perception.py --model models/culane_18.onnx
    .venv-md/bin/python md_eval_perception.py --model models/culane_18_md.onnx
"""
import argparse

import md_mac  # noqa: F401
import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs import MetaDriveEnv

import md_drive as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--map", default="SCSCSCSC")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=500, help="max control steps per map")
    args = ap.parse_args()

    env = MetaDriveEnv(dict(
        use_render=True, map=args.map, traffic_density=0.0, num_scenarios=1000, start_seed=0,
        window_size=(320, 180), show_interface=False, show_logo=False, show_fps=False,
        sensors=dict(rgb_camera=(RGBCamera, M.CAM_W, M.CAM_H)), vehicle_config=dict(show_navi_mark=False),
        on_continuous_line_done=False, log_level=50,
    ))
    rows, lost = [], 0
    for seed in args.seeds:
        env.reset(seed=seed)
        env.engine.force_fps.disable()
        v = env.agent
        cam = M.CameraLanes(env, v, args.model)
        pilot = M.Autopilot(v.FRONT_WHEELBASE + v.REAR_WHEELBASE, v.max_steering)
        for _ in range(args.steps):
            _, _, _, perc, _ = cam(v)
            tr = M.Truth(env, v)
            if perc is None:
                lost += 1
            else:
                rows.append((tr.kappa_now != 0, perc.offset - tr.lat, np.degrees(-perc.heading - tr.psi),
                             perc.curvature - tr.kappa_now))
            steer, thr, _, _, _ = pilot(v.speed, 40 / 3.6, truth=tr)
            _, _, term, trunc, _ = env.step([steer, thr])
            if term or trunc:
                break
    env.close()

    r = np.array(rows, float)
    print(f"model {args.model} | frames {len(r) + lost}, lanes lost {lost}")
    for name, m in (("straight", r[:, 0] == 0), ("curve", r[:, 0] == 1)):
        e = np.abs(r[m])
        print(f"  {name:8s} n={m.sum():4d} | offset err mean {e[:, 1].mean():.3f} m, p95 {np.percentile(e[:, 1], 95):.3f} m"
              f" | heading err mean {e[:, 2].mean():.2f} deg, p95 {np.percentile(e[:, 2], 95):.2f}"
              f" | curvature err mean {e[:, 3].mean():.4f} 1/m")


if __name__ == "__main__":
    main()
