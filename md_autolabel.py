"""Auto-labeled lane dataset from MetaDrive for fine-tuning UFLD.

The simulator knows where every lane boundary is and the dashcam model
(md_drive.CameraLanes) is exact, so ground-truth boundaries projected into the
image are free, pixel-accurate labels in UFLD's own format:
  18 row anchors x 4 lane slots (left-left, ego-left, ego-right, right-right),
  each a class 0..199 (horizontal grid cell) or 200 (no lane at that row).

To cover the states the closed-loop controller will actually visit, the car is
driven by the oracle MPC toward a lateral target that wanders within +-1 m of
the lane center, changes lanes now and then (so the ego lane is sometimes the
leftmost, middle or rightmost one), and gets occasional sideways pushes, over
random S/C maps and speeds. A fixed block string is used; every seed samples new curve radii,
angles, directions and lane counts for it.

Usage (MetaDrive venv):
    .venv-md/bin/python md_autolabel.py --frames 3000 --out data/md_lanes
"""
import argparse
from pathlib import Path

import md_mac  # noqa: F401
import cv2
import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs import MetaDriveEnv

import md_drive as M
from ufld import GRID, ROW_ANCHORS, IN_W

COL_W = (IN_W - 1) / (GRID - 1)   # pixels per grid cell (matches ufld.UFLD decoding)


def lane_chain(env, vehicle, idx, ahead=90.0):
    """Lanes with index idx along the route, starting at the vehicle's road."""
    nav = vehicle.navigation
    graph = env.current_map.road_network.graph
    cur = nav.current_lane
    road = graph[cur.index[0]][cur.index[1]]
    if not 0 <= idx < len(road):
        return None
    chain = [road[idx]]
    ck = list(nav.checkpoints)
    i = ck.index(cur.index[1]) if cur.index[1] in ck else None
    total = road[idx].length
    while total < ahead + 60 and i is not None and i + 1 < len(ck):
        lanes = graph[ck[i]][ck[i + 1]]
        if idx >= len(lanes):
            break
        chain.append(lanes[idx])
        total += lanes[idx].length
        i += 1
    return chain


def boundary_points(chain, side, start_xy, step=0.5, ahead=90.0):
    """World points of one boundary (side -1 left, +1 right) from the car forward."""
    s0, _ = chain[0].local_coordinates(start_xy)
    pts, total = [], 0.0
    for k, ln in enumerate(chain):
        a = max(s0 - 5.0, 0.0) if k == 0 else 0.0
        for s in np.arange(a, ln.length, step):
            pts.append(ln.position(s, side * ln.width_at(s) / 2))
        total += ln.length - a
        if total > ahead:
            break
    return np.array(pts)


def make_labels(env, vehicle, cam):
    """(18, 4) int labels for the current frame, plus projected polylines for checking."""
    hd = vehicle.heading_theta
    cam_xy = np.asarray(vehicle.position) + M.CAM_FWD * M.forward(hd)
    graph = env.current_map.road_network.graph
    cur = vehicle.navigation.current_lane
    road = graph[cur.index[0]][cur.index[1]]
    # ego lane = the lane under the camera
    lats = [abs(ln.local_coordinates(cam_xy)[1]) for ln in road]
    ego = int(np.argmin(lats))
    slots = [(ego - 1, -1), (ego, -1), (ego, +1), (ego + 1, +1)]
    labels = np.full((len(ROW_ANCHORS), 4), GRID, np.int64)
    polys = []
    for j, (idx, side) in enumerate(slots):
        chain = lane_chain(env, vehicle, idx)
        if chain is None:
            polys.append(None)
            continue
        P = boundary_points(chain, side, cam_xy)
        d = P - cam_xy
        X, Y = d @ M.right(hd), d @ M.forward(hd)
        keep = Y > 1.5
        if keep.sum() < 4:
            polys.append(None)
            continue
        u, v = cam.ground_to_image(X[keep], Y[keep])
        # stop at the first point where the boundary turns back up the image (behind a crest / U-turn)
        order = np.arange(len(v))
        mono = np.r_[True, np.diff(v) < 0]
        cut = order[~mono][0] if (~mono).any() else len(v)
        u, v = u[:cut], v[:cut]
        polys.append(np.stack([u, v], 1))
        if len(v) < 2:
            continue
        vs, us = v[::-1], u[::-1]                       # increasing v for interp
        for r, va in enumerate(ROW_ANCHORS):
            if vs[0] <= va <= vs[-1]:
                x = np.interp(va, vs, us)
                if 0 <= x <= IN_W - 1:
                    labels[r, j] = int(np.clip(round((x + 1) / COL_W - 1), 0, GRID - 1))
    return labels, polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--every", type=int, default=2, help="keep one frame every N control steps")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default="data/md_lanes")
    ap.add_argument("--preview", type=int, default=6, help="save N label-check images")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    env = MetaDriveEnv(dict(
        use_render=True, map="CSCCSCC", traffic_density=0.0, num_scenarios=5000, start_seed=args.seed,
        window_size=(320, 180), show_interface=False, show_logo=False, show_fps=False,
        sensors=dict(rgb_camera=(RGBCamera, M.CAM_W, M.CAM_H)), vehicle_config=dict(show_navi_mark=False),
        on_continuous_line_done=False, out_of_road_done=True, log_level=50,
    ))
    seed = args.seed - 1

    def new_map():
        nonlocal seed
        seed += 1
        env.reset(seed=seed)
        env.engine.force_fps.disable()

    new_map()
    v = env.agent
    camlanes = M.CameraLanes(env, v)
    pilot = M.Autopilot(v.FRONT_WHEELBASE + v.REAR_WHEELBASE, v.max_steering)
    target, v_set, step, want_lane = 0.0, 40 / 3.6, 0, None
    all_labels, meta, previews = [], [], 0
    while len(all_labels) < args.frames:
        if step % 30 == 0:                                   # wander inside the lane
            target = float(rng.uniform(-1.0, 1.0))
            v_set = float(rng.uniform(20, 60)) / 3.6
        if step % 75 == 37 and rng.random() < 0.5:           # occasional push
            v.set_position(np.asarray(v.position) + rng.choice([-1, 1]) * rng.uniform(0.4, 1.0) * M.right(v.heading_theta))
        cur = v.navigation.current_lane
        n_lanes = len(env.current_map.road_network.graph[cur.index[0]][cur.index[1]])
        if want_lane is None or step % 80 == 0:              # pick a lane to be in (lane changes)
            want_lane = int(rng.integers(0, n_lanes))
        want_lane = min(want_lane, n_lanes - 1)
        truth = M.Truth(env, v)
        # steer toward the wandering target inside the wanted lane (lane index grows to the right)
        truth.lat -= target + (want_lane - cur.index[2]) * truth.width
        steer, thr, _, _, _ = pilot(v.speed, v_set, truth=truth)

        if step % args.every == 0 and v.speed > 1.0:
            img = camlanes.sensor.perceive(to_float=False, new_parent_node=v.origin,
                                           position=(0.0, M.CAM_FWD, M.CAM_UP), hpr=(0.0, M.CAM_PITCH, 0.0))
            img = np.ascontiguousarray(img[..., :3])
            labels, polys = make_labels(env, v, camlanes.cam)
            if (labels[:, 1:3] < GRID).sum() >= 10:          # need a visible ego lane
                k = len(all_labels)
                cv2.imwrite(str(out / "img" / f"{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                all_labels.append(labels)
                meta.append((seed, step))
                if previews < args.preview and k % 150 == 0:
                    vis = img.copy()
                    for j, p in enumerate(polys):
                        if p is not None:
                            cv2.polylines(vis, [p.astype(np.int32)], False, (0, 255, 255) if j in (1, 2) else (255, 0, 255), 2)
                        for r, va in enumerate(ROW_ANCHORS):
                            if labels[r, j] < GRID:
                                x = (labels[r, j] + 1) * COL_W - 1
                                cv2.circle(vis, (int(x), int(va)), 3, (0, 0, 255), -1)
                    cv2.imwrite(str(out / f"label_check_{previews}.jpg"), vis)
                    previews += 1
                if len(all_labels) % 500 == 0:
                    print(f"  {len(all_labels)} frames")

        _, _, term, trunc, info = env.step([steer, thr])
        step += 1
        if term or trunc:
            new_map()
            pilot.reset()

    np.savez(out / "labels.npz", labels=np.stack(all_labels), meta=np.array(meta))
    print(f"saved {len(all_labels)} frames from {seed - args.seed + 1} maps to {out}")
    env.close()


if __name__ == "__main__":
    main()
