"""Estimate camera geometry and per-frame "recorded" vehicle state for the simulator.

The dataset ships no calibration, so geometry is inferred from the road:
  * vanishing point  <- intersection of the ego lane lines (median over frames)
  * camera height h  <- lateral scale; ego lane width assumed 3.75 m
  * focal length f   <- NOT observable reliably here (dashed-line spacing gave
                        inconsistent results). Default 600 px (HFOV ~108 deg),
                        consistent with the hood occluding the road below ~6 m.
                        Lateral offset does not depend on f; heading, curvature
                        and speed scale with it.
  * static mask      <- the car's own hood + black lens border (kept fixed when
                        the simulator re-renders a shifted viewpoint)
Then for every frame: lane offset / heading / curvature / width of the recorded
car (UFLD + BEV fit) and its speed (optical flow of road features projected
onto the ground plane).

Outputs go to cache/: camera.json, static_mask.png, truth.npz

Usage:
    .venv/bin/python calibrate.py [--lane-width 3.75] [--focal 600]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from bev import CameraParams, GroundCamera, fit_ego_lane
from play_lanes import DEFAULT_DIR, frame_paths
from ufld import UFLD, LaneTracker

CACHE = Path(__file__).parent / "cache"
FPS = 30.0

# Hood silhouette traced from the time-averaged frame (u, v) in 1640x590 pixels.
HOOD_OUTLINE = [(0, 500), (150, 482), (340, 456), (500, 428), (650, 408), (820, 399),
                (1000, 403), (1150, 416), (1300, 440), (1450, 466), (1640, 492)]


def estimate_vanishing_point(model, paths):
    vps = []
    for p in paths:
        xs, ys = model(cv2.imread(str(p)))
        L, R = xs[1], xs[2]
        ok = ~np.isnan(L) & ~np.isnan(R) & (ys > 330)  # lower rows are close to straight
        if ok.sum() < 8:
            continue
        a1, b1 = np.polyfit(ys[ok], L[ok], 1)
        a2, b2 = np.polyfit(ys[ok], R[ok], 1)
        if abs(a1 - a2) < 1e-3:
            continue
        v0 = (b2 - b1) / (a1 - a2)
        vps.append((a1 * v0 + b1, v0))
    return np.median(np.array(vps), axis=0)


def estimate_height(model, paths, params, lane_width):
    params.h = 1.0
    cam = GroundCamera(params)
    widths = []
    for p in paths:
        xs, ys = model(cv2.imread(str(p)))
        if np.isnan(xs[1]).all() or np.isnan(xs[2]).all():
            continue
        st = fit_ego_lane(cam, xs, ys)
        if st is not None:
            widths.append(st.width)
    return lane_width / np.median(widths)


def static_mask(mean_gray):
    h, w = mean_gray.shape
    mask = np.zeros((h, w), np.uint8)
    poly = np.array(HOOD_OUTLINE + [(w, h), (0, h)], np.int32)
    poly[:, 1] -= 6  # include the hood's bright rim
    cv2.fillPoly(mask, [poly], 255)
    mask[mean_gray < 14] = 255  # black border left by lens undistortion
    return mask


class SpeedEstimator:
    """Forward speed from optical flow of road features, measured in ground meters.

    Features tracked from frame t-gap to t are projected onto the road plane;
    road points all shift by the same distance toward the car. Dirt on the
    windshield stays at zero shift, so a frame counts as moving only when most
    features agree on a positive shift.
    """

    def __init__(self, cam, static, gap=2):
        self.cam, self.gap = cam, gap
        roi = np.zeros(static.shape, np.uint8)
        roi[int(cam.horizon_v) + 25:, :] = 255
        roi[static > 0] = 0
        self.roi = cv2.erode(roi, np.ones((15, 15), np.uint8))
        self.hist = []

    def update(self, img):
        self.hist.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if len(self.hist) <= self.gap:
            return np.nan
        a, b = self.hist.pop(0), self.hist[-1]
        p0 = cv2.goodFeaturesToTrack(a, 400, 0.01, 7, mask=self.roi)
        if p0 is None:
            return np.nan
        lk = dict(winSize=(21, 21), maxLevel=3)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **lk)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pb - p0, axis=2)[:, 0] < 1.0)
        p0, p1 = p0[ok, 0], p1[ok, 0]
        X0, Y0 = self.cam.image_to_ground(p0[:, 0], p0[:, 1])
        _, Y1 = self.cam.image_to_ground(p1[:, 0], p1[:, 1])
        near = (Y0 > 4) & (Y0 < 25) & (np.abs(X0) < 8)
        dY = (Y0 - Y1)[near]
        if len(dY) < 10:
            return np.nan
        moving = dY > 0.04
        if moving.mean() < 0.3:
            return 0.0
        return np.median(dY[moving]) * FPS / self.gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--lane-width", type=float, default=3.75)
    ap.add_argument("--focal", type=float, default=600.0)
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)

    paths = frame_paths(Path(args.frame_dir))
    model = UFLD()
    sample = paths[::10]

    u0, v0 = estimate_vanishing_point(model, sample)
    params = CameraParams(f=args.focal, cx=820.0, cy=295.0, vp_u=u0, vp_v=v0, h=1.0)
    params.h = estimate_height(model, sample, params, args.lane_width)
    hfov = 2 * np.degrees(np.arctan(params.cx / params.f))
    print(f"vanishing point ({u0:.1f}, {v0:.1f}) | f={params.f:.0f}px (HFOV {hfov:.0f} deg, assumed) | h={params.h:.3f} m")
    params.save(CACHE / "camera.json")

    mean_gray = np.mean([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) for p in paths[::30]], axis=0)
    static = static_mask(mean_gray)
    cv2.imwrite(str(CACHE / "static_mask.png"), static)

    cam = GroundCamera(params)
    tracker = LaneTracker()
    speed_est = SpeedEstimator(cam, static)
    n = len(paths)
    offset, heading, curv, width, speed = (np.full(n, np.nan) for _ in range(5))
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        xs, ys = model(img)
        st = fit_ego_lane(cam, tracker.update(xs), ys)
        if st is not None:
            offset[i], heading[i], curv[i], width[i] = st.offset, st.heading, st.curvature, st.width
        speed[i] = speed_est.update(img)
        if i % 1000 == 0:
            print(f"  frame {i}/{n}")

    # Fill gaps; smooth speed with a 1 s median and forbid reversing.
    idx = np.arange(n)
    for a in (offset, heading, curv, width, speed):
        ok = ~np.isnan(a)
        a[~ok] = np.interp(idx[~ok], idx[ok], a[ok])
    k = 31
    pad = np.pad(speed, k // 2, mode="edge")
    speed = np.clip([np.median(pad[i:i + k]) for i in range(n)], 0, None)

    np.savez(CACHE / "truth.npz", offset=offset, heading=heading, curvature=curv, width=width, speed=speed)
    kmh = speed * 3.6
    print(f"speed km/h: moving(>5) {np.mean(kmh > 5):.0%} of frames, p95 {np.percentile(kmh, 95):.0f}, max {kmh.max():.0f}")
    print(f"lane width median {np.median(width):.2f} m, recorded lane changes ~{int((np.abs(np.diff(offset)) > 1.5).sum())}")


if __name__ == "__main__":
    main()
