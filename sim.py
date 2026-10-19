"""Closed-loop lane-keeping simulator on a recorded drive, with a HUD.

The recording supplies forward motion and the road. Our virtual car has its
own lateral state (position in the lane, heading), driven by a kinematic
bicycle model and the steering input. Every frame is re-rendered from the
virtual car's viewpoint (exact for the road plane, see bev.py) and the lane
detector runs on that re-rendered view, so steering changes what the
perception stack sees next.

Steering is automatic by default: a lateral MPC (controller.py) plans 3 s
ahead from the perceived lane, and its predicted path is drawn on the HUD.

Run calibrate.py once first (writes cache/).

Keys: M auto (MPC) / manual, O controller input: perception / oracle (sim truth),
<-/-> or A/D steer (takes over from auto), S center the wheel, P push sideways,
R reset to the recorded pose, N/B skip +/-10 s, Space pause, Q quit.

Usage:
    .venv/bin/python sim.py [--start 0] [--speed KMH] [--manual] [--oracle]
    .venv/bin/python sim.py --headless 900 --push-every 150 --out hud.png   # no window; prints metrics
"""
import argparse
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from controller import AutoSteer
from hud_kit import C, Text, blend_poly, dashed, rrect, side_word
from bev import CameraParams, GroundCamera, fit_ego_lane, lane_points_ground
from play_lanes import DEFAULT_DIR, frame_paths
from ufld import UFLD, LaneTracker, lane_polyline, EGO_LEFT, EGO_RIGHT

CACHE = Path(__file__).parent / "cache"
FPS = 30.0
DT = 1.0 / FPS
WHEELBASE = 2.875     # Model 3-class sedan [m]
STEER_RATIO = 15.0    # steering-wheel deg per road-wheel deg
CAR_W, CAR_FRONT, CAR_REAR = 1.85, 2.2, -2.5  # body extent around the camera [m]
MAX_DX, MAX_DPSI = 2.4, np.radians(12)         # beyond this the re-render degrades

KEY_LEFT = {ord("a"), 63234, 2424832, 65361}
KEY_RIGHT = {ord("d"), 63235, 2555904, 65363}


# ======================================================================
# Recorded drive and vehicle model
# ======================================================================
def smooth_segments(a, k, breaks):
    """Moving average that does not smear across lane-change jumps."""
    out = a.copy()
    edges = [0, *breaks, len(a)]
    for s, e in zip(edges[:-1], edges[1:]):
        seg = a[s:e]
        if len(seg) < 3:
            continue
        kk = min(k, len(seg) // 2 * 2 + 1)
        pad = np.pad(seg, kk // 2, mode="edge")
        out[s:e] = np.convolve(pad, np.ones(kk) / kk, "valid")
    return out


class RecordedDrive:
    def __init__(self, path=CACHE / "truth.npz", speed_override=None):
        t = np.load(path)
        off = t["offset"]
        breaks = list(np.flatnonzero(np.abs(np.diff(off)) > 1.5) + 1)
        self.lane_change = np.zeros(len(off), bool)
        self.lane_change[breaks] = True
        self.y = smooth_segments(off, 7, breaks)                    # + right of center
        self.psi = -smooth_segments(t["heading"], 9, breaks)        # car vs lane, + left
        self.kappa = np.clip(smooth_segments(t["curvature"], 31, breaks), -0.02, 0.02)
        self.width = smooth_segments(t["width"], 31, breaks)
        speed = np.clip(t["speed"], 0, 110 / 3.6)
        self.v = np.full_like(speed, speed_override / 3.6) if speed_override else speed
        self.n = len(off)


@dataclass
class Vehicle:
    y: float = 0.0      # lateral position, + right of lane center [m]
    psi: float = 0.0    # heading relative to the lane, + left [rad]
    wheel: float = 0.0  # steering-wheel angle, + left [deg]

    def step(self, v, kappa, dt=DT):
        delta = np.radians(self.wheel / STEER_RATIO)
        self.y += -v * np.sin(self.psi) * dt
        self.psi += (v / WHEELBASE * np.tan(delta) - v * kappa) * dt


# ======================================================================
# HUD
# ======================================================================
class Hud:
    W, H = 1600, 920
    VIEW = (16, 60, 1184, 426)        # x, y, w, h  (1640x590 scaled by 0.722)
    BEV = (1216, 60, 368, 824)
    CARDS_Y, CARDS_H = 500, 220
    GRAPH = (16, 736, 1184, 148)
    BEV_PPM, BEV_Y0 = 18, -6.0

    def __init__(self, cam):
        self.cam = cam
        self.t = Text()
        bx, by, bw, bh = self.BEV
        self.bev_size = (bw, bh - 36)
        half = bw / 2 / self.BEV_PPM
        self.bev_xr = (-half, half)
        self.bev_yr = (self.BEV_Y0, self.BEV_Y0 + self.bev_size[1] / self.BEV_PPM)

    # ---------------- helpers ----------------
    def bev_px(self, X, Y):
        bx, by, _, _ = self.BEV
        col = (np.asarray(X) - self.bev_xr[0]) * self.BEV_PPM + bx
        row = (self.bev_yr[1] - np.asarray(Y)) * self.BEV_PPM + by + 36
        return col, row

    def card(self, img, x, y, w, h, title):
        rrect(img, x, y, w, h, 14, C["panel"])
        self.t((x + 18, y + 16), title, 13, C["muted"], "medium")

    # ---------------- sections ----------------
    def top_bar(self, img, info):
        t = self.t
        t((18, 14), "LANE SIM", 18, C["text"], "bold", mono=True)
        t((130, 16), "녹화 영상 폐루프 차선 유지", 15, C["muted"], "medium")
        x = 340
        if info["auto"]:
            chips = [("자동 조향 · MPC", C["plan"]),
                     ("입력: 오라클" if info["oracle"] else "입력: 인식", C["warn"] if info["oracle"] else C["muted"])]
        else:
            chips = [("수동 조향", C["path"])]
        if info["paused"]:
            chips.append(("일시정지", C["warn"]))
        for label, col in chips:
            w = int(t.width(label, 13, "semibold")) + 24
            rrect(img, x, 12, w, 26, 13, col, 1)
            t((x + 12, 17), label, 13, col, "semibold")
            x += w + 8
        sec = info["i"] / FPS
        total = info["n"] / FPS
        clock = f"{int(sec // 60):02d}:{sec % 60:04.1f} / {int(total // 60):02d}:{int(total % 60):02d}"
        t((800, 16), clock, 15, C["text"], "medium", mono=True, anchor="ma")
        t((930, 17), f"#{info['i'] + 1:05d}", 13, C["dim"], mono=True)
        mpc = f"MPC {info['plan'].solve_ms:3.1f} ms · " if info["plan"] is not None else ""
        t((self.W - 18, 17), f"{mpc}처리 {info['proc_ms']:4.1f} ms · {info['hz']:4.1f} Hz", 13, C["muted"], anchor="ra")

    def main_view(self, img, virt, original, xs, ys, info):
        x, y, w, h = self.VIEW
        sx = w / virt.shape[1]
        view = cv2.resize(virt, (w, h), interpolation=cv2.INTER_LINEAR)
        lanes = {i: lane_polyline(xs[i], ys, virt.shape[0]) for i in range(len(xs))}
        L, R = lanes.get(EGO_LEFT), lanes.get(EGO_RIGHT)
        if L is not None and R is not None:
            blend_poly(view, (np.vstack([L, R[::-1]]) * sx).astype(np.int32), C["path"], 0.18)
        for i, pts in lanes.items():
            if pts is None:
                continue
            ego = i in (EGO_LEFT, EGO_RIGHT)
            cv2.polylines(view, [(pts * sx).astype(np.int32)], False, C["line"] if ego else C["muted"],
                          4 if ego else 2, cv2.LINE_AA)
        plan = info["plan"]
        if plan is not None:
            self.draw_plan_camera(view, plan, sx)
        img[y:y + h, x:x + w] = view
        rrect(img, x, y, w, h, 2, C["edge"], 1)

        # Picture-in-picture: the untouched recording.
        iw, ih = 300, 108
        ix, iy = x + 14, y + 14
        img[iy:iy + ih, ix:ix + iw] = cv2.resize(original, (iw, ih), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (ix, iy), (ix + iw, iy + ih), C["edge"], 1)
        rrect(img, ix, iy + ih + 6, iw, 24, 6, C["panel"])
        dx = info["dx"]
        self.t((ix + 10, iy + ih + 10), "녹화 원본", 12, C["muted"], "semibold")
        self.t((ix + iw - 10, iy + ih + 10), f"시뮬 차량: {side_word(dx)} {abs(dx):.2f} m · {np.degrees(info['dpsi']):+.1f}°",
               12, C["text"], mono=False, anchor="ra")

        if info["banner"]:
            text, col = info["banner"]
            bw = int(self.t.width(text, 16, "bold")) + 40
            bx = x + (w - bw) // 2
            rrect(img, bx, y + 16, bw, 36, 18, col)
            self.t((bx + bw // 2, y + 25), text, 16, C["bg"], "bold", anchor="ma")

    def draw_plan_camera(self, view, plan, sx, half_w=0.5):
        """Project the MPC path (vehicle-frame ground points) into the camera view."""
        P = plan.path
        keep = P[:, 1] > 3.0  # closer than this is under the hood / below the frame
        if keep.sum() < 2:
            return
        P = P[keep]
        d = np.gradient(P, axis=0)
        n = np.stack([d[:, 1], -d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)  # right normal

        def proj(Q):
            u, v = self.cam.ground_to_image(Q[:, 0], Q[:, 1])
            return np.stack([u, v], 1) * sx

        band = np.vstack([proj(P - n * half_w), proj(P + n * half_w)[::-1]]).astype(np.int32)
        blend_poly(view, band, C["plan"], 0.45)
        cv2.polylines(view, [proj(P).astype(np.int32)], False, C["plan"], 2, cv2.LINE_AA)
        for t_mark in (1.0, 2.0, 3.0):
            k = np.argmin(np.abs(plan.times[keep] - t_mark))
            if abs(plan.times[keep][k] - t_mark) < 0.06:
                c = proj(P[k:k + 1])[0].astype(int)
                cv2.circle(view, tuple(c), 5, C["bg"], -1, cv2.LINE_AA)
                cv2.circle(view, tuple(c), 5, C["plan"], 2, cv2.LINE_AA)

    def bev_panel(self, img, virt, perc_lanes, perc, truth, info):
        bx, by, bw, bh = self.BEV
        rrect(img, bx, by, bw, bh, 14, C["panel"])
        self.t((bx + 16, by + 12), "BEV · 위에서 본 시점", 13, C["muted"], "medium")
        top = by + 36
        bev = self.cam.warp_bev(virt, self.bev_xr, self.bev_yr, self.BEV_PPM)
        bev = (bev * 0.5).astype(np.uint8)
        # Hood / blind zone in front of the bumper: not road, paint it as panel.
        _, r_blind = self.bev_px(0, 6.0)
        bev[int(r_blind - top):] = C["panel"]
        img[top:top + bev.shape[0], bx:bx + bw] = bev

        # Distance ticks
        for d in range(0, int(self.bev_yr[1]) + 1, 10):
            _, r = self.bev_px(0, d)
            cv2.line(img, (bx, int(r)), (bx + 10, int(r)), C["muted"], 1, cv2.LINE_AA)
            rrect(img, bx + 13, int(r) - 10, 44, 20, 6, C["panel"])
            self.t((bx + 35, int(r) - 7), f"{d} m", 11, C["muted"], mono=True, anchor="ma")

        Y = np.linspace(0.0, self.bev_yr[1] - 1, 60)

        def poly_pts(Xs):
            c, r = self.bev_px(Xs, Y)
            return np.stack([c, r], 1).astype(np.int32)

        # Driving path (perceived lane center, car-width band), Tesla style.
        if perc is not None:
            Xc = np.polyval(perc.center, Y)
            band = np.vstack([poly_pts(Xc - CAR_W / 2), poly_pts(Xc + CAR_W / 2)[::-1]])
            blend_poly(img, band, C["path"], 0.18)
        # Perceived boundaries (all lanes).
        for i, pts in enumerate(perc_lanes):
            if pts is None or len(pts) < 3:
                continue
            c = np.polyfit(pts[:, 1], pts[:, 0], 2 if len(pts) >= 6 else 1)
            Yl = np.linspace(max(6.0, pts[:, 1].min()), min(self.bev_yr[1], pts[:, 1].max() + 4), 40)
            col, row = self.bev_px(np.polyval(c, Yl), Yl)
            ego = i in (EGO_LEFT, EGO_RIGHT)
            cv2.polylines(img, [np.stack([col, row], 1).astype(np.int32)], False,
                          C["line"] if ego else C["muted"], 3 if ego else 2, cv2.LINE_AA)
        # Ground-truth lane of the simulation (where the virtual car really is).
        y, psi, kappa, w = truth
        Xc = -y + np.tan(psi) * Y - kappa * Y ** 2 / 2
        for s in (-1, 1):
            p = poly_pts(Xc + s * w / 2)
            for k in range(0, len(p) - 1, 2):
                cv2.line(img, tuple(p[k]), tuple(p[k + 1]), C["truth"], 2, cv2.LINE_AA)

        # MPC predicted path (3 s), with 1 s / 2 s / 3 s marks
        plan = info["plan"]
        if plan is not None and np.ptp(plan.path[:, 1]) > 0.3:
            P = plan.path[plan.path[:, 1] < self.bev_yr[1] - 0.5]  # keep inside the panel
            col, row = self.bev_px(P[:, 0], P[:, 1])
            pts = np.stack([col, row], 1)
            band = np.vstack([pts - [CAR_W / 2 * self.BEV_PPM, 0], (pts + [CAR_W / 2 * self.BEV_PPM, 0])[::-1]])
            blend_poly(img, band.astype(np.int32), C["plan"], 0.28)
            cv2.polylines(img, [pts.astype(np.int32)], False, C["plan"], 3, cv2.LINE_AA)
            for t_mark in (1, 2, 3):
                k = int(round(t_mark / (plan.times[1] - plan.times[0])))
                if k < len(pts):
                    c = tuple(pts[k].astype(int))
                    cv2.circle(img, c, 6, C["panel"], -1, cv2.LINE_AA)
                    cv2.circle(img, c, 6, C["plan"], 2, cv2.LINE_AA)
                    self.t((c[0] + 12, c[1] - 8), f"{t_mark}s", 11, C["plan"], "semibold")

        # Ego car
        c0, r0 = self.bev_px(-CAR_W / 2, CAR_FRONT)
        c1, r1 = self.bev_px(CAR_W / 2, CAR_REAR)
        rrect(img, int(c0), int(r0), int(c1 - c0), int(r1 - r0), 9, C["car"])
        wc0, wr0 = self.bev_px(-CAR_W / 2 + 0.2, 0.9)
        wc1, wr1 = self.bev_px(CAR_W / 2 - 0.2, 0.1)
        rrect(img, int(wc0), int(wr0), int(wc1 - wc0), int(wr1 - wr0), 4, C["panel"])

        # Legend
        ly = by + bh - 26
        cv2.line(img, (bx + 16, ly + 7), (bx + 38, ly + 7), C["line"], 3, cv2.LINE_AA)
        self.t((bx + 44, ly), "인식 (UFLD)", 12, C["muted"])
        cv2.line(img, (bx + 128, ly + 7), (bx + 150, ly + 7), C["plan"], 3, cv2.LINE_AA)
        self.t((bx + 156, ly), "MPC 예측", 12, C["muted"])
        for k in range(3):
            cv2.line(img, (bx + 236 + k * 8, ly + 7), (bx + 240 + k * 8, ly + 7), C["truth"], 2, cv2.LINE_AA)
        self.t((bx + 264, ly), "실제 차선", 12, C["muted"])

    def cards(self, img, info):
        x0, y0, h = 16, self.CARDS_Y, self.CARDS_H
        w = (self.VIEW[2] - 3 * 16) // 4
        xs = [x0 + k * (w + 16) for k in range(4)]
        t = self.t

        # 1) speed
        x = xs[0]
        self.card(img, x, y0, w, h, "속도")
        kmh = info["v"] * 3.6
        t((x + 18, y0 + 58), f"{kmh:.0f}", 64, C["text"], "semibold", mono=True)
        t((x + 22 + t.width(f"{kmh:.0f}", 64, "semibold", True), y0 + 98), "km/h", 15, C["muted"], "medium")
        t((x + 18, y0 + h - 34), "녹화 차량 · 광류 추정", 12, C["dim"])

        # 2) position in lane
        x = xs[1]
        y_true, width = info["y"], info["width"]
        margin = width / 2 - abs(y_true) - CAR_W / 2
        state_col = C["ok"] if margin > 0.3 else C["warn"] if margin > 0 else C["bad"]
        self.card(img, x, y0, w, h, "차선 내 위치")
        t.pair((x + 18, y0 + 76), side_word(y_true), f"{abs(y_true):.2f} m", 30, state_col)
        po = info["perc_offset"]
        t((x + 18, y0 + 84), "인식: " + (f"{side_word(po)} {abs(po):.2f} m" if po is not None else "—"), 12, C["muted"])
        # lane diagram
        lx0, lx1, ly = x + 22, x + w - 22, y0 + 128
        ppm = (lx1 - lx0) / (width + 1.0)
        cxp = (lx0 + lx1) / 2
        for s in (-1, 1):
            px = int(cxp + s * width / 2 * ppm)
            cv2.line(img, (px, ly), (px, ly + 62), C["text"], 3, cv2.LINE_AA)
        dashed(img, (cxp, ly), (cxp, ly + 62), C["dim"], 1, 6, 6)
        car_c = cxp + y_true * ppm
        rrect(img, int(car_c - CAR_W / 2 * ppm), ly + 10, int(CAR_W * ppm), 42, 8, state_col)
        rrect(img, int(car_c - CAR_W / 2 * ppm) + 6, ly + 16, int(CAR_W * ppm) - 12, 10, 3, C["panel"])

        # 3) heading & curvature
        x = xs[2]
        self.card(img, x, y0, w, h, "진행 방향 · 도로 곡률")
        psi_deg = np.degrees(info["psi"])
        t.pair((x + 18, y0 + 76), side_word(-psi_deg, 0.1), f"{abs(psi_deg):.1f}°", 30, C["text"])
        k = info["kappa"]
        t((x + 18, y0 + 84), "직선 도로" if abs(k) < 1 / 2000 else f"{'좌' if k > 0 else '우'}커브 · 반경 {1 / abs(k):.0f} m",
          12, C["muted"])
        # arrow relative to lane (angle exaggerated x4 so small errors are visible)
        ax, ay = x + w // 2, y0 + 168
        for s in (-1, 1):
            cv2.line(img, (ax + s * 46, ay - 44), (ax + s * 46, ay + 40), C["dim"], 2, cv2.LINE_AA)
        a = -info["psi"] * 4
        tip = (int(ax + 40 * np.sin(a)), int(ay - 40 * np.cos(a)))
        tail = (int(ax - 30 * np.sin(a)), int(ay + 30 * np.cos(a)))
        cv2.arrowedLine(img, tail, tip, C["line"], 4, cv2.LINE_AA, tipLength=0.3)
        t((x + w - 18, y0 + h - 34), "각도 ×4 확대", 11, C["dim"], anchor="ra")

        # 4) steering
        x = xs[3]
        self.card(img, x, y0, w, h, "조향")
        wheel = info["wheel"]
        t.pair((x + 18, y0 + 76), side_word(-wheel, 0.5), f"{abs(wheel):.0f}°", 30, C["text"])
        t((x + 18, y0 + 84), f"바퀴 {wheel / STEER_RATIO:+.1f}° · 핸들 {STEER_RATIO:.0f}:1", 12, C["muted"])
        cx, cy, r = x + 68, y0 + 160, 40
        cv2.circle(img, (cx, cy), r, C["text"], 7, cv2.LINE_AA)
        a = -np.radians(wheel)  # + left = counter-clockwise on screen
        for base in (0.0, np.pi, np.pi / 2):  # left, right, bottom spokes
            ang = base + a
            cv2.line(img, (cx, cy), (int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))), C["text"], 5, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 11, C["panel"], -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 11, C["text"], 3, cv2.LINE_AA)
        top = (int(cx + r * np.cos(-np.pi / 2 + a)), int(cy + r * np.sin(-np.pi / 2 + a)))
        cv2.circle(img, top, 6, C["plan"] if info["auto"] else C["path"], -1, cv2.LINE_AA)
        # planned steering-wheel angle over the horizon
        gx0, gx1, gy0, gy1 = x + 126, x + w - 18, y0 + 118, y0 + 200
        rrect(img, gx0, gy0, gx1 - gx0, gy1 - gy0, 8, C["bg"])
        gm = (gy0 + gy1) // 2
        cv2.line(img, (gx0 + 6, gm), (gx1 - 6, gm), C["dim"], 1, cv2.LINE_AA)
        plan = info["plan"]
        if plan is not None:
            wd = np.degrees(plan.deltas) * STEER_RATIO
            scale = max(60.0, np.abs(wd).max() * 1.15)
            px = np.linspace(gx0 + 6, gx1 - 6, len(wd))
            py = gm - wd / scale * (gy1 - gy0) / 2
            cv2.polylines(img, [np.stack([px, py], 1).astype(np.int32)], False, C["plan"], 2, cv2.LINE_AA)
            t((gx1 - 8, gy1 - 18), f"±{scale:.0f}°", 10, C["dim"], mono=True, anchor="ra")
            t((gx0 + 8, gy0 + 6), "예측 3초", 11, C["muted"])
        else:
            t(((gx0 + gx1) // 2, gm - 8), "수동", 13, C["dim"], anchor="ma")

    def graph(self, img, hist, info):
        x, y, w, h = self.GRAPH
        self.card(img, x, y, w, h, "차선 중앙 대비 위치 · 최근 10초")
        px0, px1 = x + 60, x + w - 230
        p_now = px0 + (px1 - px0) * 10 / 13  # 10 s history | 3 s forecast
        py0, py1 = y + 40, y + h - 16
        ymax = 2.2

        def to_py(v):
            return py0 + (py1 - py0) * (0.5 - v / (2 * ymax))

        half = info["width"] / 2
        cv2.line(img, (int(p_now), py0 - 6), (int(p_now), py1), C["dim"], 1, cv2.LINE_AA)
        self.t((int(p_now), py0 - 20), "지금", 11, C["muted"], anchor="ma")
        for v, col in ((half, C["bad"]), (-half, C["bad"]), (half - CAR_W / 2, C["warn"]), (-(half - CAR_W / 2), C["warn"])):
            dashed(img, (px0, to_py(v)), (px1, to_py(v)), col, 1, 5, 5)
        cv2.line(img, (px0, int(to_py(0))), (px1, int(to_py(0))), C["dim"], 1, cv2.LINE_AA)
        for v in (-2, 0, 2):
            self.t((px0 - 10, int(to_py(v)) - 7), f"{v:+d}" if v else "0", 11, C["dim"], mono=True, anchor="ra")
        self.t((x + 18, py1 - 12), "좌", 11, C["dim"])
        self.t((x + 18, py0 - 2), "우", 11, C["dim"])

        n = hist.maxlen
        for key, col, th in (("perc", C["path"], 2), ("true", C["text"], 2)):
            vals = [hv[key] for hv in hist]
            pts = [(p_now - (len(vals) - 1 - k) * (p_now - px0) / (n - 1), to_py(np.clip(-v, -ymax, ymax)))
                   for k, v in enumerate(vals) if v is not None]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, col, th, cv2.LINE_AA)
        plan = info["plan"]
        if plan is not None:
            fx = p_now + plan.times / 3.0 * (px1 - p_now)
            fy = to_py(np.clip(-plan.y_pred, -ymax, ymax))
            fp = np.stack([fx, fy], 1).astype(np.int32)
            for k in range(0, len(fp) - 1, 2):
                cv2.line(img, tuple(fp[k]), tuple(fp[k + 1]), C["plan"], 2, cv2.LINE_AA)

        # legend + stats
        sx = px1 + 24
        cv2.line(img, (sx, y + 22), (sx + 18, y + 22), C["text"], 2, cv2.LINE_AA)
        self.t((sx + 24, y + 15), "실제", 12, C["muted"])
        cv2.line(img, (sx + 62, y + 22), (sx + 80, y + 22), C["path"], 2, cv2.LINE_AA)
        self.t((sx + 86, y + 15), "인식", 12, C["muted"])
        cv2.line(img, (sx + 124, y + 22), (sx + 142, y + 22), C["plan"], 2, cv2.LINE_AA)
        self.t((sx + 148, y + 15), "예측", 12, C["muted"])
        stats = [("차선 이탈", f"{info['departures']}회", C["bad"] if info["departures"] else C["text"]),
                 ("인식 오차 RMS", f"{info['perc_rms']:.2f} m", C["text"]),
                 ("복귀 (렌더 한계)", f"{info['resets']}회", C["text"])]
        for k, (label, val, col) in enumerate(stats):
            yy = y + 46 + k * 30
            self.t((sx, yy), label, 13, C["muted"])
            self.t((x + w - 18, yy), val, 14, col, "semibold", mono=True, anchor="ra")

    def help_line(self):
        self.t((18, self.H - 28),
               "M 자동/수동   O 입력 인식/오라클   ←/→·A/D 조향(자동 해제)   S 핸들 중앙   P 옆으로 밀기   "
               "R 복귀   N/B ±10초   Space 일시정지   Q 종료",
               13, C["dim"])

    def draw(self, virt, original, xs, ys, perc_lanes, perc, truth, hist, info):
        img = np.empty((self.H, self.W, 3), np.uint8)
        img[:] = C["bg"]
        self.top_bar(img, info)
        self.main_view(img, virt, original, xs, ys, info)
        self.bev_panel(img, virt, perc_lanes, perc, truth, info)
        self.cards(img, info)
        self.graph(img, hist, info)
        self.help_line()
        return self.t.render(img)


# ======================================================================
# Main loop
# ======================================================================
class Prefetcher:
    """Decode the next JPEG on a worker thread while this frame is processed."""

    def __init__(self, paths):
        self.paths = paths
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futs = {}
        self.last = (None, None)

    def get(self, i):
        if self.last[0] == i:
            return self.last[1]
        f = self.futs.pop(i, None)
        img = f.result() if f else cv2.imread(str(self.paths[i]))
        for k in [k for k in self.futs if k != i + 1]:
            self.futs.pop(k).cancel()
        if i + 1 < len(self.paths) and i + 1 not in self.futs:
            self.futs[i + 1] = self.pool.submit(cv2.imread, str(self.paths[i + 1]))
        self.last = (i, img)
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument("--speed", type=float, default=None, help="override speed [km/h]")
    ap.add_argument("--headless", type=int, default=0, help="run N frames without a window")
    ap.add_argument("--out", default="hud.png", help="headless: save the last HUD frame here")
    ap.add_argument("--push-at", type=int, default=None, help="headless: frame offset to push the car 0.8 m right")
    ap.add_argument("--push-every", type=int, default=0, help="headless: push 0.8 m left/right every N frames")
    ap.add_argument("--manual", action="store_true", help="start in manual steering")
    ap.add_argument("--oracle", action="store_true", help="controller reads sim truth instead of perception")
    args = ap.parse_args()

    if not (CACHE / "camera.json").exists():
        raise SystemExit("cache/ not found: run calibrate.py first")
    cam = GroundCamera(CameraParams.load(CACHE / "camera.json"))
    static = cv2.imread(str(CACHE / "static_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    rec = RecordedDrive(speed_override=args.speed)
    paths = frame_paths(Path(args.frame_dir))
    model = UFLD()
    hud = Hud(cam)
    frames = Prefetcher(paths)

    car = Vehicle()
    tracker = LaneTracker()
    auto = AutoSteer(WHEELBASE, STEER_RATIO, DT)
    mode_auto, oracle = not args.manual, args.oracle
    rng = np.random.default_rng(0)
    metrics = dict(abs_y=[], wheel_rate=[], solve_ms=[])
    hist = deque(maxlen=int(10 * FPS))
    err2 = deque(maxlen=int(10 * FPS))
    stats = dict(departures=0, resets=0)
    banner = {"text": None, "col": None, "until": -1}
    in_departure = False
    last_steer_key = -1e9

    def reset_to_recording(i):
        car.y, car.psi = rec.y[i], rec.psi[i]
        tracker.__init__()
        auto.reset(car.wheel)

    def show_banner(text, col, i, secs=1.5):
        banner.update(text=text, col=col, until=i + int(secs * FPS))

    win = "LANE SIM"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1440, 828)

    i = max(1, args.start)
    reset_to_recording(i)
    paused = False
    t_prev = time.perf_counter()
    hz = FPS
    frames_done = 0
    out = None
    proc_ms = 0.0
    proc_log = []
    last_plan = None

    while 0 < i < rec.n:
        t0 = time.perf_counter()
        img = frames.get(i)

        if not paused:
            if rec.lane_change[i]:
                show_banner("녹화 차량 차선 변경 · 시뮬 차량도 함께 이동", C["path"], i)
            car.step(rec.v[i], rec.kappa[i])

        dx, dpsi = car.y - rec.y[i], car.psi - rec.psi[i]
        if abs(dx) > MAX_DX or abs(dpsi) > MAX_DPSI:
            stats["resets"] += 1
            show_banner("렌더 한계 초과 · 녹화 위치로 복귀", C["warn"], i)
            reset_to_recording(i)
            dx, dpsi = 0.0, 0.0

        virt = cam.render_virtual(img, dx, dpsi, static)
        xs, ys = model(virt)
        xs = tracker.update(xs)
        perc = fit_ego_lane(cam, xs, ys)
        perc_lanes = lane_points_ground(cam, xs, ys)

        plan = None
        if mode_auto and not paused:
            prev_wheel = car.wheel
            plan = auto(perc, rec.v[i], oracle=(car.y, car.psi, rec.kappa[i]) if oracle else None)
            if plan is not None:
                car.wheel = auto.wheel_deg()  # applied from the next step on (one-frame latency)
                metrics["wheel_rate"].append((car.wheel - prev_wheel) / DT)
                metrics["solve_ms"].append(plan.solve_ms)
        elif mode_auto and paused:
            plan = last_plan
        last_plan = plan
        if not paused:
            metrics["abs_y"].append(abs(car.y))

        width = rec.width[i]
        if abs(car.y) > width / 2:
            if not in_departure:
                stats["departures"] += 1
                show_banner("차선 이탈", C["bad"], i)
            in_departure = True
        elif abs(car.y) < width / 2 - 0.2:
            in_departure = False

        if not paused:
            hist.append({"true": car.y, "perc": perc.offset if perc else None})
            if perc is not None:
                err2.append((perc.offset - car.y) ** 2)

        now = time.perf_counter()
        hz = 0.9 * hz + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        info = dict(
            i=i, n=rec.n, paused=paused, dx=dx, dpsi=dpsi,
            v=rec.v[i], y=car.y, psi=car.psi, kappa=rec.kappa[i], width=width, wheel=car.wheel,
            perc_offset=perc.offset if perc else None,
            perc_rms=float(np.sqrt(np.mean(err2))) if err2 else 0.0,
            departures=stats["departures"], resets=stats["resets"],
            banner=(banner["text"], banner["col"]) if i <= banner["until"] else None,
            proc_ms=proc_ms, hz=hz, auto=mode_auto, oracle=oracle, plan=plan,
        )
        out = hud.draw(virt, img, xs, ys, perc_lanes, perc, (car.y, car.psi, rec.kappa[i], width), hist, info)
        proc_ms = (time.perf_counter() - t0) * 1000  # shown on the next frame (includes HUD drawing)

        if args.headless:
            frames_done += 1
            proc_log.append(proc_ms)
            if args.push_at is not None and frames_done == args.push_at:
                car.y += 0.8
            if args.push_every and frames_done % args.push_every == 0:
                car.y += 0.8 * rng.choice([-1, 1])
            if frames_done >= args.headless:
                break
            i += 1
            continue

        cv2.imshow(win, out)
        wait = max(1, int((DT - (time.perf_counter() - t0)) * 1000))
        key = cv2.waitKeyEx(wait)
        k = key & 0xFF if 0 <= key < 256 else key
        steered = False
        if key in KEY_LEFT or k == ord("a"):
            car.wheel = min(car.wheel + 8, 450); steered = True
        elif key in KEY_RIGHT or k == ord("d"):
            car.wheel = max(car.wheel - 8, -450); steered = True
        elif k == ord("m"):
            mode_auto = not mode_auto
            auto.reset(car.wheel)
            show_banner("자동 조향 (MPC) 작동" if mode_auto else "수동 조향", C["plan"] if mode_auto else C["path"], i)
        elif k == ord("o"):
            oracle = not oracle
            auto.reset(car.wheel)
            show_banner("제어 입력: 오라클 (시뮬 실제값)" if oracle else "제어 입력: 인식 (UFLD)", C["warn"] if oracle else C["muted"], i)
        elif k == ord("s"):
            car.wheel = 0.0
        elif k == ord("p"):
            car.y += 0.8 * np.random.choice([-1, 1])
        elif k == ord("r"):
            reset_to_recording(i)
        elif k in (ord("n"), ord("b")):
            i = int(np.clip(i + (300 if k == ord("n") else -300), 1, rec.n - 1))
            reset_to_recording(i)
            hist.clear()
            continue
        elif k == ord(" "):
            paused = not paused
        elif k == ord("q") or k == 27:
            break
        if steered:
            last_steer_key = time.perf_counter()
            if mode_auto:
                mode_auto = False
                show_banner("운전자 개입 · 자동 조향 해제 (M으로 재작동)", C["warn"], i)
        elif not mode_auto and time.perf_counter() - last_steer_key > 0.25:
            car.wheel *= 0.94  # self-centering, like a real steering rack
        if not paused:
            i += 1

    if args.headless and out is not None:
        cv2.imwrite(args.out, out)
        ms = np.array(proc_log[5:])
        print(f"saved {args.out}  ({frames_done} frames, per-frame {ms.mean():.1f} ms mean, {np.percentile(ms, 95):.1f} ms p95)")
        ay = np.array(metrics["abs_y"])
        mode = "manual" if not mode_auto else ("MPC+oracle" if oracle else "MPC+perception")
        line = (f"[{mode}] |offset| mean {ay.mean():.2f} m, p95 {np.percentile(ay, 95):.2f} m, max {ay.max():.2f} m | "
                f"departures {stats['departures']} | render resets {stats['resets']}")
        if metrics["wheel_rate"]:
            wr = np.abs(metrics["wheel_rate"])
            line += f" | wheel rate p95 {np.percentile(wr, 95):.0f} deg/s | MPC {np.mean(metrics['solve_ms']):.2f} ms"
        print(line)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
