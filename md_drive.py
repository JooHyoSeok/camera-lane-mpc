"""MetaDrive autopilot driven by a camera: dashcam -> UFLD lane detection ->
ground-plane (BEV) lane fit -> Kalman filter -> lateral MPC (+ cruise control),
with the same style of HUD as sim.py.

Control input is camera perception only. The simulator's ground truth (lane
offset, heading) is used solely to grade perception and control on the HUD and
in the printed metrics. --oracle feeds ground truth to the controller instead,
for comparison.

The dashcam mimics CULane's: 800x288 (same aspect as 1640x590), 100 deg
horizontal FOV, 1.34 m above the road, pitched 5 deg down. Unlike the recorded
video, its intrinsics and mounting are known exactly, so the BEV is exact.

Keys (HUD window focused):
  M auto/manual    <-/-> steer, Up/Down throttle/brake (takes over from auto)
  P push 0.8 m sideways    R restart map    N next map    Space pause    Q/Esc quit

Usage (MetaDrive venv; macOS renders cameras only with a window):
    .venv-md/bin/python md_drive.py [--map SCSCSCSC] [--seed 0] [--speed 40] [--traffic 0]
    .venv-md/bin/python md_drive.py --episodes 5 --fast        # evaluate, print metrics
    .venv-md/bin/python md_drive.py --oracle --episodes 5 --fast
    .venv-md/bin/python md_drive.py --episodes 1 --fast --no-hud --push-every 200 --record drive.mp4
"""
import argparse
import time
from collections import deque
from pathlib import Path

import md_mac  # noqa: F401  (before MetaDrive opens a window)
import cv2
import numpy as np
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs import MetaDriveEnv

from bev import CameraParams, GroundCamera, fit_ego_lane, lane_points_ground
from controller import LaneKF, LateralMPC, quiet_fp
from hud_kit import C, Text, blend_poly, dashed, rrect, side_word
from ufld import EGO_LEFT, EGO_RIGHT, MODEL_PATH, UFLD, LaneTracker, lane_polyline

DT = 0.1             # MetaDrive control period (0.02 s physics x 5)
HORIZON = 30         # MPC steps -> 3 s
STEER_RATIO = 15.0   # display only: steering-wheel deg per road-wheel deg
A_LAT = 2.0          # comfort lateral acceleration for curve speed [m/s^2]
PLAN_RGBA = (0.25, 0.88, 0.77, 1.0)

CAM_W, CAM_H, CAM_HFOV = 800, 288, 100.0
CAM_FWD, CAM_UP = 0.9, 0.8      # mount relative to the vehicle origin [m] (origin is ~0.54 m above the road)
CAM_PITCH = -5.0                # deg, negative = looking down
LANE_W_PRIOR = 3.5              # MetaDrive lane width, used only when one boundary is missing
SIM_MODEL = MODEL_PATH.parent / "culane_18_md.onnx"   # UFLD fine-tuned on auto-labeled MetaDrive frames


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def lane_curvature(lane):
    """+ = curving left. MetaDrive: direction -1 is clockwise (a right turn)."""
    return lane.direction / lane.radius if isinstance(lane, CircularLane) else 0.0


def forward(heading):
    return np.array([np.cos(heading), np.sin(heading)])


def right(heading):
    return np.array([np.sin(heading), -np.cos(heading)])


# ======================================================================
# Perception: dashcam + UFLD + BEV
# ======================================================================
class CameraLanes:
    def __init__(self, env, vehicle, model_path=MODEL_PATH):
        self.sensor = env.engine.get_sensor("rgb_camera")
        vfov = np.degrees(2 * np.arctan(np.tan(np.radians(CAM_HFOV / 2)) * CAM_H / CAM_W))
        self.sensor.get_lens().setFov(CAM_HFOV, vfov)  # square pixels
        f = CAM_W / 2 / np.tan(np.radians(CAM_HFOV / 2))
        cy = CAM_H / 2
        height = vehicle.origin.getZ() + CAM_UP
        self.cam = GroundCamera(CameraParams(f=f, cx=CAM_W / 2, cy=cy, vp_u=CAM_W / 2,
                                             vp_v=cy - f * np.tan(np.radians(-CAM_PITCH)), h=height,
                                             width=CAM_W, height=CAM_H))
        self.model = UFLD(model_path)
        self.model_name = Path(model_path).stem
        self.tracker = LaneTracker()

    def reset(self):
        self.tracker = LaneTracker()

    def __call__(self, vehicle):
        img = self.sensor.perceive(to_float=False, new_parent_node=vehicle.origin,
                                   position=(0.0, CAM_FWD, CAM_UP), hpr=(0.0, CAM_PITCH, 0.0))
        img = np.ascontiguousarray(img[..., :3])
        xs, ys = self.model(img)
        xs = self.tracker.update(xs)
        lane = fit_ego_lane(self.cam, xs, ys, lane_width_prior=LANE_W_PRIOR)
        return img, xs, ys, lane, lane_points_ground(self.cam, xs, ys)


# ======================================================================
# Ground truth (grading / oracle only) at the camera point
# ======================================================================
class Truth:
    def __init__(self, env, vehicle, ahead=130.0, step=1.0):
        hd = vehicle.heading_theta
        self.cam_xy = np.asarray(vehicle.position) + CAM_FWD * forward(hd)
        nav = vehicle.navigation
        lane = nav.current_lane
        s0, lat = lane.local_coordinates(self.cam_xy)
        self.lat = float(lat)                                          # + right of lane center
        self.psi = float(wrap(hd - lane.heading_theta_at(s0)))         # car vs lane, + left
        self.width = float(lane.width_at(s0))
        self.kappa_now = lane_curvature(lane)

        graph = env.current_map.road_network.graph
        ck = list(nav.checkpoints)
        seq, total = [(lane, s0)], lane.length - s0
        i = ck.index(lane.index[1]) if lane.index[1] in ck else None
        while total < ahead and i is not None and i + 1 < len(ck):
            lanes = graph[ck[i]][ck[i + 1]]
            nxt = lanes[min(lane.index[2], len(lanes) - 1)]
            seq.append((nxt, 0.0))
            total += nxt.length
            i += 1
        kap, s, acc = [], [], 0.0
        for ln, start in seq:
            k = lane_curvature(ln)
            for lo in np.arange(start, ln.length, step):
                kap.append(k)
                s.append(acc + lo - start)
            acc += ln.length - start
        self.s = np.array(s) if s else np.zeros(1)
        self.kappa_s = np.array(kap) if kap else np.zeros(1)

    def kappa_at(self, dist):
        return np.interp(dist, self.s, self.kappa_s)


# ======================================================================
# Control
# ======================================================================
class Autopilot:
    """Lane state (perceived or oracle) -> Kalman filter -> MPC; curve-aware cruise."""

    def __init__(self, wheelbase, max_steer_deg):
        self.L = wheelbase
        self.max_steer = np.radians(max_steer_deg)
        self.mpc = LateralMPC(wheelbase, horizon=HORIZON, dt=DT, ctrl_dt=DT,
                              q_y=3.0, q_psi=8.0, r=40.0, r_rate=150.0,
                              max_delta=np.radians(min(35.0, max_steer_deg)), max_rate=np.radians(40))
        # UFLD heading is noisy (p95 ~8-10 deg in curves): trusting it at 1 deg made the car weave
        # and leave the road; 3 deg lets the model smooth it out.
        self.kf = LaneKF(std_y=0.20, std_psi=np.radians(3.0), proc_y=0.05, proc_psi=np.radians(0.5))
        self.reset()

    def reset(self, delta=0.0):
        self.delta = delta
        self.kappa = 0.0
        self.kf.reset()
        self.lost = 0

    def __call__(self, v, v_set, perc=None, truth=None):
        """perc: LaneState from the camera, or None. truth: Truth when running as oracle."""
        if truth is not None:
            x0 = np.array([truth.lat, truth.psi])
            kappa = truth.kappa_at(max(v, 1.0) * np.arange(1, HORIZON + 1) * DT)
            k_ahead = np.abs(truth.kappa_at(np.linspace(0, max(30.0, 4 * v), 40))).max()
            self.lost = 0
        else:
            if perc is not None:
                self.kappa = 0.8 * self.kappa + 0.2 * float(np.clip(perc.curvature, -0.03, 0.03))
            self.kf.predict(v, self.delta, self.kappa, DT, self.L)
            if perc is not None:
                self.kf.update(np.array([perc.offset, -perc.heading]))
                self.lost = 0
            else:
                self.lost += 1
            if self.kf.x is None:
                return 0.0, -0.3, None, 0.0, None
            x0, kappa, k_ahead = self.kf.x, self.kappa, abs(self.kappa)
        plan = self.mpc.solve(x0, v, kappa, self.delta)
        self.delta = plan.delta
        v_ref = min(v_set, np.sqrt(A_LAT / max(k_ahead, 1e-4)))
        if self.lost > int(1.0 / DT):          # no lanes for 1 s: slow down
            v_ref = min(v_ref, 15 / 3.6)
        throttle = float(np.clip(0.35 * (v_ref - v), -1, 1))
        steer = float(np.clip(self.delta / self.max_steer, -1, 1))
        return steer, throttle, plan, v_ref, np.broadcast_to(kappa, (HORIZON,)).copy()


# ======================================================================
# HUD
# ======================================================================
class MdHud:
    W, H = 1600, 920
    VIEW = (16, 60, 1184, 426)        # dashcam 800x288 scaled by 1.48
    BEV = (1216, 60, 368, 824)
    CARDS_Y, CARDS_H = 500, 220
    GRAPH = (16, 736, 1184, 148)
    BEV_PPM, BEV_Y0 = 7.0, -18.0

    def __init__(self, cam):
        self.cam = cam
        self.t = Text()
        bx, by, bw, bh = self.BEV
        self.bev_h = bh - 36
        self.half_x = bw / 2 / self.BEV_PPM
        self.y1 = self.BEV_Y0 + self.bev_h / self.BEV_PPM
        self.map_lines = None

    def set_map(self, env):
        """Cache every lane boundary of the map (display only)."""
        lines = []
        for ends in env.current_map.road_network.graph.values():
            for lanes in ends.values():
                for ln in lanes:
                    s = np.linspace(0, ln.length, max(2, int(ln.length / 2) + 1))
                    for side in (-0.5, 0.5):
                        lines.append(np.array([ln.position(si, side * ln.width_at(si)) for si in s]))
        self.map_pts = np.concatenate(lines)
        self.map_splits = np.cumsum([len(l) for l in lines])[:-1]
        self.map_lines = lines

    @staticmethod
    @quiet_fp
    def to_ego(P, pos, heading):
        """World -> ego frame at the vehicle center (X right, Y forward)."""
        d = np.asarray(P, float) - pos
        return np.stack([d @ right(heading), d @ forward(heading)], -1)

    def bev_px(self, XY):
        """Ego-frame meters -> pixels inside the BEV pane (local coordinates)."""
        col = (XY[..., 0] + self.half_x) * self.BEV_PPM
        row = (self.y1 - XY[..., 1]) * self.BEV_PPM
        return np.stack([col, row], -1)

    def card(self, img, x, y, w, h, title):
        rrect(img, x, y, w, h, 14, C["panel"])
        self.t((x + 18, y + 16), title, 13, C["muted"], "medium")

    # ---------------- sections ----------------
    def top_bar(self, img, s):
        t = self.t
        t((18, 14), "METADRIVE", 18, C["text"], "bold", mono=True)
        t((150, 16), "카메라 차선 인식 + MPC", 15, C["muted"], "medium")
        x = 340
        chips = [("자동 조향 · MPC", C["plan"]) if s["auto"] else ("수동 운전", C["path"]),
                 ("입력: 오라클 (정답)", C["warn"]) if s["oracle"] else ("입력: 카메라 인식", C["line"]),
                 (f"맵 #{s['seed']}", C["muted"])]
        if s["paused"]:
            chips.append(("일시정지", C["warn"]))
        for label, col in chips:
            w = int(t.width(label, 13, "semibold")) + 24
            rrect(img, x, 12, w, 26, 13, col, 1)
            t((x + 12, 17), label, 13, col, "semibold")
            x += w + 8
        sec = s["sim_t"]
        t((960, 16), f"{int(sec // 60):02d}:{sec % 60:04.1f}", 15, C["text"], "medium", mono=True, anchor="ma")
        mpc = f"MPC {s['plan'].solve_ms:3.1f} ms · " if s["plan"] is not None else ""
        t((self.W - 18, 17), f"{mpc}루프 {s['loop_ms']:4.0f} ms", 13, C["muted"], anchor="ra")

    def main_view(self, img, cam_img, third, s):
        x, y, w, h = self.VIEW
        sx = w / CAM_W
        view = cv2.resize(cam_img, (w, h), interpolation=cv2.INTER_LINEAR)
        xs, ys = s["xs"], s["ys"]
        lanes = {i: lane_polyline(xs[i], ys, CAM_H) for i in range(len(xs))}
        L, R = lanes.get(EGO_LEFT), lanes.get(EGO_RIGHT)
        if L is not None and R is not None:
            blend_poly(view, (np.vstack([L, R[::-1]]) * sx).astype(np.int32), C["path"], 0.18)
        for i, pts in lanes.items():
            if pts is not None:
                ego = i in (EGO_LEFT, EGO_RIGHT)
                cv2.polylines(view, [(pts * sx).astype(np.int32)], False, C["line"] if ego else C["muted"],
                              4 if ego else 2, cv2.LINE_AA)
        if s["plan"] is not None:
            self.draw_plan_camera(view, s["plan"], sx)
        img[y:y + h, x:x + w] = view
        rrect(img, x, y, w, h, 2, C["edge"], 1)
        self.t((x + w - 14, y + h - 26), f"대시캠 800×288 · 화각 100° · UFLD ({s['model']})", 12, C["text"], anchor="ra")

        if third is not None:   # picture-in-picture: MetaDrive's 3rd-person window
            iw, ih = 300, 169
            ix, iy = x + 14, y + 14
            img[iy:iy + ih, ix:ix + iw] = cv2.resize(third, (iw, ih), interpolation=cv2.INTER_AREA)
            cv2.rectangle(img, (ix, iy), (ix + iw, iy + ih), C["edge"], 1)
            rrect(img, ix, iy + ih + 6, iw, 24, 6, C["panel"])
            self.t((ix + 10, iy + ih + 10), "3인칭 (MetaDrive 창)", 12, C["muted"], "semibold")
        if s["banner"]:
            text, col = s["banner"]
            bw = int(self.t.width(text, 16, "bold")) + 40
            bx = x + (w - bw) // 2
            rrect(img, bx, y + 16, bw, 36, 18, col)
            self.t((bx + bw // 2, y + 25), text, 16, C["bg"], "bold", anchor="ma")

    def draw_plan_camera(self, view, plan, sx, half_w=0.45):
        """Project the MPC path (camera-point ground frame) into the dashcam view."""
        P = plan.path
        keep = P[:, 1] > 2.5
        if keep.sum() < 2:
            return
        P = P[keep]
        d = np.gradient(P, axis=0)
        n = np.stack([d[:, 1], -d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)

        def proj(Q):
            u, v = self.cam.ground_to_image(Q[:, 0], Q[:, 1])
            return np.stack([u, v], 1) * sx

        band = np.vstack([proj(P - n * half_w), proj(P + n * half_w)[::-1]]).astype(np.int32)
        blend_poly(view, band, C["plan"], 0.45)
        cv2.polylines(view, [proj(P).astype(np.int32)], False, C["plan"], 2, cv2.LINE_AA)
        times = plan.times[keep]
        for tm in (1.0, 2.0, 3.0):
            k = int(np.argmin(np.abs(times - tm)))
            if abs(times[k] - tm) < 0.06:
                c = tuple(proj(P[k:k + 1])[0].astype(int))
                cv2.circle(view, c, 5, C["bg"], -1, cv2.LINE_AA)
                cv2.circle(view, c, 5, C["plan"], 2, cv2.LINE_AA)

    def bev_panel(self, img, s):
        bx, by, bw, bh = self.BEV
        rrect(img, bx, by, bw, bh, 14, C["panel"])
        self.t((bx + 16, by + 12), "BEV · 위에서 본 시점", 13, C["muted"], "medium")
        top = by + 36
        pos, hd = s["pos"], s["heading"]
        cam_shift = np.array([0.0, CAM_FWD])  # camera-frame items -> vehicle-center frame
        pane = np.empty((self.bev_h, bw, 3), np.uint8)
        pane[:] = C["bg"]
        margin = 60

        def visible(px):
            return ((px[:, 0] > -margin).any() and (px[:, 0] < bw + margin).any()
                    and (px[:, 1] > -margin).any() and (px[:, 1] < self.bev_h + margin).any())

        # map (ground truth, display only)
        if self.map_lines is not None:
            px = self.bev_px(self.to_ego(self.map_pts, pos, hd))
            segs = [seg.astype(np.int32) for seg in np.split(px, self.map_splits) if visible(seg)]
            cv2.polylines(pane, segs, False, C["dim"], 1, cv2.LINE_AA)
        # perceived lane boundaries (camera)
        for i, pts in enumerate(s["perc_lanes"]):
            if pts is None or len(pts) < 3:
                continue
            c = np.polyfit(pts[:, 1], pts[:, 0], 2 if len(pts) >= 6 else 1)
            Yl = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 30)
            P = self.bev_px(np.stack([np.polyval(c, Yl), Yl], 1) + cam_shift)
            ego = i in (EGO_LEFT, EGO_RIGHT)
            cv2.polylines(pane, [P.astype(np.int32)], False, C["line"] if ego else C["muted"], 3 if ego else 2,
                          cv2.LINE_AA)
        for o in s["traffic"]:
            self.draw_box(pane, o["pos"], o["heading"], o["length"], o["width"], pos, hd, C["muted"])
        plan = s["plan"]
        marks = []
        if plan is not None:
            P = self.bev_px(plan.path + cam_shift)
            cv2.polylines(pane, [P.astype(np.int32)], False, C["plan"], 3, cv2.LINE_AA)
            for tm in (1, 2, 3):
                k = int(round(tm / DT))
                if k < len(P):
                    c = tuple(P[k].astype(int))
                    cv2.circle(pane, c, 6, C["panel"], -1, cv2.LINE_AA)
                    cv2.circle(pane, c, 6, C["plan"], 2, cv2.LINE_AA)
                    marks.append((tm, c))
        L, Wd = s["ego_size"]
        self.draw_box(pane, pos, hd, L, Wd, pos, hd, C["car"], filled=True)
        cc = tuple(self.bev_px(cam_shift).astype(int))
        cv2.circle(pane, cc, 4, C["line"], -1, cv2.LINE_AA)   # camera
        img[top:top + self.bev_h, bx:bx + bw] = pane

        for tm, (cx, cy) in marks:
            if 0 <= cy < self.bev_h - 20 and 0 <= cx < bw - 30:
                self.t((bx + cx + 12, top + cy - 8), f"{tm}s", 11, C["plan"], "semibold")
        for d in range(0, int(self.y1) + 1, 20):
            r = top + self.bev_px(np.array([0.0, d]))[1]
            if r > self.bev_h + top - 40:
                continue
            cv2.line(img, (bx, int(r)), (bx + 10, int(r)), C["muted"], 1, cv2.LINE_AA)
            rrect(img, bx + 13, int(r) - 10, 44, 20, 6, C["panel"])
            self.t((bx + 35, int(r) - 7), f"{d} m", 11, C["muted"], mono=True, anchor="ma")
        ly = by + bh - 26
        rrect(img, bx, ly - 10, bw, 36, 10, C["panel"])
        cv2.line(img, (bx + 14, ly + 7), (bx + 32, ly + 7), C["line"], 3, cv2.LINE_AA)
        self.t((bx + 37, ly), "인식", 12, C["muted"])
        cv2.line(img, (bx + 78, ly + 7), (bx + 96, ly + 7), C["plan"], 3, cv2.LINE_AA)
        self.t((bx + 101, ly), "MPC 예측", 12, C["muted"])
        cv2.line(img, (bx + 166, ly + 7), (bx + 184, ly + 7), C["dim"], 1, cv2.LINE_AA)
        self.t((bx + 189, ly), "지도(정답)", 12, C["muted"])
        rrect(img, bx + 262, ly + 1, 16, 12, 3, C["muted"])
        self.t((bx + 283, ly), "다른 차", 12, C["muted"])

    def draw_box(self, pane, p, heading, length, width, pos, hd, color, filled=False):
        f, l = forward(heading), -right(heading)
        corners = [p + f * length / 2 + l * width / 2, p + f * length / 2 - l * width / 2,
                   p - f * length / 2 - l * width / 2, p - f * length / 2 + l * width / 2]
        poly = self.bev_px(self.to_ego(np.array(corners), pos, hd)).astype(np.int32)
        h, w = pane.shape[:2]
        if not ((poly[:, 0] > 0).any() and (poly[:, 0] < w).any() and (poly[:, 1] > 0).any() and (poly[:, 1] < h).any()):
            return
        if filled:
            cv2.fillPoly(pane, [poly], color, cv2.LINE_AA)
        else:
            blend_poly(pane, poly, color, 0.55)
            cv2.polylines(pane, [poly], True, color, 1, cv2.LINE_AA)

    def cards(self, img, s):
        x0, y0, h = 16, self.CARDS_Y, self.CARDS_H
        w = (self.VIEW[2] - 3 * 16) // 4
        xs = [x0 + k * (w + 16) for k in range(4)]
        t = self.t
        truth, perc = s["truth"], s["perc"]

        # 1) speed
        x = xs[0]
        self.card(img, x, y0, w, h, "속도")
        kmh = s["v"] * 3.6
        t((x + 18, y0 + 58), f"{kmh:.0f}", 64, C["text"], "semibold", mono=True)
        t((x + 22 + t.width(f"{kmh:.0f}", 64, "semibold", True), y0 + 98), "km/h", 15, C["muted"], "medium")
        if s["auto"] and s["v_ref"] is not None:
            t((x + 18, y0 + 150), f"목표 {s['v_ref'] * 3.6:.0f} km/h", 13, C["plan"], "semibold")
            reason = "차선 놓침 · 감속" if s["lost"] else ("커브 감속" if s["v_ref"] < s["v_set"] - 0.3 else "설정 속도")
            t((x + 18, y0 + 172), reason, 12, C["warn"] if s["lost"] else C["dim"])

        # 2) position in lane: perceived (controller input) vs ground truth
        x = xs[1]
        width = truth.width
        lat_true = truth.lat
        margin = width / 2 - abs(lat_true) - s["ego_size"][1] / 2
        col = C["ok"] if margin > 0.3 else C["warn"] if margin > 0 else C["bad"]
        self.card(img, x, y0, w, h, "차선 내 위치")
        if perc is not None:
            t.pair((x + 18, y0 + 76), side_word(perc.offset), f"{abs(perc.offset):.2f} m", 30, C["line"])
        else:
            t((x + 18, y0 + 50), "인식 없음", 26, C["warn"], "semibold")
        t((x + 18, y0 + 84), f"인식 ↑ · 정답 {side_word(lat_true)} {abs(lat_true):.2f} m", 12, C["muted"])
        lx0, lx1, ly = x + 22, x + w - 22, y0 + 128
        ppm = (lx1 - lx0) / (width + 1.0)
        cxp = (lx0 + lx1) / 2
        for sgn in (-1, 1):
            px = int(cxp + sgn * width / 2 * ppm)
            cv2.line(img, (px, ly), (px, ly + 62), C["text"], 3, cv2.LINE_AA)
        dashed(img, (cxp, ly), (cxp, ly + 62), C["dim"], 1, 6, 6)
        cw = s["ego_size"][1]
        car_c = cxp + lat_true * ppm
        rrect(img, int(car_c - cw / 2 * ppm), ly + 10, int(cw * ppm), 42, 8, col)
        rrect(img, int(car_c - cw / 2 * ppm) + 6, ly + 16, int(cw * ppm) - 12, 10, 3, C["panel"])
        if perc is not None:
            pc = int(cxp + perc.offset * ppm)
            cv2.line(img, (pc, ly - 4), (pc, ly + 66), C["line"], 2, cv2.LINE_AA)

        # 3) heading & curvature
        x = xs[2]
        self.card(img, x, y0, w, h, "진행 방향 · 도로 곡률")
        psi_deg = np.degrees(-perc.heading) if perc is not None else np.degrees(truth.psi)
        t.pair((x + 18, y0 + 76), side_word(-psi_deg, 0.1), f"{abs(psi_deg):.1f}°", 30,
               C["line"] if perc is not None else C["text"])
        k = s["kappa_used"]
        kt = truth.kappa_now
        t((x + 18, y0 + 84), ("직선" if abs(k) < 1e-4 else f"{'좌' if k > 0 else '우'}커브 R {1 / abs(k):.0f} m")
          + " · 정답 " + ("직선" if abs(kt) < 1e-4 else f"R {1 / abs(kt):.0f} m"), 12, C["muted"])
        ax, ay = x + w // 2, y0 + 168
        for sgn in (-1, 1):
            cv2.line(img, (ax + sgn * 46, ay - 44), (ax + sgn * 46, ay + 40), C["dim"], 2, cv2.LINE_AA)
        a = -np.radians(psi_deg) * 4
        tip = (int(ax + 40 * np.sin(a)), int(ay - 40 * np.cos(a)))
        tail = (int(ax - 30 * np.sin(a)), int(ay + 30 * np.cos(a)))
        cv2.arrowedLine(img, tail, tip, C["line"], 4, cv2.LINE_AA, tipLength=0.3)
        t((x + w - 18, y0 + h - 34), "각도 ×4 확대", 11, C["dim"], anchor="ra")

        # 4) steering
        x = xs[3]
        self.card(img, x, y0, w, h, "조향")
        road = np.degrees(s["steer"] * s["max_steer"])
        wheel = road * STEER_RATIO
        t.pair((x + 18, y0 + 76), side_word(-wheel, 0.5), f"{abs(wheel):.0f}°", 30, C["text"])
        t((x + 18, y0 + 84), f"바퀴 {road:+.1f}° · 핸들 {STEER_RATIO:.0f}:1 (표시용)", 12, C["muted"])
        cx, cy, r = x + 68, y0 + 160, 40
        cv2.circle(img, (cx, cy), r, C["text"], 7, cv2.LINE_AA)
        a = -np.radians(wheel)
        for base in (0.0, np.pi, np.pi / 2):
            ang = base + a
            cv2.line(img, (cx, cy), (int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))), C["text"], 5, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 11, C["panel"], -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 11, C["text"], 3, cv2.LINE_AA)
        topp = (int(cx + r * np.cos(-np.pi / 2 + a)), int(cy + r * np.sin(-np.pi / 2 + a)))
        cv2.circle(img, topp, 6, C["plan"] if s["auto"] else C["path"], -1, cv2.LINE_AA)
        gx0, gx1, gy0, gy1 = x + 126, x + w - 18, y0 + 118, y0 + 200
        rrect(img, gx0, gy0, gx1 - gx0, gy1 - gy0, 8, C["bg"])
        gm = (gy0 + gy1) // 2
        cv2.line(img, (gx0 + 6, gm), (gx1 - 6, gm), C["dim"], 1, cv2.LINE_AA)
        plan = s["plan"]
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

    def graph(self, img, hist, s):
        x, y, w, h = self.GRAPH
        self.card(img, x, y, w, h, "차선 중앙 대비 위치 · 최근 10초")
        px0, px1 = x + 60, x + w - 250
        p_now = px0 + (px1 - px0) * 10 / 13
        py0, py1 = y + 40, y + h - 16
        ymax = 2.2

        def to_py(v):
            return py0 + (py1 - py0) * (0.5 - v / (2 * ymax))

        cv2.line(img, (int(p_now), py0 - 6), (int(p_now), py1), C["dim"], 1, cv2.LINE_AA)
        self.t((int(p_now), py0 - 20), "지금", 11, C["muted"], anchor="ma")
        half = s["truth"].width / 2
        cw = s["ego_size"][1]
        for v, col in ((half, C["bad"]), (-half, C["bad"]), (half - cw / 2, C["warn"]), (-(half - cw / 2), C["warn"])):
            dashed(img, (px0, to_py(v)), (px1, to_py(v)), col, 1, 5, 5)
        cv2.line(img, (px0, int(to_py(0))), (px1, int(to_py(0))), C["dim"], 1, cv2.LINE_AA)
        for v in (-2, 0, 2):
            self.t((px0 - 10, int(to_py(v)) - 7), f"{v:+d}" if v else "0", 11, C["dim"], mono=True, anchor="ra")
        self.t((x + 18, py1 - 12), "좌", 11, C["dim"])
        self.t((x + 18, py0 - 2), "우", 11, C["dim"])
        n = hist.maxlen
        for key, col in (("perc", C["line"]), ("true", C["text"])):
            vals = [hv[key] for hv in hist]
            pts = [(p_now - (len(vals) - 1 - k) * (p_now - px0) / (n - 1), to_py(np.clip(-v, -ymax, ymax)))
                   for k, v in enumerate(vals) if v is not None]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
        plan = s["plan"]
        if plan is not None:
            fx = p_now + plan.times / 3.0 * (px1 - p_now)
            fy = to_py(np.clip(-plan.y_pred, -ymax, ymax))
            fp = np.stack([fx, fy], 1).astype(np.int32)
            for k in range(0, len(fp) - 1, 2):
                cv2.line(img, tuple(fp[k]), tuple(fp[k + 1]), C["plan"], 2, cv2.LINE_AA)
        sx = px1 + 22
        for k, (label, col) in enumerate((("정답", C["text"]), ("인식", C["line"]), ("예측", C["plan"]))):
            cv2.line(img, (sx + k * 66, y + 22), (sx + k * 66 + 16, y + 22), col, 2, cv2.LINE_AA)
            self.t((sx + k * 66 + 21, y + 15), label, 12, C["muted"])
        st = s["stats"]
        rows = [("인식 오차 RMS", f"{st['perc_rms']:.2f} m", C["text"]),
                ("도로 이탈 · 충돌", f"{st['out_of_road']} · {st['crash']}회",
                 C["bad"] if st["out_of_road"] or st["crash"] else C["text"]),
                ("완주 · 주행 거리", f"{st['arrive']}개 · {st['dist'] / 1000:.2f} km", C["text"])]
        for k, (label, val, col) in enumerate(rows):
            yy = y + 46 + k * 30
            self.t((sx, yy), label, 13, C["muted"])
            self.t((x + w - 18, yy), val, 14, col, "semibold", mono=True, anchor="ra")

    def help_line(self):
        self.t((18, self.H - 28),
               "HUD 창에서:  M 자동/수동   ←/→ 조향   ↑/↓ 가속/감속 (누르면 자동 해제)   P 옆으로 밀기   "
               "R 다시 시작   N 다음 맵   Space 일시정지   Q 종료",
               13, C["dim"])

    def draw(self, cam_img, third, hist, s):
        img = np.empty((self.H, self.W, 3), np.uint8)
        img[:] = C["bg"]
        self.top_bar(img, s)
        self.main_view(img, cam_img, third, s)
        self.bev_panel(img, s)
        self.cards(img, s)
        self.graph(img, hist, s)
        self.help_line()
        return self.t.render(img)


# ======================================================================
# Main loop
# ======================================================================
def grab_3d(env):
    tex = env.engine.win.getScreenshot()
    if tex is None:
        return None
    rgb = np.frombuffer(tex.getRamImageAs("RGB"), np.uint8).reshape(tex.getYSize(), tex.getXSize(), 3)[::-1]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class PlanDrawer:
    """MPC path as spheres in the 3D scene (GL line width is fixed at 1 px on macOS)."""

    def __init__(self, env):
        from metadrive.constants import CamMask
        self.small = env.engine.make_point_drawer(scale=1.4)
        self.big = env.engine.make_point_drawer(scale=3.0)
        for d in (self.small, self.big):   # display only: keep it out of the dashcam the lane model sees
            d.hide(CamMask.RgbCam)

    def update(self, plan, origin_xy, heading):
        self.small.reset()
        self.big.reset()
        if plan is None:
            return
        world = origin_xy + np.outer(plan.path[:, 1], forward(heading)) + np.outer(plan.path[:, 0], right(heading))
        pts = [(p[0], p[1], 0.25) for p in world[1:]]
        self.small.draw_points(pts, [PLAN_RGBA] * len(pts))
        marks = [pts[int(round(tm / DT)) - 1] for tm in (1, 2, 3) if int(round(tm / DT)) <= len(pts)]
        self.big.draw_points(marks, [PLAN_RGBA] * len(marks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="SCSCSCSC", help="block string (S straight, C curve, ...) or block count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traffic", type=float, default=0.0, help="traffic density (no ACC yet: keep low)")
    ap.add_argument("--speed", type=float, default=40.0, help="cruise speed [km/h]")
    ap.add_argument("--oracle", action="store_true", help="feed ground truth to the controller (comparison)")
    ap.add_argument("--model", default=None,
                    help="UFLD ONNX (default: the MetaDrive fine-tuned model if present, else the CULane original)")
    ap.add_argument("--hud-scale", type=float, default=0.8)
    ap.add_argument("--fast", action="store_true", help="run as fast as possible instead of real time")
    ap.add_argument("--episodes", type=int, default=0, help="stop after N episodes and print metrics")
    ap.add_argument("--push-every", type=int, default=0, help="push 0.8 m sideways every N steps")
    ap.add_argument("--steps", type=int, default=0, help="stop after N steps (testing)")
    ap.add_argument("--hud-out", default=None, help="save the last HUD frame here")
    ap.add_argument("--no-hud", action="store_true", help="no HUD window (faster evaluation)")
    ap.add_argument("--record", default=None, help="write the HUD to this .mp4 (H.264, real-time 10 fps)")
    args = ap.parse_args()

    env = MetaDriveEnv(dict(
        use_render=True, manual_control=False,
        map=int(args.map) if args.map.isdigit() else args.map,
        traffic_density=args.traffic, num_scenarios=1000, start_seed=0,
        window_size=(640, 360), show_interface=False, show_logo=False, show_fps=False,
        sensors=dict(rgb_camera=(RGBCamera, CAM_W, CAM_H)),
        on_continuous_line_done=False, crash_vehicle_done=False, out_of_road_done=True,
        vehicle_config=dict(show_navi_mark=False),
        log_level=50,
    ))
    seed = args.seed
    env.reset(seed=seed)
    env.engine.force_fps.disable()  # we pace the whole loop (sim + perception + HUD) ourselves
    v = env.agent
    pilot = Autopilot(v.FRONT_WHEELBASE + v.REAR_WHEELBASE, v.max_steering)
    model_path = Path(args.model) if args.model else (SIM_MODEL if SIM_MODEL.exists() else MODEL_PATH)
    camera = CameraLanes(env, v, model_path)
    print(f"lane model: {model_path.name}")
    hud = MdHud(camera.cam)
    hud.set_map(env)
    drawer = PlanDrawer(env)
    rng = np.random.default_rng(0)
    v_set = args.speed / 3.6

    screen = None
    if not args.no_hud:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((int(MdHud.W * args.hud_scale), int(MdHud.H * args.hud_scale)))
        pygame.display.set_caption("MetaDrive · camera + MPC HUD")

    writer = None
    if args.record:
        rec_size = (int(MdHud.W * args.hud_scale) // 2 * 2, int(MdHud.H * args.hud_scale) // 2 * 2)
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"avc1"), round(1 / DT), rec_size)

    hist = deque(maxlen=int(10 / DT))
    err2 = deque(maxlen=int(10 / DT))
    stats = dict(out_of_road=0, crash=0, arrive=0, dist=0.0, perc_rms=0.0)
    metrics = dict(abs_lat=[], perc_err=[], lost=0, steps=0, solve_ms=[], episodes=[])
    banner = {"text": None, "col": None, "until": -1}
    auto, paused = True, False
    manual_steer = 0.0
    step, sim_t, loop_ms = 0, 0.0, 0.0
    ep_start = 0
    was_crash = False
    out = None

    def show_banner(text, col, secs=2.0):
        banner.update(text=text, col=col, until=step + int(secs / DT))

    def new_episode(next_seed):
        nonlocal seed, ep_start
        seed = next_seed
        env.reset(seed=seed)
        env.engine.force_fps.disable()
        hud.set_map(env)
        pilot.reset()
        camera.reset()
        hist.clear()
        ep_start = step

    def push():
        side = rng.choice([-1, 1])
        v.set_position(np.asarray(v.position) + side * 0.8 * right(v.heading_theta))
        show_banner(f"{'오른' if side > 0 else '왼'}쪽으로 0.8 m 밀기", C["warn"], 1.0)

    running = True
    while running:
        t0 = time.perf_counter()

        # ---- keyboard (HUD window) ----
        k_l = k_r = k_u = k_d = False
        if screen is not None:
            import pygame
            restarted = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif ev.key == pygame.K_m:
                        auto = not auto
                        pilot.reset(manual_steer * pilot.max_steer)
                        show_banner("자동 조향 (MPC) 작동" if auto else "수동 운전", C["plan"] if auto else C["path"])
                    elif ev.key == pygame.K_p:
                        push()
                    elif ev.key == pygame.K_r:
                        new_episode(seed)
                        restarted = True
                    elif ev.key == pygame.K_n:
                        new_episode(seed + 1)
                        show_banner(f"맵 #{seed}", C["path"])
                        restarted = True
                    elif ev.key == pygame.K_SPACE:
                        paused = not paused
            if restarted or not running:
                continue
            keys = pygame.key.get_pressed()
            k_l, k_r, k_u, k_d = keys[pygame.K_LEFT], keys[pygame.K_RIGHT], keys[pygame.K_UP], keys[pygame.K_DOWN]
            if auto and (k_l or k_r or k_u or k_d):
                auto = False
                manual_steer = pilot.delta / pilot.max_steer
                show_banner("운전자 개입 · 자동 해제 (M으로 재작동)", C["warn"])
        if args.push_every and step > ep_start and (step - ep_start) % args.push_every == 0:
            push()

        # ---- perceive ----
        cam_img, xs, ys, perc, perc_lanes = camera(v)
        truth = Truth(env, v)

        # ---- control ----
        plan = v_ref = kappa_used = None
        if auto:
            steer, throttle, plan, v_ref, kap = pilot(v.speed, v_set, perc=perc,
                                                      truth=truth if args.oracle else None)
            kappa_used = float(kap[0]) if kap is not None else 0.0
            if plan is not None:
                metrics["solve_ms"].append(plan.solve_ms)
        else:
            manual_steer = float(np.clip(manual_steer + 0.06 * (k_l - k_r), -1, 1)) if (k_l or k_r) \
                else manual_steer * 0.7
            steer = manual_steer
            throttle = 0.6 if k_u else (-0.6 if k_d else 0.0)
            kappa_used = perc.curvature if perc is not None else 0.0

        # ---- step ----
        if not paused:
            pos_before = np.asarray(v.position).copy()
            _, _, term, trunc, info = env.step([steer, throttle])
            step += 1
            sim_t += DT
            stats["dist"] += float(np.linalg.norm(np.asarray(v.position) - pos_before))
            hist.append({"true": truth.lat, "perc": perc.offset if perc is not None else None})
            metrics["abs_lat"].append(abs(truth.lat))
            metrics["steps"] += 1
            if perc is None:
                metrics["lost"] += 1
            else:
                e = perc.offset - truth.lat
                err2.append(e * e)
                metrics["perc_err"].append(abs(e))
            stats["perc_rms"] = float(np.sqrt(np.mean(err2))) if err2 else 0.0
            crash = bool(info.get("crash_vehicle") or info.get("crash_object"))
            if crash and not was_crash:
                stats["crash"] += 1
                show_banner("충돌", C["bad"])
            was_crash = crash
            if term or trunc:
                outcome = ("arrive" if info.get("arrive_dest") else
                           "out_of_road" if info.get("out_of_road") else "other")
                if outcome != "other":
                    stats[outcome] += 1
                metrics["episodes"].append((seed, outcome, step - ep_start))
                if args.episodes and len(metrics["episodes"]) >= args.episodes:
                    running = False
                else:
                    show_banner("목적지 도착 · 다음 맵" if outcome == "arrive" else "도로 이탈 · 다음 맵",
                                C["ok"] if outcome == "arrive" else C["bad"])
                    new_episode(seed + 1)
                    continue

        # ---- draw ----
        drawer.update(plan, truth.cam_xy, v.heading_theta)
        if screen is not None or args.hud_out or writer is not None:
            traffic = [dict(pos=np.asarray(o.position), heading=o.heading_theta, length=o.LENGTH, width=o.WIDTH)
                       for o in env.engine.traffic_manager.traffic_vehicles]
            s = dict(
                auto=auto, oracle=args.oracle, paused=paused, seed=seed, sim_t=sim_t, loop_ms=loop_ms, plan=plan,
                banner=(banner["text"], banner["col"]) if step <= banner["until"] else None,
                pos=np.asarray(v.position), heading=v.heading_theta, ego_size=(v.LENGTH, v.WIDTH),
                traffic=traffic, v=v.speed, v_ref=v_ref, v_set=v_set, lost=pilot.lost > int(1.0 / DT),
                truth=truth, perc=perc, xs=xs, ys=ys, perc_lanes=perc_lanes, kappa_used=kappa_used or 0.0,
                steer=steer, max_steer=pilot.max_steer, stats=stats, model=camera.model_name,
            )
            out = hud.draw(cam_img, grab_3d(env), hist, s)
            if writer is not None and not paused:
                writer.write(cv2.resize(out, rec_size, interpolation=cv2.INTER_AREA))
            if screen is not None:
                import pygame
                rgb = cv2.cvtColor(cv2.resize(out, screen.get_size(), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                screen.blit(pygame.image.frombuffer(rgb.tobytes(), screen.get_size(), "RGB"), (0, 0))
                pygame.display.flip()
        loop_ms = (time.perf_counter() - t0) * 1000
        if not args.fast:
            time.sleep(max(0.0, DT - (time.perf_counter() - t0)))
        if args.steps and step >= args.steps:
            running = False

    if writer is not None:
        writer.release()
        print(f"saved {args.record}")
    if args.hud_out and out is not None:
        cv2.imwrite(args.hud_out, out)
        print(f"saved {args.hud_out}")
    if metrics["abs_lat"]:
        al = np.array(metrics["abs_lat"])
        pe = np.array(metrics["perc_err"]) if metrics["perc_err"] else np.zeros(1)
        eps = metrics["episodes"]
        mode = "oracle" if args.oracle else f"camera:{camera.model_name}"
        print(f"[{mode}] steps {step} | |lateral| mean {al.mean():.3f} m, p95 {np.percentile(al, 95):.3f} m, "
              f"max {al.max():.2f} m | perception |err| mean {pe.mean():.3f} m, p95 {np.percentile(pe, 95):.3f} m, "
              f"lanes lost {metrics['lost'] / max(metrics['steps'], 1):.1%} of steps")
        print(f"[{mode}] episodes {len(eps)}: arrive {sum(e[1] == 'arrive' for e in eps)}, "
              f"out_of_road {sum(e[1] == 'out_of_road' for e in eps)}, crashes {stats['crash']}, "
              f"distance {stats['dist'] / 1000:.2f} km | MPC {np.mean(metrics['solve_ms'] or [0]):.2f} ms")
    env.close()


if __name__ == "__main__":
    main()
