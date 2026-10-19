"""Flat-ground camera model: image <-> ground plane, bird's-eye view (BEV),
virtual-viewpoint re-rendering, and lane geometry (offset / heading / curvature).

Vehicle frame (meters): X to the right, Y forward, Z up; ground is Z = 0 and
the camera sits at (0, 0, h). Image frame: OpenCV pixels (u right, v down).

The road is a plane, so moving the camera sideways or yawing it maps road
pixels by an exact homography. That is what lets a recorded video act as a
closed-loop simulator for lateral control.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

# numpy 2.0 + macOS Accelerate raises spurious FP warnings in matmul (see controller.py).
quiet_fp = np.errstate(divide="ignore", over="ignore", invalid="ignore")

from ufld import EGO_LEFT, EGO_RIGHT


@dataclass
class CameraParams:
    f: float          # focal length [px]
    cx: float         # principal point [px]
    cy: float
    vp_u: float       # vanishing point of the road direction [px]
    vp_v: float
    h: float          # camera height above the road [m]
    width: int = 1640
    height: int = 590

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class GroundCamera:
    def __init__(self, p: CameraParams):
        self.p = p
        self.K = np.array([[p.f, 0, p.cx], [0, p.f, p.cy], [0, 0, 1]], np.float64)
        # Camera orientation from the vanishing point (no roll assumed).
        fwd = np.array([(p.vp_u - p.cx) / p.f, (p.vp_v - p.cy) / p.f, 1.0])
        fwd /= np.linalg.norm(fwd)
        up = np.array([0.0, -1.0, 0.0])
        up -= up.dot(fwd) * fwd
        up /= np.linalg.norm(up)
        right = np.cross(fwd, up)
        self.R = np.stack([right, fwd, up], axis=1)  # world (X, Y, Z) -> camera
        # Ground (X, Y, 1) -> image homography.
        self.H = self.K @ np.stack([self.R[:, 0], self.R[:, 1], -p.h * self.R[:, 2]], axis=1)
        self.H_inv = np.linalg.inv(self.H)
        self.horizon_v = p.vp_v

    # ---------- point mapping ----------
    @quiet_fp
    def image_to_ground(self, u, v):
        pts = np.stack([u, v, np.ones_like(u)], axis=0)
        g = self.H_inv @ pts
        return g[0] / g[2], g[1] / g[2]

    @quiet_fp
    def ground_to_image(self, X, Y):
        pts = np.stack([X, Y, np.ones_like(X)], axis=0)
        q = self.H @ pts
        return q[0] / q[2], q[1] / q[2]

    # ---------- bird's-eye view ----------
    def bev_matrix(self, x_range, y_range, ppm):
        """BEV pixel (col, row) -> ground (X, Y). Row 0 is the far edge."""
        x0, _ = x_range
        _, y1 = y_range
        return np.array([[1 / ppm, 0, x0], [0, -1 / ppm, y1], [0, 0, 1]], np.float64)

    def warp_bev(self, img, x_range=(-10, 10), y_range=(0, 50), ppm=10):
        A = self.bev_matrix(x_range, y_range, ppm)
        size = (int((x_range[1] - x_range[0]) * ppm), int((y_range[1] - y_range[0]) * ppm))
        return cv2.warpPerspective(img, self.H @ A, size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    # ---------- virtual viewpoint ----------
    @staticmethod
    def pose_transform(dx, dpsi):
        """Ground coords in the recorded vehicle frame -> coords in a vehicle moved
        dx meters to the right and yawed dpsi radians to the left (CCW)."""
        c, s = np.cos(dpsi), np.sin(dpsi)
        Rz_inv = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        T = np.array([[1, 0, -dx], [0, 1, 0], [0, 0, 1]])
        return Rz_inv @ T

    def render_virtual(self, img, dx, dpsi, static_mask=None):
        """Re-render the recorded frame as seen from a camera displaced by
        (dx right, dpsi yaw-left). Road pixels are exact; things above the
        horizon get a rotation-only warp (correct at infinity). Pixels in
        static_mask (hood, lens border) belong to the car and are kept."""
        w, h = self.p.width, self.p.height
        flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        s0 = int(self.horizon_v + 4)   # ground warp from this row down
        s1 = s0 + 12                   # sky warp down to this row; blend in between

        def rows_from(r):  # offset so a warp can fill only rows r.. of the output
            return np.array([[1, 0, 0], [0, 1, r], [0, 0, 1]], np.float64)

        # Ground part: virtual pixel -> virtual ground -> recorded ground -> recorded pixel.
        M_ground = self.H @ np.linalg.inv(self.pose_transform(dx, dpsi)) @ self.H_inv
        ground = cv2.warpPerspective(img, M_ground @ rows_from(s0), (w, h - s0), flags=flags,
                                     borderMode=cv2.BORDER_CONSTANT)
        # Sky part: pure rotation about the world up axis (exact at infinity).
        Rc, _ = cv2.Rodrigues(self.R[:, 2] * dpsi)
        M_sky = self.K @ Rc @ np.linalg.inv(self.K)  # virtual ray -> recorded ray
        sky = cv2.warpPerspective(img, M_sky, (w, s1), flags=flags, borderMode=cv2.BORDER_REPLICATE)

        out = np.empty_like(img)
        out[:s0] = sky[:s0]
        out[s1:] = ground[s1 - s0:]
        alpha = np.linspace(0, 1, s1 - s0, dtype=np.float32)[:, None, None]
        out[s0:s1] = (ground[:s1 - s0] * alpha + sky[s0:s1] * (1 - alpha)).astype(np.uint8)
        if static_mask is not None:
            if static_mask.dtype == bool:
                static_mask = static_mask.astype(np.uint8) * 255
            cv2.copyTo(img, static_mask, out)
        return out


# ---------- lane geometry ----------
@dataclass
class LaneState:
    offset: float      # vehicle position relative to lane center [m], + = right of center
    heading: float     # lane direction relative to vehicle heading [rad], + = lane turns left
    curvature: float   # [1/m], + = curving left
    width: float       # lane width at the vehicle [m]
    left: np.ndarray   # polynomial X(Y) of the left boundary (np.polyfit order: a, b, c)
    right: np.ndarray
    center: np.ndarray


def lane_points_ground(cam, xs, ys, y_min=3.0, y_max=60.0):
    """UFLD output -> per-lane ground points [(X, Y)] within a trusted range."""
    lanes = []
    for row in xs:
        ok = ~np.isnan(row) & (ys > cam.horizon_v + 12)
        if ok.sum() < 3:
            lanes.append(None)
            continue
        X, Y = cam.image_to_ground(row[ok], ys[ok])
        keep = (Y > y_min) & (Y < y_max)
        lanes.append(np.stack([X[keep], Y[keep]], 1) if keep.sum() >= 3 else None)
    return lanes


def fit_ego_lane(cam, xs, ys, lane_width_prior=3.75):
    """Fit X = a*Y^2 + b*Y + c to the ego boundaries; return LaneState or None.
    With one boundary missing, the other is shifted by the width prior."""
    lanes = lane_points_ground(cam, xs, ys)
    L, R = lanes[EGO_LEFT], lanes[EGO_RIGHT]

    def fit(pts):
        deg = 2 if len(pts) >= 6 else 1
        c = np.polyfit(pts[:, 1], pts[:, 0], deg, w=1.0 / (1.0 + 0.05 * pts[:, 1]))
        return np.pad(c, (3 - len(c), 0))

    if L is None and R is None:
        return None
    pl = fit(L) if L is not None else None
    pr = fit(R) if R is not None else None
    if pl is None:
        pl = pr - np.array([0, 0, lane_width_prior])
    if pr is None:
        pr = pl + np.array([0, 0, lane_width_prior])
    pc = (pl + pr) / 2
    a, b, c = pc
    return LaneState(
        offset=-c,
        heading=np.arctan(-b),  # lane drifting to -X (left) as Y grows => + (points left)
        curvature=-2 * a,
        width=pr[2] - pl[2],
        left=pl, right=pr, center=pc,
    )
