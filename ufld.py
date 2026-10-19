"""Deep-learning lane detection with Ultra Fast Lane Detection (UFLD, ECCV 2020).

Model: official ResNet18 weights trained on CULane, ONNX export from ailia-models
(https://storage.googleapis.com/ailia-models/ultra-fast-lane-detection/culane_18.onnx).

UFLD treats lanes as row-wise classification: for each of 18 fixed image rows
("row anchors") and each of 4 lane slots, it picks one of 200 horizontal grid
cells or "no lane". CULane slot order is: 0 = left-left, 1 = ego left,
2 = ego right, 3 = right-right.
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).parent / "models" / "culane_18.onnx"

IN_W, IN_H = 800, 288
GRID = 200
# 18 rows in 288-px input space (official CULane anchors, not evenly spaced).
ROW_ANCHORS = np.array([121, 131, 141, 150, 160, 170, 180, 189, 199,
                        209, 219, 228, 238, 248, 258, 267, 277, 287])
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EGO_LEFT, EGO_RIGHT = 1, 2


class UFLD:
    # CoreML (Apple Neural Engine/GPU) is ~5 ms/frame on M4 vs ~90 ms on CPU.
    def __init__(self, model_path=MODEL_PATH,
                 providers=("CoreMLExecutionProvider", "CPUExecutionProvider")):
        opts = ort.SessionOptions()
        # Idle ORT worker threads otherwise busy-spin after each run and slow down OpenCV.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=list(providers))
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, img):
        """Return (xs, ys): xs is (4, 18) image-space x per lane/row (NaN = absent),
        ys is (18,) image-space y of each row anchor."""
        h, w = img.shape[:2]
        x = cv2.resize(img, (IN_W, IN_H))
        x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
        x = x.transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.input_name: x})[0][0]  # (201, 18, 4)

        # Expected grid position = softmax-weighted average over the 200 cells.
        logits = out[:GRID]
        prob = np.exp(logits - logits.max(axis=0))
        prob /= prob.sum(axis=0)
        loc = (prob * np.arange(1, GRID + 1)[:, None, None]).sum(axis=0)  # (18, 4)
        loc[out.argmax(axis=0) == GRID] = np.nan  # "no lane" class won

        col_w = (IN_W - 1) / (GRID - 1)
        xs = (loc * col_w * w / IN_W - 1).T  # (4, 18)
        xs[np.sum(~np.isnan(xs), axis=1) <= 2] = np.nan  # drop lanes with too few points
        ys = ROW_ANCHORS * h / IN_H - 1
        return xs, ys


class LaneTracker:
    """Per-lane, per-row exponential smoothing with a short hold on dropouts.

    Removes frame-to-frame jitter and bridges brief misses (occlusion, glare)
    without inventing lanes that have been gone for a while.
    """

    def __init__(self, alpha=0.4, hold_frames=8, jump_px=120):
        self.alpha = alpha
        self.hold = hold_frames
        self.jump_px = jump_px
        self.xs = None
        self.miss = None

    def update(self, xs):
        if self.xs is None:
            self.xs = xs.copy()
            self.miss = np.where(np.isnan(xs), self.hold + 1, 0)
            return self.xs.copy()

        seen = ~np.isnan(xs)
        tracked = ~np.isnan(self.xs)
        # A large jump means the slot now holds a different line (e.g. lane change): snap.
        snap = seen & (~tracked | (np.abs(xs - np.nan_to_num(self.xs)) > self.jump_px))
        blend = seen & ~snap
        self.xs[snap] = xs[snap]
        self.xs[blend] = self.alpha * xs[blend] + (1 - self.alpha) * self.xs[blend]

        self.miss[seen] = 0
        self.miss[~seen] += 1
        self.xs[self.miss > self.hold] = np.nan
        return self.xs.copy()


def lane_polyline(x_row, ys, h):
    """Smooth one lane with a quadratic x = f(y) and sample it densely."""
    ok = ~np.isnan(x_row)
    if ok.sum() < 3:
        return None
    coef = np.polyfit(ys[ok], x_row[ok], 2 if ok.sum() >= 5 else 1)
    y_dense = np.linspace(ys[ok].min(), ys[ok].max(), 40)
    pts = np.stack([np.polyval(coef, y_dense), y_dense], axis=1)
    return pts.round().astype(np.int32)


LANE_COLORS = {0: (200, 200, 200), 1: (255, 160, 0), 2: (255, 160, 0), 3: (200, 200, 200)}


def draw_lanes(img, xs, ys):
    out = img.copy()
    h = img.shape[0]
    lanes = {i: lane_polyline(xs[i], ys, h) for i in range(len(xs))}

    left, right = lanes.get(EGO_LEFT), lanes.get(EGO_RIGHT)
    if left is not None and right is not None:
        overlay = out.copy()
        cv2.fillPoly(overlay, [np.vstack([left, right[::-1]])], (0, 180, 0))
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)

    for i, pts in lanes.items():
        if pts is None:
            continue
        thick = 8 if i in (EGO_LEFT, EGO_RIGHT) else 4
        cv2.polylines(out, [pts], False, LANE_COLORS[i], thick, cv2.LINE_AA)
    return out
