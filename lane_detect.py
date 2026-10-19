"""Classical lane detection on a single CULane-style frame (1640x590).

Pipeline: top-hat paint mask + Canny edges -> trapezoid ROI -> Hough
segments -> slope-based left/right split -> weighted line fit per ego lane.

Usage:
    .venv/bin/python lane_detect.py [image_path] [--out result.png]
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_IMAGE = Path(__file__).parent / "05081544_0305" / "05081544_0305-001000.jpg"

# ROI as fractions of (W, H). The bottom edge sits above the car hood.
ROI = np.array([[0.12, 0.74], [0.40, 0.50], [0.56, 0.50], [0.92, 0.74]])
Y_BOTTOM, Y_TOP = 0.74, 0.52  # vertical extent of the drawn lane lines


def paint_mask(img):
    """Pixels brighter than their horizontal neighborhood (lane paint).

    A horizontal top-hat compares each pixel with the road around it, so dull
    gray paint under shade is kept where an absolute white threshold fails.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
    return mask


def edge_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


def roi_mask(shape):
    h, w = shape[:2]
    poly = (ROI * [w, h]).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask, poly


def hough_segments(binary):
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi / 180, threshold=25,
                            minLineLength=25, maxLineGap=40)
    # OpenCV 4 returns (N, 1, 4), OpenCV 5 returns (N, 4).
    return np.empty((0, 4), int) if lines is None else lines.reshape(-1, 4)


def split_left_right(segments, width):
    """Split segments into ego-lane left/right boundaries by slope and side."""
    left, right = [], []
    cx = width * 0.47  # approx. horizontal position of the vanishing point
    for x1, y1, x2, y2 in segments:
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3 or abs(slope) > 3:  # drop near-horizontal / vertical
            continue
        length = np.hypot(x2 - x1, y2 - y1)
        mid_x = (x1 + x2) / 2
        if slope < 0 and mid_x < cx:
            left.append((x1, y1, x2, y2, length))
        elif slope > 0 and mid_x > cx:
            right.append((x1, y1, x2, y2, length))
    return np.array(left), np.array(right)


def fit_lane(segs, h):
    """Length-weighted least squares fit of x = m*y + b; returns two endpoints."""
    if len(segs) == 0:
        return None
    ys = np.concatenate([segs[:, 1], segs[:, 3]])
    xs = np.concatenate([segs[:, 0], segs[:, 2]])
    wts = np.concatenate([segs[:, 4], segs[:, 4]])
    m, b = np.polyfit(ys, xs, 1, w=wts)
    y1, y2 = int(h * Y_BOTTOM), int(h * Y_TOP)
    return (int(m * y1 + b), y1), (int(m * y2 + b), y2)


def detect(img):
    h, w = img.shape[:2]
    cmask = paint_mask(img)
    edges = edge_map(img)
    # Edges of painted pixels: dilate the color mask so paint borders survive.
    combined = cv2.bitwise_and(edges, cv2.dilate(cmask, np.ones((5, 5), np.uint8)))
    rmask, poly = roi_mask(img.shape)
    roi_edges = cv2.bitwise_and(combined, rmask)
    segments = hough_segments(roi_edges)
    left, right = split_left_right(segments, w)
    return {
        "paint_mask": cmask,
        "edges": edges,
        "roi_edges": roi_edges,
        "roi_poly": poly,
        "segments": segments,
        "left_segs": left,
        "right_segs": right,
        "left": fit_lane(left, h),
        "right": fit_lane(right, h),
    }


def draw_result(img, r):
    out = img.copy()
    if r["left"] and r["right"]:
        lane = np.array([r["left"][0], r["left"][1], r["right"][1], r["right"][0]])
        overlay = out.copy()
        cv2.fillPoly(overlay, [lane], (0, 200, 0))
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
    for side, color in (("left", (255, 80, 0)), ("right", (0, 0, 255))):
        if r[side]:
            cv2.line(out, r[side][0], r[side][1], color, 8, cv2.LINE_AA)
    return out


def draw_segments(img, r):
    out = img.copy()
    cv2.polylines(out, [r["roi_poly"]], True, (0, 255, 255), 2)
    for x1, y1, x2, y2 in r["segments"]:
        cv2.line(out, (x1, y1), (x2, y2), (128, 128, 128), 2)
    for segs, color in ((r["left_segs"], (255, 80, 0)), (r["right_segs"], (0, 0, 255))):
        for x1, y1, x2, y2, _ in segs.astype(int) if len(segs) else []:
            cv2.line(out, (x1, y1), (x2, y2), color, 3)
    return out


def visualize(img, r, out_path):
    rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    panels = [
        ("1. Input", rgb(img), None),
        ("2. Paint mask (top-hat)", r["paint_mask"], "gray"),
        ("3. Canny edges", r["edges"], "gray"),
        ("4. Paint edges inside ROI", r["roi_edges"], "gray"),
        ("5. Hough segments (blue=L, red=R)", rgb(draw_segments(img, r)), None),
        ("6. Detected ego lane", rgb(draw_result(img, r)), None),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16, 9))
    for ax, (title, im, cmap) in zip(axes.flat, panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    print(f"saved {out_path}")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE))
    ap.add_argument("--out", default="lane_result.png")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    r = detect(img)
    print(f"{len(r['segments'])} Hough segments | left={r['left']} right={r['right']}")
    visualize(img, r, args.out)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
