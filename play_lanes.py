"""Run lane detection over every frame in the sequence and play it at 30 Hz.

Press q to quit.

Usage:
    .venv/bin/python play_lanes.py [frame_dir] [--hz 30] [--method dl|classic] [--no-track]
"""
import argparse
import time
from pathlib import Path

import cv2

DEFAULT_DIR = Path(__file__).parent / "05081544_0305"


def frame_paths(frame_dir):
    # imgList.txt holds the frame order; fall back to sorted file names.
    listing = frame_dir / "imgList.txt"
    if listing.exists():
        names = listing.read_text().split()
        return [frame_dir / n for n in names]
    return sorted(frame_dir.glob("*.jpg"))


def make_pipeline(method, track):
    """Return a function img -> annotated image."""
    if method == "classic":
        from lane_detect import detect, draw_result
        return lambda img: draw_result(img, detect(img))

    from ufld import UFLD, LaneTracker, draw_lanes
    model = UFLD()
    tracker = LaneTracker() if track else None

    def run(img):
        xs, ys = model(img)
        if tracker:
            xs = tracker.update(xs)
        return draw_lanes(img, xs, ys)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--method", choices=["dl", "classic"], default="dl")
    ap.add_argument("--no-track", action="store_true", help="disable temporal smoothing (dl only)")
    args = ap.parse_args()

    paths = frame_paths(Path(args.frame_dir))
    pipeline = make_pipeline(args.method, not args.no_track)
    period = 1.0 / args.hz
    win = "lane detection (q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for i, path in enumerate(paths):
        t0 = time.perf_counter()
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip unreadable {path}")
            continue
        out = pipeline(img)
        proc_ms = (time.perf_counter() - t0) * 1000

        hud = f"{args.method}  {i + 1}/{len(paths)}  {path.name}  proc {proc_ms:.1f} ms"
        cv2.putText(out, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, out)

        # Sleep for whatever is left of the frame period (at least 1 ms for waitKey).
        wait_ms = max(1, int((period - (time.perf_counter() - t0)) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
