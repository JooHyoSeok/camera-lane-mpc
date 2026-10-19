"""HUD drawing kit shared by sim.py (recorded-video simulator) and md_drive.py
(MetaDrive): palette, Korean-capable cached text, rounded panels, translucent
polygons, dashed lines.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def bgr(hex_rgb):
    h = hex_rgb.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


C = dict(
    bg=bgr("#0B0E12"), panel=bgr("#141A21"), edge=bgr("#25303B"),
    text=bgr("#ECEFF2"), muted=bgr("#8794A1"), dim=bgr("#4A5663"),
    path=bgr("#2D8CFF"), line=bgr("#8CCBFF"), truth=bgr("#DDE3E8"), plan=bgr("#3FE0C5"),
    ok=bgr("#5BD18A"), warn=bgr("#F5B53A"), bad=bgr("#F2555A"), car=bgr("#E4E8EC"),
)

FONT_KR = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"
KR_WEIGHT = {"regular": 0, "medium": 2, "semibold": 4, "bold": 6}


class Text:
    """Queue text during cv2 drawing, then render it all in one PIL pass
    (OpenCV's Hershey fonts have no Korean glyphs)."""

    def __init__(self):
        self.items = []
        self.glyphs = {}
        self.cache = {}

    def font(self, size, weight="regular", mono=False):
        key = (size, weight, mono)
        if key not in self.cache:
            try:
                if mono:
                    f = ImageFont.truetype(FONT_MONO, size)
                    try:
                        f.set_variation_by_name("Semibold" if weight in ("semibold", "bold") else "Regular")
                    except Exception:
                        pass
                else:
                    f = ImageFont.truetype(FONT_KR, size, index=KR_WEIGHT[weight])
            except OSError:
                f = ImageFont.load_default()
            self.cache[key] = f
        return self.cache[key]

    def __call__(self, xy, s, size=14, color=C["text"], weight="regular", mono=False, anchor="la"):
        self.items.append((xy, s, self.font(size, weight, mono), color, anchor))

    def width(self, s, size=14, weight="regular", mono=False):
        return self.font(size, weight, mono).getlength(s)

    def pair(self, xy, word, number, size, color, weight="semibold"):
        """Korean word + monospace number (the mono font has no Hangul glyphs)."""
        x, y = xy  # y is the text baseline
        if word:
            self((x, y), word, int(size * 0.8), color, weight, anchor="ls")
            x += self.width(word, int(size * 0.8), weight) + 8
        self((x, y), number, size, color, weight, mono=True, anchor="ls")

    def _glyph(self, s, f, anchor):
        """Rasterize a string once into an alpha mask; reused every frame."""
        key = (s, id(f), anchor)
        g = self.glyphs.get(key)
        if g is None:
            l, t, r, b = f.getbbox(s, anchor=anchor)
            if r <= l or b <= t:
                return None
            im = Image.new("L", (r - l, b - t), 0)
            ImageDraw.Draw(im).text((-l, -t), s, font=f, fill=255, anchor=anchor)
            g = (l, t, np.asarray(im, np.float32)[:, :, None] / 255.0)
            if len(self.glyphs) > 5000:
                self.glyphs.clear()
            self.glyphs[key] = g
        return g

    def render(self, canvas):
        H, W = canvas.shape[:2]
        for xy, s, f, color, anchor in self.items:
            g = self._glyph(s, f, anchor)
            if g is None:
                continue
            l, t, a = g
            x0, y0 = int(round(xy[0] + l)), int(round(xy[1] + t))
            x1, y1 = min(x0 + a.shape[1], W), min(y0 + a.shape[0], H)
            if x1 <= max(x0, 0) or y1 <= max(y0, 0):
                continue
            a = a[max(0, -y0):y1 - y0, max(0, -x0):x1 - x0]
            roi = canvas[max(y0, 0):y1, max(x0, 0):x1]
            roi[:] = (roi * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)
        self.items.clear()
        return canvas


def rrect(img, x, y, w, h, r, color, thickness=-1):
    """Rounded rectangle (filled or outlined), anti-aliased corners."""
    r = int(min(r, w // 2, h // 2))
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        for (cx, cy), a0 in (((x + r, y + r), 180), ((x + w - r, y + r), 270),
                             ((x + w - r, y + h - r), 0), ((x + r, y + h - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 90, color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)


def blend_poly(img, pts, color, alpha):
    """Translucent polygon, blended only inside its bounding box."""
    x, y, w, h = cv2.boundingRect(pts)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.fillPoly(overlay, [pts - (x0, y0)], color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def dashed(img, p0, p1, color, thickness=1, dash=8, gap=6):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    length = np.linalg.norm(p1 - p0)
    if length < 1:
        return
    d = (p1 - p0) / length
    s = 0.0
    while s < length:
        e = min(s + dash, length)
        cv2.line(img, tuple((p0 + d * s).astype(int)), tuple((p0 + d * e).astype(int)), color, thickness, cv2.LINE_AA)
        s += dash + gap


def side_word(v, eps=0.05):
    """Korean side label for a value where + means RIGHT (lateral offsets).
    For + left quantities (heading, steering) pass the negated value."""
    return "중앙" if abs(v) < eps else ("우" if v > 0 else "좌")
