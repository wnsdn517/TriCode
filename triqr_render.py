"""Anchor rendering and template management."""

import os

import numpy as np
from PIL import Image, ImageDraw

from triqr_common import ANCHOR_PATTERNS, ANCHOR_SIZE, TMPL_DIR, TRI_DL, TRI_DR, TRI_UL, TRI_UR


def _tri_pts(x0, y0, x1, y1, d):
    if d == TRI_UL:
        return [(x0, y0), (x1, y0), (x0, y1)]
    if d == TRI_UR:
        return [(x0, y0), (x1, y0), (x1, y1)]
    if d == TRI_DR:
        return [(x1, y0), (x1, y1), (x0, y1)]
    return [(x0, y0), (x1, y1), (x0, y1)]


def _rotate_pat(pat, n90):
    """Rotate the 2x2 anchor pattern clockwise by 90-degree steps."""
    tmap = {TRI_UL: TRI_UR, TRI_UR: TRI_DR, TRI_DR: TRI_DL, TRI_DL: TRI_UL}
    a = ANCHOR_SIZE
    p = [r[:] for r in pat]
    for _ in range(n90 % 4):
        new = [[None] * a for _ in range(a)]
        for r in range(a):
            for c in range(a):
                new[c][a - 1 - r] = tmap[p[r][c]]
        p = new
    return p


def render_anchor(corner, cell_px, n90=0):
    """Return anchor image as a grayscale ndarray."""
    pat = _rotate_pat(ANCHOR_PATTERNS[corner], n90)
    sz = ANCHOR_SIZE * cell_px
    img = Image.new("L", (sz, sz), 255)
    d = ImageDraw.Draw(img)
    for dr in range(ANCHOR_SIZE):
        for dc in range(ANCHOR_SIZE):
            x0 = dc * cell_px
            y0 = dr * cell_px
            d.polygon(_tri_pts(x0, y0, x0 + cell_px, y0 + cell_px, pat[dr][dc]), fill=0)
    return np.array(img)


_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def save_templates(cell_px=20):
    os.makedirs(TMPL_DIR, exist_ok=True)
    for corner in ANCHOR_PATTERNS:
        for n90 in range(4):
            for sc in _SCALES:
                cpx = max(4, round(cell_px * sc))
                arr = render_anchor(corner, cpx, n90)
                Image.fromarray(arr).save(os.path.join(TMPL_DIR, f"{corner}_r{n90}_{cpx:03d}.png"))
    cnt = 4 * 4 * len(_SCALES)
    print(f"[templates] {TMPL_DIR}  ({cnt}개)")


def load_templates():
    """Return {corner: [(arr, n90), ...]}."""
    t = {c: [] for c in ANCHOR_PATTERNS}
    if os.path.isdir(TMPL_DIR):
        for fn in sorted(os.listdir(TMPL_DIR)):
            if not fn.endswith(".png"):
                continue
            p = fn[:-4].split("_")
            if len(p) < 3:
                continue
            corner, n90 = p[0], int(p[1][1:])
            if corner not in t:
                continue
            arr = np.array(Image.open(os.path.join(TMPL_DIR, fn)).convert("L"))
            t[corner].append((arr, n90))
    if not any(t.values()):
        for corner in ANCHOR_PATTERNS:
            for n90 in range(4):
                for cpx in [8, 12, 16, 20, 24, 30]:
                    t[corner].append((render_anchor(corner, cpx, n90), n90))
    return t
