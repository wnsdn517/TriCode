"""Anchor rendering and template management."""

import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from tricode_common import ANCHOR_PATTERNS, ANCHOR_SCHEMA, ANCHOR_SIZE, CELL_EMPTY, CELL_FULL, TMPL_DIR, TRI_DL, TRI_DR, TRI_UL, TRI_UR


def _tri_pts(x0, y0, x1, y1, d):
    if d == TRI_UL:
        return [(x0, y0), (x1, y0), (x0, y1)]
    if d == TRI_UR:
        return [(x0, y0), (x1, y0), (x1, y1)]
    if d == TRI_DR:
        return [(x1, y0), (x1, y1), (x0, y1)]
    return [(x0, y0), (x1, y1), (x0, y1)]


def _draw_anchor_cell(draw, x0, y0, x1, y1, code):
    if code == CELL_FULL:
        draw.rectangle([x0, y0, x1, y1], fill=0)
        return
    if code == CELL_EMPTY:
        return
    draw.polygon(_tri_pts(x0, y0, x1, y1, code), fill=0)


def _rotate_pat(pat, n90):
    """Rotate the anchor pattern clockwise by 90-degree steps."""
    tmap = {
        TRI_UL: TRI_UR,
        TRI_UR: TRI_DR,
        TRI_DR: TRI_DL,
        TRI_DL: TRI_UL,
        CELL_FULL: CELL_FULL,
        CELL_EMPTY: CELL_EMPTY,
    }
    a = ANCHOR_SIZE
    p = [r[:] for r in pat]
    for _ in range(n90 % 4):
        new = [[None] * a for _ in range(a)]
        for r in range(a):
            for c in range(a):
                new[c][a - 1 - r] = tmap[p[r][c]]
        p = new
    return p


@lru_cache(maxsize=512)
def render_anchor(corner, cell_px, n90=0):
    """Return anchor image as a grayscale ndarray (cached)."""
    pat = _rotate_pat(ANCHOR_PATTERNS[corner], n90)
    sz = ANCHOR_SIZE * cell_px
    arr = np.full((sz, sz), 255, dtype=np.uint8)

    y_l, x_l = np.mgrid[0:cell_px, 0:cell_px]
    tri_masks = {
        TRI_UL: y_l < (cell_px - x_l),
        TRI_UR: y_l <= x_l,
        TRI_DR: y_l > (cell_px - 1 - x_l),
        TRI_DL: y_l >= x_l,
    }

    for dr in range(ANCHOR_SIZE):
        for dc in range(ANCHOR_SIZE):
            code = pat[dr][dc]
            if code == CELL_EMPTY:
                continue
            r0, c0 = dr * cell_px, dc * cell_px
            cell = arr[r0:r0 + cell_px, c0:c0 + cell_px]
            if code == CELL_FULL:
                cell[:] = 0
            elif code in tri_masks:
                cell[tri_masks[code]] = 0

    arr.flags.writeable = False
    return arr


_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def save_templates(cell_px=20):
    os.makedirs(TMPL_DIR, exist_ok=True)
    for corner in ANCHOR_PATTERNS:
        for n90 in range(4):
            for sc in _SCALES:
                cpx = max(4, round(cell_px * sc))
                arr = render_anchor(corner, cpx, n90)
                Image.fromarray(arr).save(os.path.join(TMPL_DIR, f"{ANCHOR_SCHEMA}_{corner}_r{n90}_{cpx:03d}.png"))
    cnt = 4 * 4 * len(_SCALES)
    print(f"[templates] {TMPL_DIR}  {ANCHOR_SCHEMA}  ({cnt}개)")


@lru_cache(maxsize=2)
def load_templates(schema=ANCHOR_SCHEMA):
    """Return {corner: [(arr, n90), ...]}."""
    t = {c: [] for c in ANCHOR_PATTERNS}
    if os.path.isdir(TMPL_DIR):
        for fn in sorted(os.listdir(TMPL_DIR)):
            if not fn.endswith(".png"):
                continue
            base = fn[:-4]
            if not base.startswith(schema + "_"):
                continue
            p = base.split("_")
            if len(p) < 4:
                continue
            corner, n90 = p[1], int(p[2][1:])
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
