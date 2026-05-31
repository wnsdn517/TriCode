"""Encoding pipeline."""

import math
import os

from PIL import Image, ImageDraw

from rs_gf256 import rs_encode
from triqr_common import ANCHOR_PATTERNS, ANCHOR_BUF, ANCHOR_SIZE, CELL_PX, ECC_RATIO, MARGIN, TMPL_DIR
from triqr_payload import build_payload
from triqr_render import _tri_pts, load_templates, save_templates


def _anchor_res(side):
    a = ANCHOR_SIZE
    b = ANCHOR_BUF
    cells = set()
    for r0, c0 in [(0, 0), (0, side - a), (side - a, 0), (side - a, side - a)]:
        for dr in range(a):
            for dc in range(a):
                cells.add((r0 + dr, c0 + dc))
        for r in range(max(0, r0 - b), min(side, r0 + a + b)):
            for c in range(max(0, c0 - b), min(side, c0 + a + b)):
                cells.add((r, c))
    return len(cells)


def _layout(nd):
    ms = ANCHOR_SIZE * 2 + ANCHOR_BUF * 2 + 2
    nb = max(4, math.ceil(nd * ECC_RATIO / (1 - ECC_RATIO)))
    s = max(ms, math.ceil(math.sqrt((nd + nb) * 4 + 36)))
    while True:
        av = s * s - _anchor_res(s)
        if av % 4 == 0 and av // 4 - nd >= nb:
            break
        s += 1
    return s, av // 4 - nd


def _data_pos(side):
    a = ANCHOR_SIZE
    b = ANCHOR_BUF
    res = set()
    for r0, c0 in [(0, 0), (0, side - a), (side - a, 0), (side - a, side - a)]:
        for dr in range(a):
            for dc in range(a):
                res.add((r0 + dr, c0 + dc))
        for r in range(max(0, r0 - b), min(side, r0 + a + b)):
            for c in range(max(0, c0 - b), min(side, c0 + a + b)):
                res.add((r, c))
    return [(r, c) for r in range(side) for c in range(side) if (r, c) not in res]


def encode(text, cell_px=CELL_PX, margin=MARGIN, sign_name=None, sign_pw=None):
    payload = build_payload(text, sign_name, sign_pw)
    nd = len(payload)
    side, nsym = _layout(nd)
    enc = rs_encode(payload, nsym)
    dibits = []
    [dibits.extend([(b >> s) & 3 for s in (6, 4, 2, 0)]) for b in enc]
    pos = _data_pos(side)
    sz = (side + margin * 2) * cell_px
    img = Image.new("RGB", (sz, sz), (255, 255, 255))
    d = ImageDraw.Draw(img)
    a = ANCHOR_SIZE
    for corner, (r0, c0) in [("TL", (0, 0)), ("TR", (0, side - a)), ("BL", (side - a, 0)), ("BR", (side - a, side - a))]:
        for dr in range(a):
            for dc in range(a):
                x0 = (c0 + dc + margin) * cell_px
                y0 = (r0 + dr + margin) * cell_px
                d.polygon(_tri_pts(x0, y0, x0 + cell_px, y0 + cell_px, ANCHOR_PATTERNS[corner][dr][dc]), fill=(10, 10, 10))
    for i, (r, c) in enumerate(pos):
        dv = dibits[i] if i < len(dibits) else 0
        x0 = (c + margin) * cell_px
        y0 = (r + margin) * cell_px
        d.polygon(_tri_pts(x0, y0, x0 + cell_px, y0 + cell_px, dv), fill=(10, 10, 10))
    if not os.path.isdir(TMPL_DIR) or not os.listdir(TMPL_DIR):
        save_templates(cell_px)
    return img
