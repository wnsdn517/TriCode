"""Encoding pipeline."""

import math
from functools import lru_cache

from PIL import Image, ImageDraw

from tricode_rs import rs_encode_multiblock
from tricode_common import ANCHOR_PATTERNS, ANCHOR_BUF, ANCHOR_SIZE, CELL_EMPTY, CELL_FULL, CELL_PX, MARGIN, TRI_UL, TRI_UR, TRI_DR, TRI_DL, ecc_ratio_for_data
from tricode_payload import build_payload
from tricode_render import _tri_pts, _TRI_FRAC

_BYTE_TO_DIBITS = tuple(((b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, b & 3) for b in range(256))


def _corner_tri_pts(x0: int, y0: int, px: int, dibit: int):
    """Small right-angle triangle at one of 4 cell corners — encodes 2-bit dibit.

    Each triangle is confined to its quadrant, so the quadrant decoder masks
    score it unambiguously (correct quadrant ~t²/2 px, others 0).
    """
    q = px // 2          # quadrant side
    p = max(1, px // 10) # padding from cell edge
    t = q - 2 * p        # triangle leg length
    if dibit == TRI_UL:  # top-left corner ↖
        return [(x0 + p,     y0 + p),
                (x0 + p + t, y0 + p),
                (x0 + p,     y0 + p + t)]
    if dibit == TRI_UR:  # top-right corner ↗
        return [(x0 + q + p,     y0 + p),
                (x0 + q + p + t, y0 + p),
                (x0 + q + p + t, y0 + p + t)]
    if dibit == TRI_DR:  # bottom-right corner ↘
        return [(x0 + q + p + t, y0 + q + p),
                (x0 + q + p + t, y0 + q + p + t),
                (x0 + q + p,     y0 + q + p + t)]
    # TRI_DL: bottom-left corner ↙
    return [(x0 + p,     y0 + q + p),
            (x0 + p + t, y0 + q + p + t),
            (x0 + p,     y0 + q + p + t)]


def _draw_anchor_cell(draw, x0, y0, x1, y1, code):
    px = x1 - x0
    if code == CELL_FULL:
        rr = max(1, px // 4)
        draw.rounded_rectangle([x0, y0, x1 - 1, y1 - 1], radius=rr, fill=(10, 10, 10))
        return
    if code == CELL_EMPTY:
        return
    # Triangle cell → indicator circle at the corresponding quadrant center
    ry, rx = _TRI_FRAC[code]
    cx = int(x0 + rx * px)
    cy = int(y0 + ry * px)
    rad = max(1, px // 5)
    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(10, 10, 10))


@lru_cache(maxsize=None)
def _reserved_cells(side):
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
    return frozenset(cells)


@lru_cache(maxsize=None)
def _anchor_res(side):
    return len(_reserved_cells(side))


def _layout(nd, signed=False):
    ms = ANCHOR_SIZE * 2 + ANCHOR_BUF * 2 + 2
    ecc_ratio = ecc_ratio_for_data(nd, signed=signed)
    nb = max(4, math.ceil(nd * ecc_ratio / (1 - ecc_ratio)))
    s = max(ms, math.ceil(math.sqrt((nd + nb) * 4 + 36)))
    while True:
        av = s * s - _anchor_res(s)
        if av % 4 == 0 and av // 4 - nd >= nb:
            break
        s += 1
    return s, av // 4 - nd


@lru_cache(maxsize=None)
def _data_pos(side):
    res = _reserved_cells(side)
    return [(r, c) for r in range(side) for c in range(side) if (r, c) not in res]


def _rounded_corners(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=radius, fill=255)
    out = Image.new("RGB", img.size, (255, 255, 255))
    out.paste(img, mask=mask)
    return out


def encode(text, cell_px=CELL_PX, margin=MARGIN, sign_name=None, sign_pw=None, return_info=False):
    payload_result = build_payload(text, sign_name, sign_pw, return_meta=True)
    payload, payload_meta = payload_result
    nd = len(payload)
    signed = bool(sign_name and sign_pw)
    side, nsym = _layout(nd, signed=signed)
    enc = rs_encode_multiblock(payload, nsym)
    dibits = bytearray(len(enc) * 4)
    for i, b in enumerate(enc):
        base = i * 4
        d0, d1, d2, d3 = _BYTE_TO_DIBITS[b]
        dibits[base] = d0
        dibits[base + 1] = d1
        dibits[base + 2] = d2
        dibits[base + 3] = d3
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
                _draw_anchor_cell(d, x0, y0, x0 + cell_px, y0 + cell_px, ANCHOR_PATTERNS[corner][dr][dc])
    # Data cells: small corner triangle — encodes dibit as corner position.
    for i, (r, c) in enumerate(pos):
        dv = dibits[i] if i < len(dibits) else 0
        cx = (c + margin) * cell_px
        cy = (r + margin) * cell_px
        d.polygon(_corner_tri_pts(cx, cy, cell_px, dv), fill=(28, 28, 28))
    img = _rounded_corners(img, radius=cell_px * margin // 2)
    if return_info:
        return img, {
            "payload_len": nd,
            "ecc_len": nsym,
            "ecc_ratio": ecc_ratio_for_data(nd, signed=signed),
            "grid_side": side,
            "signed": signed,
            "signer": sign_name if sign_name and sign_pw else None,
            "compression": payload_meta["compression"],
            "compression_level": payload_meta["compression_level"],
            "compression_ratio": payload_meta["compression_ratio"],
        }
    return img
