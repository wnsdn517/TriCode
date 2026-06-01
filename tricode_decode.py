"""TriQR payload decoding.

This decoder reconstructs the raw RS payload bytes first, then passes them
through `parse_payload()` so signature and compression work correctly.
"""

import math
import os
from collections import deque

import numpy as np
from PIL import Image, ImageOps

from tricode_rs import rs_decode, rs_decode_erasure, rs_decode_multiblock, rs_decode_erasure_multiblock
from tricode_common import ANCHOR_BUF, ANCHOR_SIZE, CELL_PX, HERE, MARGIN, TRI_DL, TRI_DR, TRI_UL, TRI_UR, ecc_ratio_for_data
from tricode_payload import parse_payload
from tricode_render import load_templates, render_anchor

try:
    import cv2

    _CV2 = True
except ImportError:
    _CV2 = False
    cv2 = None

CONF_THRESH = 25
PHOTO_ROTATIONS = (0, -15, 15, -30, 30)
PHOTO_THRESHOLDS = (80, 110, 140, 170, 200)
PHOTO_DOWNSAMPLE = 8
PHOTO_PAD = 4

try:
    _NEAREST = Image.Resampling.NEAREST
except AttributeError:
    _NEAREST = Image.NEAREST


def _get_masks(px: int) -> dict:
    y, x = np.mgrid[0:px, 0:px]
    return {
        TRI_UL: y < (px - x),
        TRI_UR: y <= x,
        TRI_DR: y > (px - 1 - x),
        TRI_DL: y >= x,
    }


def _image_bg_value(arr_gray: np.ndarray) -> float:
    h, w = arr_gray.shape[:2]
    s = max(1, min(h, w) // 10)
    corners = [arr_gray[:s, :s], arr_gray[:s, w - s :], arr_gray[h - s :, :s], arr_gray[h - s :, w - s :]]
    return float(np.mean([c.mean() for c in corners]))


def _otsu_threshold(arr_gray: np.ndarray) -> int:
    hist = np.bincount(arr_gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return int(np.nanargmax(sigma_b))


def _mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _resize_gray(arr: np.ndarray, size):
    if arr.size == 0:
        return arr
    return np.array(Image.fromarray(arr.astype(np.uint8), mode="L").resize(size, _NEAREST))


def _template_match_score(crop: np.ndarray, tmpl: np.ndarray) -> float:
    if crop.size == 0 or tmpl.size == 0:
        return -1.0
    if crop.shape != tmpl.shape:
        crop = _resize_gray(crop, (tmpl.shape[1], tmpl.shape[0]))
    crop_bin = crop < 128
    tmpl_bin = tmpl < 128
    return float(np.mean(crop_bin == tmpl_bin))


def _binarize_cell(cell: np.ndarray, bg_val: float) -> np.ndarray:
    if bg_val >= 128:
        return cell < (bg_val * 0.55)
    thresh = min(255, bg_val + (255 - bg_val) * 0.45)
    return cell > thresh


def _read_cell_np(arr, r0, c0, px, masks, bg_val):
    cell = arr[r0 : r0 + px, c0 : c0 + px]
    fg = _binarize_cell(cell, bg_val)
    scores = {d: int(np.count_nonzero(fg & masks[d])) for d in masks}
    sv = sorted(scores.values(), reverse=True)
    return max(scores, key=scores.get), sv[0] - sv[1]


def _anchor_cells(side: int):
    a = ANCHOR_SIZE
    b = ANCHOR_BUF
    origins = {
        "TL": (0, 0),
        "TR": (0, side - a),
        "BL": (side - a, 0),
        "BR": (side - a, side - a),
    }
    cells = {}
    for corner, (r0, c0) in origins.items():
        for dr in range(a):
            for dc in range(a):
                cells[(r0 + dr, c0 + dc)] = corner
        for r in range(max(0, r0 - b), min(side, r0 + a + b)):
            for c in range(max(0, c0 - b), min(side, c0 + a + b)):
                cells.setdefault((r, c), corner)
    return cells


def _data_pos(side):
    reserved = _anchor_cells(side)
    return [(r, c) for r in range(side) for c in range(side) if (r, c) not in reserved]


def _dibits_to_bytes(dibits):
    out = bytearray()
    for i in range(0, len(dibits), 4):
        chunk = dibits[i : i + 4]
        while len(chunk) < 4:
            chunk.append(0)
        out.append((chunk[0] << 6) | (chunk[1] << 4) | (chunk[2] << 2) | chunk[3])
    return bytes(out)


def _candidate_data_lengths(total_bytes):
    # 서명 여부(signed)에 따라 ECC 비율이 달라지므로 둘 다 시도하고 중복 제거.
    max_data = max(1, total_bytes - 4)
    seen: set = set()
    for signed in (False, True):
        for n_data in range(max_data, 0, -1):
            nsym = total_bytes - n_data
            ecc_ratio = ecc_ratio_for_data(n_data, signed=signed)
            min_nsym = max(4, math.ceil(n_data * ecc_ratio / (1 - ecc_ratio)))
            if nsym >= min_nsym:
                key = (n_data, nsym)
                if key not in seen:
                    seen.add(key)
                    yield n_data, nsym


def _component_bboxes(mask: np.ndarray, top_n: int = 6):
    # cv2가 있으면 C 구현으로 훨씬 빠르게 처리
    if _CV2:
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8) * 255, connectivity=4
        )
        comps = []
        for label in range(1, n_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            comps.append((area, x, y, x + w - 1, y + h - 1))
        comps.sort(reverse=True)
        return comps[:top_n]
    # fallback: pure-Python BFS
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            cnt = 0
            minx = maxx = x
            miny = maxy = y
            while q:
                cy, cx = q.popleft()
                cnt += 1
                if cx < minx: minx = cx
                if cx > maxx: maxx = cx
                if cy < miny: miny = cy
                if cy > maxy: maxy = cy
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append((cnt, minx, miny, maxx, maxy))
    comps.sort(reverse=True)
    return comps[:top_n]


def _photo_crop_candidates(img: Image.Image):
    """Yield a small number of square-ish crop candidates for large photos."""
    base = img
    for ang in PHOTO_ROTATIONS:
        rot = base.rotate(ang, expand=True, fillcolor=(0, 0, 0)) if ang else base
        small = rot.convert("L").resize((max(1, rot.width // PHOTO_DOWNSAMPLE), max(1, rot.height // PHOTO_DOWNSAMPLE)))
        arr = np.array(small)
        for thr in PHOTO_THRESHOLDS:
            for mask in (arr > thr, arr < thr):
                for best in _component_bboxes(mask, top_n=6):
                    area, x0, y0, x1, y1 = best
                    bw = x1 - x0 + 1
                    bh = y1 - y0 + 1
                    if bw < 12 or bh < 12:
                        continue
                    ratio = bw / max(bh, 1)
                    if not (0.50 <= ratio <= 1.60):
                        continue
                    bbox_frac = (bw * bh) / max(arr.shape[0] * arr.shape[1], 1)
                    if not (0.01 <= bbox_frac <= 0.80):
                        continue
                    base_pad = max(PHOTO_PAD, int(max(bw, bh) * 0.05))
                    pad_values = sorted(
                        {
                            0,
                            max(0, base_pad - 2),
                            base_pad,
                            base_pad + 2,
                            base_pad + 4,
                        }
                    )
                    scale = PHOTO_DOWNSAMPLE
                    squareness = 1.0 - min(abs(1.0 - ratio), 1.0)
                    for pad in pad_values:
                        box = (
                            max(0, (x0 - pad) * scale),
                            max(0, (y0 - pad) * scale),
                            min(rot.width, (x1 + pad) * scale),
                            min(rot.height, (y1 + pad) * scale),
                        )
                        score = area * (0.75 + squareness) / (1.0 + bbox_frac * 2.0 + pad * 0.15)
                        yield score, ang, rot.crop(box), box
                # fall through to next threshold/polarity
                continue


def _prepare_photo_image(img: Image.Image):
    """Return the most promising crop for a large photo, or the original image."""
    if min(img.size) < 240 and max(img.size) < 360:
        return img, 0
    candidates = []
    for score, ang, crop, box in _photo_crop_candidates(img):
        candidates.append((score, ang, crop, box))
    if not candidates:
        return img, 0
    score, ang, crop, _ = max(candidates, key=lambda item: item[0])
    return crop, ang


def _photo_canvas_crop(img: Image.Image):
    """Find a large square-ish bright canvas inside a photo crop."""
    if min(img.size) < 240:
        return img

    scale = max(4, min(8, max(1, min(img.size) // 300)))
    small = img.convert("L").resize((max(1, img.width // scale), max(1, img.height // scale)))
    arr = np.array(small)

    best = None
    for thr in (220, 230, 235, 240, 245):
        mask = arr >= thr
        for area, x0, y0, x1, y1 in _component_bboxes(mask, top_n=12):
            bw = x1 - x0 + 1
            bh = y1 - y0 + 1
            if bw < 20 or bh < 20:
                continue
            ratio = bw / max(bh, 1)
            if not (0.75 <= ratio <= 1.35):
                continue
            if x0 <= 1 or y0 <= 1 or x1 >= arr.shape[1] - 2 or y1 >= arr.shape[0] - 2:
                continue
            score = area / (1.0 + abs(1.0 - ratio) * 20.0)
            if best is None or score > best[0]:
                best = (score, x0, y0, x1, y1)

    if best is None:
        return img

    _, x0, y0, x1, y1 = best
    pad = 2
    box = (
        max(0, (x0 - pad) * scale),
        max(0, (y0 - pad) * scale),
        min(img.width, (x1 + pad + 1) * scale),
        min(img.height, (y1 + pad + 1) * scale),
    )
    return img.crop(box)


def _decode_once(img: Image.Image, templates, thresh, photo_mode, verify_pw=None):
    from tricode_detect import detect, _warp_gray

    anchors, angle, rect, enh, binary = detect(img, templates, thresh=thresh, photo_mode=photo_mode)
    rot_gray = np.array(_warp_gray(np.array(img.convert("L")), angle)) if angle else np.array(img.convert("L"))
    parsed, used_anchor, side, cpx = _decode_with_anchor(rot_gray, anchors, rect, verify_pw=verify_pw)
    parsed["angle"] = angle
    parsed["side"] = side
    parsed["cpx"] = cpx
    parsed["anchor"] = used_anchor["corner"]
    parsed["anchors"] = anchors
    parsed["rect"] = rect
    return parsed, anchors, angle, rect, enh, binary


def _decode_raw_from_anchor(arr, side, cpx_f, r_anchor_center, c_anchor_center, corner):
    """Decode cells using anchor CENTER coordinates for accurate grid alignment."""
    ah, aw = arr.shape[:2]
    cpx = max(1, round(cpx_f))
    masks = _get_masks(cpx)
    bg_val = _image_bg_value(arr)
    REF = {
        "TL": (0, 0),
        "TR": (0, side - ANCHOR_SIZE),
        "BL": (side - ANCHOR_SIZE, 0),
        "BR": (side - ANCHOR_SIZE, side - ANCHOR_SIZE),
    }
    r_ref, c_ref = REF[corner]
    # Anchor bounding box center = (r_ref + ANCHOR_SIZE/2, c_ref + ANCHOR_SIZE/2) in cell units from grid origin.
    # Solve for grid origin:
    grid_origin_r = r_anchor_center - (r_ref + ANCHOR_SIZE / 2) * cpx_f
    grid_origin_c = c_anchor_center - (c_ref + ANCHOR_SIZE / 2) * cpx_f
    dibits = []
    confs = []
    for (r, c) in _data_pos(side):
        r0 = round(grid_origin_r + r * cpx_f)
        c0 = round(grid_origin_c + c * cpx_f)
        if r0 < 0 or r0 + cpx > ah or c0 < 0 or c0 + cpx > aw:
            dibits.append(0)
            confs.append(0)
        else:
            d, conf = _read_cell_np(arr, r0, c0, cpx, masks, bg_val)
            dibits.append(d)
            confs.append(conf)
    return _dibits_to_bytes(dibits), confs


def _decode_upright_pure_python(img: Image.Image, verify_pw=None):
    """Decode a clean upright Tricode image without cv2."""
    arr = np.array(img.convert("L"))
    thr = _otsu_threshold(arr)
    mask = arr <= thr
    h, w = mask.shape
    bbox_candidates = []
    bbox = _mask_bbox(mask)
    if bbox is not None:
        bbox_candidates.append(bbox)
        x0, y0, x1, y1 = bbox
        if x0 <= 1 or y0 <= 1 or x1 >= w - 2 or y1 >= h - 2 or (x1 - x0 + 1) > int(w * 0.9) or (y1 - y0 + 1) > int(h * 0.9):
            for trim in (0.03, 0.05, 0.08, 0.12):
                x0t = int(w * trim)
                y0t = int(h * trim)
                x1t = int(w * (1.0 - trim))
                y1t = int(h * (1.0 - trim))
                if x1t <= x0t or y1t <= y0t:
                    continue
                sub = mask[y0t:y1t, x0t:x1t]
                bb = _mask_bbox(sub)
                if bb is not None:
                    bbox_candidates.append((bb[0] + x0t, bb[1] + y0t, bb[2] + x0t, bb[3] + y0t))
    if not bbox_candidates:
        raise ValueError("TriQR 영역을 찾지 못했습니다")

    def _bbox_score(box):
        bx0, by0, bx1, by1 = box
        bw = bx1 - bx0 + 1
        bh = by1 - by0 + 1
        ratio = bw / max(bh, 1)
        edge_penalty = 0
        if bx0 <= 1:
            edge_penalty += 1
        if by0 <= 1:
            edge_penalty += 1
        if bx1 >= w - 2:
            edge_penalty += 1
        if by1 >= h - 2:
            edge_penalty += 1
        return (edge_penalty, -bw * bh / (1.0 + abs(1.0 - ratio) * 10.0), bw * bh)

    x0, y0, x1, y1 = sorted(bbox_candidates, key=_bbox_score)[0]
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    if bw < 40 or bh < 40:
        raise ValueError("TriQR 영역이 너무 작습니다")

    best = None
    for side in range(6, 41):
        cpx = max(1, round(min(bw, bh) / side))
        if cpx < 4:
            continue
        if abs(side * cpx - bw) > max(3, cpx):
            continue
        if abs(side * cpx - bh) > max(3, cpx):
            continue

        a_px = ANCHOR_SIZE * cpx
        if x0 + a_px > arr.shape[1] or y0 + a_px > arr.shape[0]:
            continue

        score = 0.0
        ok = True
        for corner in ("TL", "TR", "BL", "BR"):
            if corner == "TL":
                cx0, cy0 = x0, y0
            elif corner == "TR":
                cx0, cy0 = x1 - a_px + 1, y0
            elif corner == "BL":
                cx0, cy0 = x0, y1 - a_px + 1
            else:
                cx0, cy0 = x1 - a_px + 1, y1 - a_px + 1

            if cx0 < 0 or cy0 < 0 or cx0 + a_px > arr.shape[1] or cy0 + a_px > arr.shape[0]:
                ok = False
                break

            crop = arr[cy0 : cy0 + a_px, cx0 : cx0 + a_px]
            tmpl = render_anchor(corner, cpx, 0)
            score += _template_match_score(crop, tmpl)

        if not ok:
            continue

        avg = score / 4.0
        if best is None or avg > best[0]:
            best = (avg, side, cpx, x0, y0)

    if best is None:
        raise ValueError("TriQR 앵커를 판독하지 못했습니다")

    _, side, cpx, x0, y0 = best

    best_parsed = None
    best_meta = None
    best_score = float("-inf")

    side_lo = max(6, side - 2)
    side_hi = side + 2
    cpx_lo = max(1, cpx - 2)
    cpx_hi = cpx + 2

    for side_try in range(side_lo, side_hi + 1):
        for cpx_try in range(cpx_lo, cpx_hi + 1):
            a_px = ANCHOR_SIZE * cpx_try
            side_px = side_try * cpx_try
            corners = {
                "TL": (x0, y0),
                "TR": (x0 + side_px - a_px, y0),
                "BL": (x0, y0 + side_px - a_px),
                "BR": (x0 + side_px - a_px, y0 + side_px - a_px),
            }
            if corners["TR"][0] < 0 or corners["BL"][1] < 0:
                continue
            if corners["BR"][0] + a_px > arr.shape[1] or corners["BR"][1] + a_px > arr.shape[0]:
                continue

            anchors = []
            for corner, (cx0, cy0) in corners.items():
                anchors.append(
                    {
                        "corner": corner,
                        "r": int(cy0),
                        "c": int(cx0),
                        "w": a_px,
                        "h": a_px,
                        "cpx": cpx_try,
                        "score": 1.0,
                        "n90": 0,
                        "cx": float(cx0 + a_px / 2.0),
                        "cy": float(cy0 + a_px / 2.0),
                    }
                )

            rect = {
                "quality": "pure-python",
                "angle": 0.0,
                "side": side_try,
                "cpx": cpx_try,
                "anchors_used": ["TL", "TR", "BL", "BR"],
                "corners": {k: (float(v[0]), float(v[1])) for k, v in corners.items()},
            }

            try:
                a_px = ANCHOR_SIZE * cpx_try
                # Pass TL anchor CENTER (x0, y0 are top-left; center = top-left + half anchor)
                enc, confs = _decode_raw_from_anchor(arr, side_try, cpx_try,
                                                     y0 + a_px / 2, x0 + a_px / 2, "TL")
                parsed = _try_parse_encoded(enc, confs, verify_pw=verify_pw)
            except Exception:
                continue

            score = _score_parsed_result(parsed)
            if _is_plausible_text(parsed):
                parsed["angle"] = 0.0
                parsed["side"] = side_try
                parsed["cpx"] = cpx_try
                parsed["anchor"] = "TL"
                parsed["anchors"] = anchors
                parsed["rect"] = rect
                enh = arr.copy()
                binary = (mask.astype(np.uint8) * 255)
                return parsed, anchors, 0.0, rect, enh, binary
            if score > best_score:
                best_score = score
                best_parsed = parsed
                best_meta = (anchors, rect, side_try, cpx_try)

    if best_parsed is None or best_meta is None:
        raise ValueError("payload decode 실패")

    anchors, rect, side_try, cpx_try = best_meta
    best_parsed["angle"] = 0.0
    best_parsed["side"] = side_try
    best_parsed["cpx"] = cpx_try
    best_parsed["anchor"] = "TL"
    best_parsed["anchors"] = anchors
    best_parsed["rect"] = rect
    enh = arr.copy()
    binary = (mask.astype(np.uint8) * 255)
    return best_parsed, anchors, 0.0, rect, enh, binary


def _try_parse_encoded(enc, confidences, verify_pw=None):
    total_bytes = len(enc)
    era = [bi for bi in range(total_bytes) if bi * 4 < len(confidences) and min(confidences[bi * 4 : bi * 4 + 4]) <= CONF_THRESH]

    for n_data, nsym in _candidate_data_lengths(total_bytes):
        try:
            decoded = None
            if era and len(era) <= nsym:
                try:
                    decoded = rs_decode_erasure_multiblock(enc, n_data, nsym, era)
                except Exception:
                    decoded = None
            if decoded is None:
                decoded = rs_decode_multiblock(enc, n_data, nsym)
            return parse_payload(decoded, verify_pw=verify_pw)
        except Exception:
            continue
    raise ValueError("payload decode 실패")


def _score_parsed_result(parsed) -> float:
    text = str(parsed.get("text", ""))
    if not text:
        return -1.0
    printable = sum(32 <= ord(ch) <= 126 or ch in "\n\r\t" for ch in text)
    control = sum(ord(ch) < 32 and ch not in "\n\r\t" for ch in text)
    repl = text.count("�")
    score = printable / len(text)
    score += min(len(text), 40) / 50.0
    score -= control * 0.25
    score -= repl * 0.35
    if parsed.get("signer"):
        score += 0.25
    if parsed.get("sig_ok") is True:
        score += 0.25
    return score


def _is_plausible_text(parsed) -> bool:
    text = str(parsed.get("text", ""))
    if len(text) < 2:
        return False
    # 출력 불가능한 제어 문자(탭/줄바꿈 제외)가 있으면 거짓
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text):
        return False
    printable = sum(32 <= ord(ch) <= 126 or ch in "\n\r\t" for ch in text)
    return printable / len(text) >= 0.90


def _geo_cpx(rect, side):
    """Estimate cell pixel size from rect anchor-center positions."""
    corners = rect.get("corners", {})
    n_cells = max(1, side - ANCHOR_SIZE)
    spans = []
    if "TL" in corners and "TR" in corners:
        dx = corners["TR"][0] - corners["TL"][0]
        dy = corners["TR"][1] - corners["TL"][1]
        spans.append(math.hypot(dx, dy) / n_cells)
    if "TL" in corners and "BL" in corners:
        dx = corners["BL"][0] - corners["TL"][0]
        dy = corners["BL"][1] - corners["TL"][1]
        spans.append(math.hypot(dx, dy) / n_cells)
    if not spans:
        return None
    return sum(spans) / len(spans)


def _decode_with_anchor(rot_gray, anchors, rect, verify_pw=None):
    if not anchors or rect is None:
        raise ValueError("앵커 정보 부족")

    # Compute inter-anchor span for geometry-derived side estimates
    corners = rect.get("corners", {}) if rect else {}
    _spans = []
    if "TL" in corners and "TR" in corners:
        _spans.append(math.hypot(corners["TR"][0] - corners["TL"][0], corners["TR"][1] - corners["TL"][1]))
    if "TL" in corners and "BL" in corners:
        _spans.append(math.hypot(corners["BL"][0] - corners["TL"][0], corners["BL"][1] - corners["TL"][1]))
    _span = sum(_spans) / len(_spans) if _spans else 0

    for anchor in sorted(anchors, key=lambda a: -a.get("score", 0.0)):
        min_side = ANCHOR_SIZE * 2 + ANCHOR_BUF * 2 + 2
        side_candidates = []

        # From rect["side"] ± narrow range (fast path for aligned templates)
        if rect.get("side"):
            for delta in range(-3, 4):
                s = rect["side"] + delta
                if s >= min_side:
                    side_candidates.append(s)

        # Span-based side estimates: sweep cpx hypotheses around anchor["cpx"]
        # This covers the case where cpx_template ≠ cpx_actual
        if _span > 0 and anchor.get("cpx"):
            base = int(anchor["cpx"])
            for cpx_hyp in range(max(4, base - 8), min(40, base + 9)):
                s = round(_span / cpx_hyp + ANCHOR_SIZE)
                if s >= min_side:
                    side_candidates.append(s)

        side_candidates = list(dict.fromkeys(side_candidates))

        for side in side_candidates:
            cpx_candidates = []

            # Geometry-derived cpx (most accurate — independent of template size)
            geo = _geo_cpx(rect, side)
            if geo and geo > 0:
                for delta in (-1, 0, 1):
                    c = max(1, round(geo + delta))
                    if c not in cpx_candidates:
                        cpx_candidates.append(c)

            # Template-size cpx as fallback
            base_cpx = float(anchor.get("cpx", 0))
            for delta in (-1, 0, 1):
                c = max(1, round(base_cpx + delta)) if base_cpx else None
                if c and c not in cpx_candidates:
                    cpx_candidates.append(c)

            if not cpx_candidates and rect.get("cpx"):
                cpx_candidates.append(max(1, round(rect["cpx"])))

            for cpx in cpx_candidates:
                try:
                    enc, confs = _decode_raw_from_anchor(
                        rot_gray,
                        side,
                        cpx,
                        anchor["cy"],   # anchor CENTER row (not template top-left)
                        anchor["cx"],   # anchor CENTER col
                        anchor["corner"],
                    )
                    return _try_parse_encoded(enc, confs, verify_pw=verify_pw), anchor, side, cpx
                except Exception:
                    continue
    raise ValueError("앵커 기반 복호 실패")


def decode_image(img: Image.Image, templates=None, thresh=0.55, photo_mode=False, verify_pw=None):
    """Decode TriQR payload into parsed text/signature info."""
    if templates is None:
        templates = load_templates()

    if not _CV2:
        if photo_mode or max(img.size) >= 1600:
            try:
                prepared, ang = _prepare_photo_image(img)
                if abs(ang) > 0.5:
                    prepared = prepared.rotate(-ang, expand=True, fillcolor=(255, 255, 255))
                canvas = _photo_canvas_crop(prepared)
                if canvas.size != prepared.size:
                    return _decode_upright_pure_python(canvas, verify_pw=verify_pw)
                return _decode_upright_pure_python(prepared, verify_pw=verify_pw)
            except Exception:
                pass
        return _decode_upright_pure_python(img, verify_pw=verify_pw)

    orig_img = img
    candidates = []
    seen = set()

    def add_candidate(candidate_img, candidate_photo_mode):
        key = (id(candidate_img), candidate_img.size, candidate_photo_mode)
        if key not in seen:
            seen.add(key)
            candidates.append((candidate_img, candidate_photo_mode))

    if photo_mode or max(img.size) >= 1600:
        try:
            prepared, _ = _prepare_photo_image(img)
            add_candidate(prepared, photo_mode)
            for pad in (8, 16, 24, 32):
                add_candidate(ImageOps.expand(prepared, border=pad, fill=(255, 255, 255)), photo_mode)
        except Exception:
            pass
    add_candidate(img, photo_mode)
    add_candidate(img, not photo_mode)
    if candidates and candidates[0][0] is not orig_img:
        add_candidate(orig_img, photo_mode)
        add_candidate(orig_img, not photo_mode)
    candidates.sort(key=lambda item: item[0].size[0] * item[0].size[1])

    try:
        best = None
        best_score = float("-inf")
        last_error = None
        for candidate_img, candidate_photo_mode in candidates:
            try:
                if _CV2:
                    parsed, anchors, angle, rect, enh, binary = _decode_once(
                        candidate_img, templates, thresh, candidate_photo_mode, verify_pw=verify_pw
                    )
                else:
                    raise RuntimeError("cv2가 필요합니다")
                score = _score_parsed_result(parsed)
                if _is_plausible_text(parsed):
                    return parsed, anchors, angle, rect, enh, binary
                if score > best_score:
                    best_score = score
                    best = (parsed, anchors, angle, rect, enh, binary)
            except Exception as e:
                last_error = e
                continue
        if best is not None:
            return best
        if last_error:
            raise last_error
    except Exception:
        raise
