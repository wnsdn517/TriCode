"""TriQR payload decoding.

This decoder reconstructs the raw RS payload bytes first, then passes them
through `parse_payload()` so signature and compression work correctly.
"""

import math
import os
from collections import deque

import numpy as np
from PIL import Image, ImageOps

from rs_gf256 import rs_decode, rs_decode_erasure
from triqr_common import ANCHOR_BUF, ANCHOR_SIZE, CELL_PX, ECC_RATIO, HERE, MARGIN, TRI_DL, TRI_DR, TRI_UL, TRI_UR
from triqr_payload import parse_payload
from triqr_render import load_templates

try:
    import cv2

    _CV2 = True
except ImportError:
    _CV2 = False
    cv2 = None

try:
    import triqr_v2 as _legacy
except Exception:
    _legacy = None

CONF_THRESH = 25
PHOTO_ROTATIONS = (0, -15, 15, -30, 30)
PHOTO_THRESHOLDS = (80, 110, 140, 170, 200)
PHOTO_DOWNSAMPLE = 8
PHOTO_PAD = 4


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
    # Payload is roughly 70% of total bytes because ECC ratio is ~30%.
    # Try larger payloads first.
    max_data = max(1, total_bytes - 4)
    for n_data in range(max_data, 0, -1):
        nsym = total_bytes - n_data
        min_nsym = max(4, math.ceil(n_data * ECC_RATIO / (1 - ECC_RATIO)))
        if nsym >= min_nsym:
            yield n_data, nsym


def _component_bboxes(mask: np.ndarray, top_n: int = 6):
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
                if cx < minx:
                    minx = cx
                if cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                if cy > maxy:
                    maxy = cy
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

    if _legacy is not None:
        best = None
        for score, ang, crop, box in candidates:
            try:
                anchors = _legacy._scan_anchors(crop)
                count = len(anchors)
                area = crop.size[0] * crop.size[1]
                if best is None or count > best[0] or (count == best[0] and area < best[1]) or (
                    count == best[0] and area == best[1] and score > best[2]
                ):
                    best = (count, area, score, ang, crop, box)
            except Exception:
                continue
        if best is not None and best[0] > 0:
            _, _, _, ang, crop, _ = best
            return crop, ang

    score, ang, crop, _ = max(candidates, key=lambda item: item[0])
    return crop, ang


def _decode_once(img: Image.Image, templates, thresh, photo_mode, verify_pw=None):
    from triqr_detect import detect, _warp_gray

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


def _decode_raw_from_anchor(arr, side, cpx_f, r_start, c_start, corner):
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
    dibits = []
    confs = []
    for (r, c) in _data_pos(side):
        r0 = round(r_start + (r - r_ref) * cpx_f)
        c0 = round(c_start + (c - c_ref) * cpx_f)
        if r0 < 0 or r0 + cpx > ah or c0 < 0 or c0 + cpx > aw:
            dibits.append(0)
            confs.append(0)
        else:
            d, conf = _read_cell_np(arr, r0, c0, cpx, masks, bg_val)
            dibits.append(d)
            confs.append(conf)
    return _dibits_to_bytes(dibits), confs


def _try_parse_encoded(enc, confidences, verify_pw=None):
    total_bytes = len(enc)
    era = [bi for bi in range(total_bytes) if bi * 4 < len(confidences) and min(confidences[bi * 4 : bi * 4 + 4]) <= CONF_THRESH]

    for n_data, nsym in _candidate_data_lengths(total_bytes):
        try:
            decoded = None
            if era and len(era) <= nsym:
                try:
                    decoded = rs_decode_erasure(enc, nsym, era)
                except Exception:
                    decoded = None
            if decoded is None:
                decoded = rs_decode(enc, nsym)
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
    if len(text) < 4:
        return False
    printable = sum(32 <= ord(ch) <= 126 or ch in "\n\r\t" for ch in text)
    if printable / len(text) < 0.95:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters / len(text) >= 0.7


def _decode_with_anchor(rot_gray, anchors, rect, verify_pw=None):
    if not anchors or rect is None:
        raise ValueError("앵커 정보 부족")

    # Try anchors in descending confidence order.
    for anchor in sorted(anchors, key=lambda a: -a.get("score", 0.0)):
        side_candidates = []
        if rect.get("side"):
            for delta in range(-3, 4):
                s = rect["side"] + delta
                if s >= (ANCHOR_SIZE * 2 + ANCHOR_BUF * 2 + 2):
                    side_candidates.append(s)
        if anchor.get("cpx"):
            est = max(1, round(anchor["cpx"]))
            for delta in range(-2, 3):
                s = rect["side"] + delta if rect.get("side") else max(6, round(min(rot_gray.shape) / est) + delta)
                if s >= (ANCHOR_SIZE * 2 + ANCHOR_BUF * 2 + 2):
                    side_candidates.append(s)
        side_candidates = list(dict.fromkeys(side_candidates))
        cpx_candidates = []
        base_cpx = float(anchor.get("cpx", 0))
        for delta in (-1, 0, 1):
            c = max(1, round(base_cpx + delta)) if base_cpx else None
            if c and c not in cpx_candidates:
                cpx_candidates.append(c)
        if not cpx_candidates and rect.get("cpx"):
            cpx_candidates.append(max(1, round(rect["cpx"])))

        for side in side_candidates:
            for cpx in cpx_candidates:
                try:
                    enc, confs = _decode_raw_from_anchor(
                        rot_gray,
                        side,
                        cpx,
                        anchor["r"],
                        anchor["c"],
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
                    if _legacy is None:
                        raise RuntimeError("cv2 또는 triqr_v2가 필요합니다")

                    raw_anchors = _legacy._scan_anchors(candidate_img)
                    if raw_anchors:
                        detected_rot = raw_anchors[0].get("_rotation", 0)
                        if detected_rot != 0:
                            candidate_img = candidate_img.rotate(-detected_rot, expand=True, fillcolor=(255, 255, 255))
                            raw_anchors = _legacy._scan_anchors(candidate_img)
                    iw, ih = candidate_img.size
                    angle = _legacy._estimate_rotation_continuous(raw_anchors, iw, ih) if raw_anchors else 0.0
                    rotated = _legacy._rotate_image_arbitrary(candidate_img, angle) if abs(angle) > 0.5 else candidate_img
                    rot_gray = np.array(rotated.convert("L"))
                    legacy_anchors = _legacy._scan_anchors(rotated) if raw_anchors else []
                    anchors = []
                    for a in legacy_anchors:
                        aa = dict(a)
                        if "cpx" not in aa:
                            aa["cpx"] = max(1, round(aa.get("cpx_f", 1)))
                        anchors.append(aa)
                    if anchors:
                        pr = _legacy._params_from_anchors(anchors, *rotated.size)
                        if pr:
                            rect = {
                                "side": pr[0],
                                "cpx": pr[1],
                                "mc": pr[2],
                                "angle": angle,
                                "quality": f"{len(anchors)}-corner",
                                "anchors_used": [a["corner"] for a in anchors],
                                "corners": {a["corner"]: (a["cx"], a["cy"]) for a in anchors},
                            }
                        else:
                            rect = None
                    else:
                        rect = None
                    enh = rot_gray.copy()
                    binary = (rot_gray < 128).astype(np.uint8) * 255

                    parsed, used_anchor, side, cpx = _decode_with_anchor(rot_gray, anchors, rect, verify_pw=verify_pw)
                    parsed["angle"] = angle
                    parsed["side"] = side
                    parsed["cpx"] = cpx
                    parsed["anchor"] = used_anchor["corner"]
                    parsed["anchors"] = anchors
                    parsed["rect"] = rect
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
