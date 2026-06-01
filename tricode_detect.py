"""Anchor detection and geometric reconstruction."""

import math
from functools import lru_cache

import numpy as np
from PIL import Image

try:
    import cv2

    _CV2 = True
except ImportError:
    _CV2 = False
    cv2 = None

from tricode_common import ANCHOR_PATTERNS, ANCHOR_SIZE, ANCHOR_BUF, CORNER_COLORS, _IVEC
from tricode_render import load_templates, render_anchor


@lru_cache(maxsize=2)
def _get_clahe(photo_mode=False):
    clip = 2.0 if not photo_mode else 2.5
    tile = (8, 8) if not photo_mode else (6, 6)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)


@lru_cache(maxsize=1)
def _get_coarse_templates(scale=0.5):
    return {
        c: [(render_anchor(c, max(4, round(cpx * scale)), n90), n90) for n90 in range(4) for cpx in [8, 12, 16, 20, 24, 30]]
        for c in ("TL", "TR", "BL", "BR")
    }


def preprocess(gray_or_bgr, photo_mode=False):
    if not _CV2:
        raise RuntimeError("pip install opencv-python")
    g = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY) if gray_or_bgr.ndim == 3 else gray_or_bgr.copy()
    enh = _get_clahe(photo_mode).apply(g)
    if photo_mode:
        # 사진: 언샵 + 대형 블록 크기 적응형 임계값
        blur = cv2.GaussianBlur(enh, (0, 0), 1.5)
        enh = np.clip(cv2.addWeighted(enh, 1.8, blur, -0.8, 0), 0, 255).astype(np.uint8)
        binary = cv2.adaptiveThreshold(
            enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 4
        )
    else:
        # 클린 디지털 이미지: Otsu 전역 임계값 — CELL_FULL 등 대형 균일 블록에 강인
        _, binary = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return enh, binary


def _warp_gray(gray, angle_deg):
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cs, sn = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sn + w * cs)
    nh = int(h * cs + w * sn)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _geom_err(cm):
    def d(a, b):
        return math.hypot(cm[b]["cx"] - cm[a]["cx"], cm[b]["cy"] - cm[a]["cy"])

    err = 0.0
    cnt = 0
    dh = [d(a, b) for a, b in [("TL", "TR"), ("BL", "BR")] if a in cm and b in cm]
    dv = [d(a, b) for a, b in [("TL", "BL"), ("TR", "BR")] if a in cm and b in cm]
    if len(dh) == 2:
        err += abs(dh[0] - dh[1]) / (max(dh) + 1e-6)
        cnt += 1
    if len(dv) == 2:
        err += abs(dv[0] - dv[1]) / (max(dv) + 1e-6)
        cnt += 1
    if dh and dv:
        err += abs(sum(dh) / len(dh) - sum(dv) / len(dv)) / (max(sum(dh) / len(dh), sum(dv) / len(dv)) + 1e-6)
        cnt += 1
    return err / max(cnt, 1)


def _corner_quadrant_valid(cm):
    """Return True if each detected corner lies in the expected image quadrant."""
    if len(cm) < 4:
        return True
    cxs = [v["cx"] for v in cm.values()]
    cys = [v["cy"] for v in cm.values()]
    cx_mid = (min(cxs) + max(cxs)) / 2
    cy_mid = (min(cys) + max(cys)) / 2
    checks = {
        "TL": lambda v: v["cx"] < cx_mid and v["cy"] < cy_mid,
        "TR": lambda v: v["cx"] > cx_mid and v["cy"] < cy_mid,
        "BL": lambda v: v["cx"] < cx_mid and v["cy"] > cy_mid,
        "BR": lambda v: v["cx"] > cx_mid and v["cy"] > cy_mid,
    }
    return all(checks[c](cm[c]) for c in cm if c in checks)


def _match_binary(binary, templates, thresh, photo_mode=False):
    h, w = binary.shape
    by_cpx = {}

    for corner, arr_list in templates.items():
        for tmpl, n90 in arr_list:
            th, tw = tmpl.shape[:2]
            if th > h or tw > w:
                continue
            res = cv2.matchTemplate(binary, 255 - tmpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mxloc = cv2.minMaxLoc(res)
            if mx < thresh:
                continue
            cpx = tw // ANCHOR_SIZE
            if cpx not in by_cpx:
                by_cpx[cpx] = {}
            ex = by_cpx[cpx].get(corner)
            if ex is None or mx > ex["score"]:
                bx, by_ = mxloc
                by_cpx[cpx][corner] = {
                    "corner": corner,
                    "r": int(by_),
                    "c": int(bx),
                    "w": tw,
                    "h": th,
                    "cpx": cpx,
                    "score": float(mx),
                    "n90": n90,
                    "cx": float(bx + tw / 2),
                    "cy": float(by_ + th / 2),
                }

    if photo_mode:
        existing_cpx = set(by_cpx.keys())
        for cpx in range(4, min(h // 4, 41)):  # practical max 40; covers all real-world cell sizes
            if cpx in existing_cpx:
                continue
            a_px = ANCHOR_SIZE * cpx
            if a_px > h or a_px > w:
                continue
            for n90 in range(4):
                for corner in ("TL", "TR", "BL", "BR"):
                    tmpl = render_anchor(corner, cpx, n90)
                    res = cv2.matchTemplate(binary, 255 - tmpl, cv2.TM_CCOEFF_NORMED)
                    _, mx, _, mxloc = cv2.minMaxLoc(res)
                    if mx < thresh:
                        continue
                    if cpx not in by_cpx:
                        by_cpx[cpx] = {}
                    ex = by_cpx[cpx].get(corner)
                    if ex is None or mx > ex["score"]:
                        bx, by_ = mxloc
                        by_cpx[cpx][corner] = {
                            "corner": corner,
                            "r": int(by_),
                            "c": int(bx),
                            "w": a_px,
                            "h": a_px,
                            "cpx": cpx,
                            "score": float(mx),
                            "n90": n90,
                            "cx": float(bx + a_px / 2),
                            "cy": float(by_ + a_px / 2),
                        }

    best = []
    bk = (0, 9.9)
    for cpx, cm in sorted(by_cpx.items(), key=lambda x: -x[0]):
        n = len(cm)
        err = _geom_err(cm)
        if n == 4 and not _corner_quadrant_valid(cm):
            err += 10.0  # large penalty: all 4 found but corners are in wrong quadrants
        if n > bk[0] or (n == bk[0] and err < bk[1]):
            best = list(cm.values())
            bk = (n, err)
        # Only break early when scores are high-confidence — prevents wrong cpx winning
        # at low score over the correct cpx with perfect score
        if n == 4 and err < 0.10 and all(a["score"] >= 0.85 for a in cm.values()):
            break
    return best


def detect_anchors(gray, templates, thresh_init=0.60, photo_mode=False):
    h, w = gray.shape
    _, b0 = preprocess(gray, photo_mode)
    best = _match_binary(b0, templates, thresh_init, photo_mode)
    if len(best) == 4:
        for a in best:
            a["_angle"] = 0
        return best, 0

    best_n = len(best)
    best_ang = 0
    best_anchors = best

    scale = 0.5
    mini = _get_coarse_templates(scale)
    coarse_thresh = min(thresh_init, 0.50)

    # 소각도 먼저, 이후 90/180/270 대회전까지 전방향 탐색
    coarse_angles = [0]
    for step in range(10, 50, 10):
        coarse_angles.extend([-step, step])
    coarse_angles.extend([90, -90, 135, -135, 180])

    for ang in coarse_angles:
        gs = _warp_gray(gray, ang)
        gs_s = cv2.resize(gs, (int(w * scale), int(h * scale)))
        _, bs = preprocess(gs_s, photo_mode)
        a = _match_binary(bs, mini, coarse_thresh, False)
        if len(a) > best_n:
            best_n = len(a)
            best_ang = ang
            best_anchors = a
        if best_n == 4:
            break

    for ang in range(best_ang - 3, best_ang + 4):
        gs = _warp_gray(gray, ang) if ang else gray
        _, bs = preprocess(gs, photo_mode)
        a = _match_binary(bs, templates, thresh_init, photo_mode)
        if len(a) > best_n:
            best_n = len(a)
            best_ang = ang
            best_anchors = a
        if best_n == 4:
            break

    low_thresh = 0.45 if photo_mode else 0.50
    if best_n < 4:
        for thresh in [0.50, 0.47, low_thresh]:
            for ang in [best_ang] + list(range(best_ang - 3, best_ang + 4)):
                gs = _warp_gray(gray, ang) if ang else gray
                _, bs = preprocess(gs, photo_mode)
                a = _match_binary(bs, templates, thresh, photo_mode)
                if len(a) > best_n:
                    best_n = len(a)
                    best_ang = ang
                    best_anchors = a
                if best_n == 4:
                    break
            if best_n == 4:
                break

    for a in best_anchors:
        a["_angle"] = best_ang
    return best_anchors, best_ang


def reconstruct_rect(anchors):
    if not anchors:
        return None
    cm = {a["corner"]: a for a in anchors}
    cpx = round(sum(a["cpx"] for a in anchors) / len(anchors))

    def ctr(c):
        return cm[c]["cx"], cm[c]["cy"]

    angs = []
    spans = []
    for (ca, cb), (ix, iy) in _IVEC.items():
        if ca not in cm or cb not in cm:
            continue
        ax, ay = ctr(ca)
        bx, by = ctr(cb)
        angs.append(math.degrees(math.atan2(by - ay, bx - ax)) - math.degrees(math.atan2(iy, ix)))
        spans.append(math.hypot(bx - ax, by - ay) / math.hypot(ix, iy))
    if not angs:
        return None
    s_ = sum(math.sin(math.radians(a)) for a in angs)
    c_ = sum(math.cos(math.radians(a)) for a in angs)
    ang = math.degrees(math.atan2(s_ / len(angs), c_ / len(angs)))
    span = sum(spans) / len(spans)
    side = max(6, round(span / cpx + ANCHOR_SIZE))
    sp = (side - ANCHOR_SIZE) * cpx

    def rv(vx, vy):
        rad = math.radians(ang)
        return vx * math.cos(rad) - vy * math.sin(rad), vx * math.sin(rad) + vy * math.cos(rad)

    cxy = {c: ctr(c) for c in cm}
    for _ in range(3):
        for mc in [c for c in ("TL", "TR", "BL", "BR") if c not in cxy]:
            for kc, kp in list(cxy.items()):
                if (kc, mc) not in _IVEC:
                    continue
                ix, iy = _IVEC[(kc, mc)]
                dx, dy = rv(ix * sp, iy * sp)
                cxy[mc] = (kp[0] + dx, kp[1] + dy)
                break
    return {
        "corners": cxy,
        "angle": ang,
        "cpx": cpx,
        "side": side,
        "quality": {4: "4-corner", 3: "3-corner", 2: "2-corner", 1: "1-corner"}.get(len(cm), "?"),
        "anchors_used": list(cm.keys()),
    }


def enhance_for_decode(gray_orig: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enh = clahe.apply(gray_orig)
    lo, hi = np.percentile(enh, [2, 98])
    if hi > lo:
        enh = np.clip((enh.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return enh


def detect(img_pil, templates=None, thresh=0.55, photo_mode=False):
    if not _CV2:
        raise RuntimeError("pip install opencv-python")
    if templates is None:
        templates = load_templates()

    orig_pil = img_pil
    orig_gray = np.array(img_pil.convert("L"))

    scale = 1.0
    if photo_mode:
        max_dim = 800
        iw, ih = img_pil.size
        if max(iw, ih) > max_dim:
            scale = max_dim / max(iw, ih)
            img_pil = img_pil.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)

    detect_gray = np.array(img_pil.convert("L"))
    anchors, angle = detect_anchors(detect_gray, templates, thresh, photo_mode=photo_mode)

    if scale != 1.0 and anchors:
        for a in anchors:
            for k in ("r", "c", "h", "w", "cx", "cy"):
                if k in a:
                    a[k] = a[k] / scale

    if angle:
        orig_gray_corrected = _warp_gray(orig_gray, angle)
    else:
        orig_gray_corrected = orig_gray

    orig_enhanced = enhance_for_decode(orig_gray_corrected)
    _, detect_binary = preprocess(orig_gray_corrected, photo_mode=False)
    rect = reconstruct_rect(anchors)
    return anchors, angle, rect, orig_enhanced, detect_binary
