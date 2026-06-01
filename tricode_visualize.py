"""Visualization helpers."""

import math

from PIL import Image, ImageDraw

from tricode_common import CORNER_COLORS


def visualize(img, anchors, enhanced, binary, rect, decode_result=None, out_path=None):
    iw, ih = img.size
    orig = img.convert("RGB")
    enh_img = Image.fromarray(enhanced).convert("RGB")
    bin_img = Image.fromarray(binary).convert("RGB")

    def _panel(base, show_rect):
        out = base.copy()
        drw = ImageDraw.Draw(out)
        for a in anchors:
            col = CORNER_COLORS.get(a["corner"], (180, 180, 180))
            r, c, bh, bw = a["r"], a["c"], int(a["h"]), int(a["w"])
            drw.rectangle([c - 1, r - 1, c + bw + 1, r + bh + 1], outline=(0, 0, 0), width=1)
            drw.rectangle([c, r, c + bw, r + bh], outline=col, width=3)
            cy2, cx2 = int(a["cy"]), int(a["cx"])
            drw.line([cx2 - 12, cy2, cx2 + 12, cy2], fill=col, width=2)
            drw.line([cx2, cy2 - 12, cx2, cy2 + 12], fill=col, width=2)
            lx = c + bw + 4 if c + bw + 90 < out.width else c - 92
            ly = max(0, r)
            drw.rectangle([lx - 1, ly, lx + 90, ly + 13], fill=(0, 0, 0))
            drw.text((lx + 1, ly + 1), f"{a['corner']} {a['score']:.2f}", fill=col)
            drw.rectangle([lx - 1, ly + 13, lx + 90, ly + 24], fill=(15, 15, 15))
            drw.text((lx + 1, ly + 14), f"cpx={a['cpx']}", fill=(160, 160, 160))
        if show_rect and rect:
            cv_ = rect["corners"]
            conf = set(rect["anchors_used"])
            order = ["TL", "TR", "BR", "BL"]
            for i in range(4):
                ca = order[i]
                cb = order[(i + 1) % 4]
                if ca not in cv_ or cb not in cv_:
                    continue
                p1 = (int(cv_[ca][0]), int(cv_[ca][1]))
                p2 = (int(cv_[cb][0]), int(cv_[cb][1]))
                solid = ca in conf and cb in conf
                if solid:
                    drw.line([p1, p2], fill=(0, 255, 200), width=3)
                else:
                    n = max(int(math.hypot(p2[0] - p1[0], p2[1] - p1[1])) // 8, 1)
                    for si in range(n):
                        if si % 2 == 0:
                            t0, t1 = si / n, (si + 1) / n
                            drw.line(
                                [
                                    (int(p1[0] + t0 * (p2[0] - p1[0])), int(p1[1] + t0 * (p2[1] - p1[1]))),
                                    (int(p1[0] + t1 * (p2[0] - p1[0])), int(p1[1] + t1 * (p2[1] - p1[1]))),
                                ],
                                fill=(80, 200, 140),
                                width=2,
                            )
            for c2 in order:
                if c2 in cv_ and c2 not in conf:
                    ex, ey = int(cv_[c2][0]), int(cv_[c2][1])
                    sz = 8
                    drw.polygon([(ex, ey - sz), (ex + sz, ey), (ex, ey + sz), (ex - sz, ey)], outline=(80, 200, 140))
            if "TL" in cv_ and "TR" in cv_:
                p1 = (int(cv_["TL"][0]), int(cv_["TL"][1]))
                p2 = (int(cv_["TR"][0]), int(cv_["TR"][1]))
                mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                dist = math.hypot(dx, dy)
                if dist > 0:
                    adx, ady = dx / dist * 18, dy / dist * 18
                    ax, ay = int(mx + adx), int(my + ady)
                    drw.line([(mx, my), (ax, ay)], fill=(255, 255, 0), width=3)
                    drw.polygon(
                        [(ax, ay), (int(ax - ady * 0.45), int(ay + adx * 0.45)), (int(ax + ady * 0.45), int(ay - adx * 0.45))],
                        fill=(255, 255, 0),
                    )
        return out

    bar = 48
    w = iw * 3
    h = ih + bar
    res = Image.new("RGB", (w, h), (18, 18, 18))
    res.paste(_panel(orig, True), (0, 0))
    res.paste(_panel(enh_img, False), (iw, 0))
    res.paste(_panel(bin_img, False), (iw * 2, 0))
    drw = ImageDraw.Draw(res)
    for lx, lt in [(4, "원본+탐지"), (iw + 4, "대비강화(디코딩용)"), (iw * 2 + 4, "이진화(탐지용)")]:
        drw.rectangle([lx, 2, lx + 100, 15], fill=(0, 0, 0))
        drw.text((lx + 2, 3), lt, fill=(200, 200, 200))
    drw.line([(iw, 0), (iw, ih)], fill=(60, 60, 60), width=2)
    drw.line([(iw * 2, 0), (iw * 2, ih)], fill=(60, 60, 60), width=2)

    n = len(anchors)
    q = rect["quality"] if rect else "-"
    ag = f"{rect['angle']:.1f}°" if rect else "-"
    cs = ", ".join(sorted(a["corner"] for a in anchors)) or "없음"
    drw.text((6, ih + 4), f"탐지:{n}/4[{cs}]  복원:{q}  기울기:{ag}", fill=(180, 180, 180))

    if decode_result:
        text = decode_result.get("text", "")
        signer = decode_result.get("signer")
        sig_ok = decode_result.get("sig_ok")
        preview = text if len(text) <= 60 else text[:57] + "..."
        drw.text((6, ih + 18), f"데이터: {preview}", fill=(255, 255, 255))
        if signer and sig_ok is True:
            drw.text((6, ih + 32), f"출처: {signer}  ✓ 서명 검증됨", fill=(80, 255, 120))
        elif signer and sig_ok is False:
            drw.text((6, ih + 32), f"출처: {signer}  ✗ 서명 불일치!", fill=(255, 80, 80))
        else:
            drw.rectangle([4, ih + 30, w - 4, ih + 46], fill=(180, 60, 0))
            drw.text((8, ih + 32), "⚠  서명 없음 — 출처 불명, 신뢰 불가", fill=(255, 240, 100))
    else:
        drw.text((6, ih + 18), "(디코드 정보 없음)", fill=(100, 100, 100))

    if out_path:
        res.save(out_path)
    return res
