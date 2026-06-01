"""TriQR CLI entrypoint.

Feature-specific code lives in:
  - tricode_common.py
  - tricode_render.py
  - tricode_security.py
  - tricode_payload.py
  - tricode_detect.py
  - tricode_encode.py
  - tricode_visualize.py
"""

import getpass
import os
import sys

import numpy as np
from PIL import Image

from tricode_common import HERE
from tricode_detect import detect, _warp_gray
from tricode_decode import decode_image as decode_payload_image
from tricode_encode import encode
from tricode_render import load_templates, save_templates
from tricode_security import cmd_enroll, list_enrolled_names
from tricode_visualize import visualize


def main():
    if len(sys.argv)<2: print(__doc__); return
    cmd=sys.argv[1]; args=sys.argv[2:]

    # ── enroll ──
    if cmd=='enroll':
        if len(args)<2: print("enroll <name> <password>"); return
        cmd_enroll(args[0],args[1])

    # ── encode ──
    elif cmd=='encode':
        if len(args)<2: print("encode <text> <out.png> [--sign]"); return
        text=args[0]; out=args[1]; do_sign='--sign' in args
        sn=sp=None
        if do_sign:
            names=list_enrolled_names()
            if not names: print("먼저 enroll 실행"); return
            sn=names[0] if len(names)==1 else input(f"사용자({', '.join(names)}): ").strip()
            sp=getpass.getpass(f"'{sn}' 비밀번호: ")
        img, meta = encode(text, sign_name=sn, sign_pw=sp, return_info=True)
        img.save(out)
        print(f"[encode] {out!r}")
        print(f"  텍스트 : {text!r}")
        print(f"  페이로드: {meta['payload_len']}B  ECC: {meta['ecc_len']}B  그리드: {meta['grid_side']}×{meta['grid_side']}")
        print(f"  ECC비율 : {meta['ecc_ratio']:.2f}")
        if meta["compression"] == "deflate":
            pct = 100.0 * (1.0 - meta["compression_ratio"])
            print(f"  압축   : raw deflate L{meta['compression_level']}  절감 {pct:.1f}%")
        else:
            print("  압축   : 생략 (압축 이득 없음)")
        if meta["signed"]: print(f"  서명   : {meta['signer']} (HMAC-SHA256/16B)")

    # ── detect ──
    elif cmd=='detect':
        if not args:
            print("detect <image.png> [--out x.png] [--thresh 0.55] [--photo] [--verify-pw PW]")
            return
        in_path    = args[0]
        out_path   = args[args.index('--out')+1] if '--out' in args \
                     else os.path.splitext(in_path)[0]+'_det.png'
        thresh     = float(args[args.index('--thresh')+1]) if '--thresh' in args else 0.55
        photo_mode = '--photo' in args
        verify_pw  = args[args.index('--verify-pw')+1] if '--verify-pw' in args else None

        import time
        img    = Image.open(in_path)
        tmpls  = load_templates()
        print(f"[detect] {in_path}  {img.size[0]}×{img.size[1]}"
              + (" [photo]" if photo_mode else ""))

        st = time.perf_counter()
        anchors, angle, rect, enh, binary = detect(img, tmpls, thresh,
                                                    photo_mode=photo_mode)
        el = time.perf_counter()-st

        # ── 앵커 탐지 결과 ──
        print(f"  탐지: {len(anchors)}/4  ({el:.2f}s)" +
              (f"  보정각: {angle}°" if angle else ""))
        for a in sorted(anchors, key=lambda x: x['corner']):
            print(f"    {a['corner']}  score={a['score']:.2f}"
                  f"  pos=({a['r']},{a['c']})  cpx={a['cpx']}")

        if rect:
            print(f"  복원: {rect['quality']}  기울기={rect['angle']:.1f}°"
                  f"  side={rect['side']}  cpx={rect['cpx']}")
            for c in ('TL','TR','BL','BR'):
                if c in rect['corners']:
                    x, y = rect['corners'][c]
                    tag  = '탐지' if c in rect['anchors_used'] else '추정'
                    print(f"    {c}({tag}) ({x:.0f},{y:.0f})")
        else:
            print("  복원 실패 (앵커 2개 이상 필요)")

        # ── 데이터 / 서명 출력 ──
        # detect만으로는 실제 데이터를 읽을 수 없음 (warp+셀읽기 필요)
        # 그러나 앵커로 payload를 읽을 수 있는 경우 표시
        # 현재는 서명 상태를 decode_result 형태로 visualize에 넘김
        decode_result = None
        print("  ⓘ  데이터 디코딩: detect 후 warp 필요 (decode 명령 별도)")

        # ── 시각화 저장 ──
        visualize(img, anchors, enh, binary, rect,
                  decode_result=decode_result, out_path=out_path)
        print(f"  결과: {out_path}")
        print()
        # 서명 없음 경고
        if not decode_result or not decode_result.get('signer'):
            print("  ⚠  서명 정보 없음 — decode 후 출처 확인 필요")

    # ── decode ──
    elif cmd=='decode':
        if not args:
            print("decode <image.png> [--photo] [--verify-pw PW]"); return
        in_path   = args[0]
        photo_mode= '--photo' in args
        verify_pw = args[args.index('--verify-pw')+1] if '--verify-pw' in args else None
        out_path  = os.path.splitext(in_path)[0]+'_dec.png'

        img    = Image.open(in_path)
        tmpls  = load_templates()
        print(f"[decode] {in_path}  {img.size[0]}×{img.size[1]}")

        import time
        st = time.perf_counter()
        parsed = None
        anchors = []
        angle = 0
        rect = None
        enh = np.array(img.convert("L"))
        binary = np.array(img.convert("L"))
        try:
            parsed, anchors, angle, rect, enh, binary = decode_payload_image(
                img, templates=tmpls, thresh=0.55, photo_mode=photo_mode, verify_pw=verify_pw
            )
        except Exception as e:
            print(f"  디코딩 실패: {e}")
        el = time.perf_counter()-st
        print(f"  탐지/복호: {len(anchors)}/4  ({el:.2f}s)" +
              (f"  보정각: {angle}°" if angle else ""))

        decode_result = None
        if parsed:
            decode_result = parsed
            text   = parsed['text']
            signer = parsed['signer']
            sig_ok = parsed['sig_ok']

            print()
            print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"  데이터  : {text!r}")
            if signer:
                status = "✓ 서명 검증됨" if sig_ok is True \
                    else ("✗ 서명 불일치!" if sig_ok is False else "(검증 안 함)")
                color  = "✓" if sig_ok is True else "✗"
                print(f"  출처    : {signer}  {status}")
            else:
                print(f"  ⚠  서명 없음 — 출처 불명, 신뢰 불가")
            print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        visualize(img, anchors, enh, binary, rect,
                  decode_result=decode_result, out_path=out_path)
        print(f"  결과: {out_path}")

    # ── templates ──
    elif cmd=='templates':
        save_templates()

    else:
        print(f"알 수 없는 명령: {cmd}"); print(__doc__)

if __name__=='__main__':
    main()
