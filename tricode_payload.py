"""Payload packing and parsing."""

import struct
import zlib

from tricode_common import (
    FLAG_SIG, FLAG_SIG_ED25519, FLAG_ZLIB,
    MODE_A7, MODE_AN, MODE_U8,
    SIG_LEN, SIG_LEN_ED25519,
    _AI, _AN,
)
from tricode_security import sign_data, verify_data


def _sel(t):
    if all(c in _AI for c in t):
        return MODE_AN
    if all(ord(c) < 128 for c in t):
        return MODE_A7
    return MODE_U8


def _enc(t, m):
    if m == MODE_AN:
        bits = nb = 0
        o = bytearray()
        for c in t:
            bits = (bits << 6) | _AI[c]
            nb += 6
            while nb >= 8:
                nb -= 8
                o.append((bits >> nb) & 0xFF)
        if nb:
            o.append((bits << (8 - nb)) & 0xFF)
        return bytes(o)
    if m == MODE_A7:
        bits = nb = 0
        o = bytearray()
        for c in t:
            bits = (bits << 7) | ord(c)
            nb += 7
            while nb >= 8:
                nb -= 8
                o.append((bits >> nb) & 0xFF)
        if nb:
            o.append((bits << (8 - nb)) & 0xFF)
        return bytes(o)
    return t.encode("utf-8")


def _compress_deflate(data: bytes, level: int) -> bytes:
    comp = zlib.compressobj(level=level, method=zlib.DEFLATED, wbits=-15)
    return comp.compress(data) + comp.flush()


def _decompress_body(body: bytes) -> bytes:
    try:
        return zlib.decompress(body, wbits=-15)
    except zlib.error:
        return zlib.decompress(body)


def _pick_body(text_body: bytes):
    best_kind = "raw"
    best_level = 0
    best_body = text_body
    for level in (1, 6, 9):
        comp = _compress_deflate(text_body, level)
        if len(comp) < len(best_body):
            best_kind = "deflate"
            best_level = level
            best_body = comp
    return best_kind, best_level, best_body


def _repack_text(text: str, mode: int) -> bytes:
    return struct.pack(">BH", mode, len(text)) + _enc(text, mode)


def _dec_an(enc_bytes, char_len):
    bits = nb = 0
    chars = []
    for b in enc_bytes:
        bits = (bits << 8) | b
        nb += 8
        while nb >= 6 and len(chars) < char_len:
            nb -= 6
            chars.append(_AN[(bits >> nb) & 0x3F])
    return "".join(chars)


def _dec_a7(enc_bytes, char_len):
    bits = nb = 0
    chars = []
    for b in enc_bytes:
        bits = (bits << 8) | b
        nb += 8
        while nb >= 7 and len(chars) < char_len:
            nb -= 7
            chars.append(chr((bits >> nb) & 0x7F))
    return "".join(chars)


def build_payload(text, sign_name=None, sign_pw=None, return_meta=False):
    char_len = len(text)
    if char_len > 0xFFFF:
        raise ValueError("텍스트가 너무 깁니다 (최대 65535자)")
    m = _sel(text)
    text_body = struct.pack(">BH", m, char_len) + _enc(text, m)
    flag = 0

    compression, level, body = _pick_body(text_body)
    if compression != "raw":
        flag |= FLAG_ZLIB

    if sign_name and sign_pw:
        name_bytes = sign_name.encode("utf-8")
        if len(name_bytes) > 0xFF:
            raise ValueError("서명자 이름이 너무 깁니다 (최대 255바이트)")
        sig = sign_data(text_body, sign_name, sign_pw)
        # Set the right flag based on sig length
        if len(sig) == SIG_LEN_ED25519:
            flag |= FLAG_SIG_ED25519
        else:
            flag |= FLAG_SIG
        tail = sig + name_bytes + struct.pack("B", len(name_bytes))
        payload = struct.pack("B", flag) + body + tail
    else:
        payload = struct.pack("B", flag) + body

    if return_meta:
        meta = {
            "raw_len": len(text_body),
            "body_len": len(body),
            "compression": compression,
            "compression_level": level if compression != "raw" else None,
            "compression_ratio": (len(body) / len(text_body)) if len(text_body) else 1.0,
        }
        return payload, meta
    return payload


def parse_payload(raw, verify_pw=None):
    if not raw:
        raise ValueError("빈 payload")
    flag = raw[0]

    # Determine signature length from flag
    if flag & FLAG_SIG_ED25519:
        sig_len = SIG_LEN_ED25519
    elif flag & FLAG_SIG:
        sig_len = SIG_LEN
    else:
        sig_len = 0

    if sig_len > 0:
        if len(raw) < sig_len + 2:
            raise ValueError("서명 payload가 너무 짧습니다")
        sl = raw[-1]
        tail_start = len(raw) - 1 - sl
        if tail_start < 1 + sig_len:
            raise ValueError("서명 payload 길이가 맞지 않습니다")
        signer = raw[tail_start:-1].decode("utf-8")
        sig = raw[tail_start - sig_len : tail_start]
        body = raw[1 : tail_start - sig_len]
    else:
        body = raw[1:]
        signer = None
        sig = None

    text_body = _decompress_body(body) if (flag & FLAG_ZLIB) else body
    if len(text_body) < 3:
        raise ValueError("payload 헤더가 손상되었습니다")

    if signer and sig is not None:
        if flag & FLAG_SIG_ED25519:
            # Ed25519: public-key verification — no password needed
            sig_ok = verify_data(text_body, sig, signer)
        else:
            # HMAC legacy: password required
            sig_ok = verify_data(text_body, sig, signer, verify_pw) if verify_pw else None
    else:
        sig_ok = None

    m, char_len = struct.unpack(">BH", text_body[:3])
    enc_bytes = text_body[3:]

    if m == MODE_AN:
        text = _dec_an(enc_bytes, char_len)
    elif m == MODE_A7:
        text = _dec_a7(enc_bytes, char_len)
    else:
        text = enc_bytes.decode("utf-8", "strict")

    if _repack_text(text, m) != text_body:
        raise ValueError("payload roundtrip 검증 실패")

    return {"text": text, "signer": signer, "sig_ok": sig_ok}
