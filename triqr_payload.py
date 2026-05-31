"""Payload packing and parsing."""

import struct
import zlib

from triqr_common import FLAG_SIG, FLAG_ZLIB, MODE_A7, MODE_AN, MODE_U8, SIG_LEN, _AI, _AN
from triqr_security import _sign_hmac, _verify_hmac


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


def build_payload(text, sign_name=None, sign_pw=None):
    m = _sel(text)
    text_body = struct.pack(">BH", m, len(text)) + _enc(text, m)
    flag = 0

    compressed = zlib.compress(text_body, level=6)
    if len(compressed) < len(text_body):
        flag |= FLAG_ZLIB
        body = compressed
    else:
        body = text_body

    if sign_name and sign_pw:
        sig = _sign_hmac(text_body, sign_name, sign_pw)
        nb = sign_name.encode()
        tail = sig + nb + struct.pack("B", len(nb))
        flag |= FLAG_SIG
        return struct.pack("B", flag) + body + tail

    return struct.pack("B", flag) + body


def parse_payload(raw, verify_pw=None):
    flag = raw[0]

    if flag & FLAG_SIG:
        sl = raw[-1]
        signer = raw[-1 - sl : -1].decode("utf-8")
        sig = raw[-1 - sl - SIG_LEN : -1 - sl]
        body = raw[1:-1 - sl - SIG_LEN]
    else:
        body = raw[1:]
        signer = None
        sig = None

    text_body = zlib.decompress(body) if (flag & FLAG_ZLIB) else body

    if signer and sig is not None:
        sig_ok = _verify_hmac(text_body, sig, signer, verify_pw) if verify_pw else None
    else:
        sig_ok = None

    m, char_len = struct.unpack(">BH", text_body[:3])
    enc_bytes = text_body[3:]

    if m == MODE_AN:
        bits = nb = 0
        chars = []
        for b in enc_bytes:
            bits = (bits << 8) | b
            nb += 8
            while nb >= 6 and len(chars) < char_len:
                nb -= 6
                chars.append(_AN[(bits >> nb) & 0x3F])
        text = "".join(chars)
    elif m == MODE_A7:
        bits = nb = 0
        chars = []
        for b in enc_bytes:
            bits = (bits << 8) | b
            nb += 8
            while nb >= 7 and len(chars) < char_len:
                nb -= 7
                chars.append(chr((bits >> nb) & 0x7F))
        text = "".join(chars)
    else:
        text = enc_bytes.decode("utf-8", "replace")

    return {"text": text, "signer": signer, "sig_ok": sig_ok}
