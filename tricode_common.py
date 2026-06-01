"""Shared constants and paths for TriQR modules."""

import base64
import json
import math
import os
import struct
import zlib

CELL_PX = 20
MARGIN = 2
ANCHOR_SIZE = 3
ANCHOR_BUF = 1
ECC_RATIO = 0.24
ANCHOR_SCHEMA = "v3"

TRI_UL = 0b00
TRI_UR = 0b01
TRI_DR = 0b10
TRI_DL = 0b11
CELL_EMPTY = 0b100
CELL_FULL = 0b101


# 각 모서리 패턴과 그 90°/180°/270° 회전이 서로 구별되도록 설계한 비대칭 L자 패턴.
# 검증 완료: 4코너 × 4회전 = 16가지 패턴이 모두 서로 다름.
#
# TL: [F F DR / F E E / F E E]  - 왼쪽+상단 L, DR 삼각형 상단-우
# TR: [DL F F / E E F / E E F]  - 오른쪽+상단 L, DL 삼각형 상단-좌
# BL: [F E E / F E E / F F DR]  - 왼쪽+하단 L, DR 삼각형 하단-우
# BR: [E E F / E E F / DR F F]  - 오른쪽+하단 L, DR 삼각형 하단-좌
ANCHOR_PATTERNS = {
    "TL": [
        [CELL_FULL, CELL_FULL, TRI_DR],
        [CELL_FULL, CELL_EMPTY, CELL_EMPTY],
        [CELL_FULL, CELL_EMPTY, CELL_EMPTY],
    ],
    "TR": [
        [TRI_DL, CELL_FULL, CELL_FULL],
        [CELL_EMPTY, CELL_EMPTY, CELL_FULL],
        [CELL_EMPTY, CELL_EMPTY, CELL_FULL],
    ],
    "BL": [
        [CELL_FULL, CELL_EMPTY, CELL_EMPTY],
        [CELL_FULL, CELL_EMPTY, CELL_EMPTY],
        [CELL_FULL, CELL_FULL, TRI_DR],
    ],
    "BR": [
        [CELL_EMPTY, CELL_EMPTY, CELL_FULL],
        [CELL_EMPTY, CELL_EMPTY, CELL_FULL],
        [TRI_DR, CELL_FULL, CELL_FULL],
    ],
}

_IVEC = {
    ("TL", "TR"): (1, 0),
    ("TL", "BL"): (0, 1),
    ("TL", "BR"): (1, 1),
    ("TR", "TL"): (-1, 0),
    ("TR", "BL"): (-1, 1),
    ("TR", "BR"): (0, 1),
    ("BL", "TL"): (0, -1),
    ("BL", "TR"): (1, -1),
    ("BL", "BR"): (1, 0),
    ("BR", "TL"): (-1, -1),
    ("BR", "TR"): (0, -1),
    ("BR", "BL"): (-1, 0),
}

CORNER_COLORS = {
    "TL": (255, 60, 60),
    "TR": (60, 210, 60),
    "BL": (60, 120, 255),
    "BR": (255, 200, 0),
}

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DB = os.path.join(HERE, "tricode_server.json")
KEYS_DIR = os.path.join(HERE, "tricode_keys")
TMPL_DIR = os.path.join(HERE, "tricode_templates")

SIG_LEN_ED25519 = 64     # Ed25519
FLAG_ZLIB = 0x01
FLAG_SIG_ED25519 = 0x04  # Ed25519 (64-byte sig, public-key verifiable — no password)

_AN = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "[:62]
_AI = {c: i for i, c in enumerate(_AN)}
MODE_AN = 1
MODE_A7 = 2
MODE_U8 = 3


def codeword_stride(n: int) -> int:
    """Return an invertible permutation stride for a codeword of length n."""
    if n <= 1:
        return 1
    stride = max(3, (n // 2) | 1)
    while math.gcd(stride, n) != 1:
        stride += 2
        if stride >= n:
            stride = 3
    return stride


def codeword_offset(n: int) -> int:
    """Return a deterministic offset used with the stride permutation."""
    if n <= 1:
        return 0
    off = n // 3
    if off == 0:
        off = 1
    return off


def codeword_permutation(n: int):
    stride = codeword_stride(n)
    off = codeword_offset(n)
    return [(off + i * stride) % n for i in range(n)]


def codeword_unpermute(data: bytes) -> bytes:
    perm = codeword_permutation(len(data))
    out = bytearray(len(data))
    for i, p in enumerate(perm):
        out[i] = data[p]
    return bytes(out)


def codeword_permute(data: bytes) -> bytes:
    perm = codeword_permutation(len(data))
    out = bytearray(len(data))
    for i, p in enumerate(perm):
        out[p] = data[i]
    return bytes(out)


def codeword_permutation_inverse(n: int):
    perm = codeword_permutation(n)
    inv = [0] * n
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def rs_block_plan(nd: int, nsym: int) -> list:
    """총 (nd+nsym) > 255 일 때 다중 RS 블록 구조를 반환. [(data_bytes, ecc_bytes), ...]"""
    total = nd + nsym
    if total <= 255:
        return [(nd, nsym)]
    n_blocks = math.ceil(total / 255)
    base_d, rem_d = divmod(nd, n_blocks)
    base_e, rem_e = divmod(nsym, n_blocks)
    return [
        (base_d + (1 if i < rem_d else 0), base_e + (1 if i < rem_e else 0))
        for i in range(n_blocks)
    ]


def ecc_ratio_for_data(n_data: int, signed: bool = False) -> float:
    """
    Adaptive ECC ratio:
    - small payloads: stronger correction
    - larger payloads: keep capacity efficient
    - signed payloads: slightly more conservative
    """
    if n_data <= 32:
        base = 0.34
    elif n_data <= 96:
        base = 0.30
    elif n_data <= 224:
        base = 0.26
    else:
        base = 0.22
    if signed:
        base += 0.03
    return min(0.36, max(0.20, base))
