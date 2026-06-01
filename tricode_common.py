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

ANCHOR_PATTERNS = {
    "TL": [
        [CELL_FULL, CELL_FULL, CELL_FULL],
        [CELL_FULL, CELL_EMPTY, TRI_DR],
        [CELL_FULL, TRI_UL, CELL_FULL],
    ],
    "TR": [
        [CELL_FULL, CELL_FULL, CELL_FULL],
        [TRI_DL, CELL_EMPTY, CELL_FULL],
        [CELL_FULL, TRI_UR, CELL_FULL],
    ],
    "BL": [
        [CELL_FULL, TRI_DL, CELL_FULL],
        [CELL_FULL, CELL_EMPTY, TRI_UR],
        [CELL_FULL, CELL_FULL, CELL_FULL],
    ],
    "BR": [
        [CELL_FULL, TRI_DR, CELL_FULL],
        [TRI_UL, CELL_EMPTY, CELL_FULL],
        [CELL_FULL, CELL_FULL, CELL_FULL],
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

SIG_LEN = 16
FLAG_ZLIB = 0x01
FLAG_SIG = 0x02

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
