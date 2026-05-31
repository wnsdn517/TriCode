"""Shared constants and paths for TriQR modules."""

import base64
import json
import os
import struct
import zlib

CELL_PX = 20
MARGIN = 2
ANCHOR_SIZE = 2
ANCHOR_BUF = 1
ECC_RATIO = 0.30

TRI_UL = 0b00
TRI_UR = 0b01
TRI_DR = 0b10
TRI_DL = 0b11

ANCHOR_PATTERNS = {
    "TL": [[TRI_UL, TRI_DR], [TRI_DL, TRI_UR]],
    "TR": [[TRI_DR, TRI_DL], [TRI_UL, TRI_UR]],
    "BL": [[TRI_UL, TRI_DL], [TRI_UR, TRI_DR]],
    "BR": [[TRI_DL, TRI_DR], [TRI_UL, TRI_UR]],
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
SERVER_DB = os.path.join(HERE, "triqr_server.json")
KEYS_DIR = os.path.join(HERE, "triqr_keys")
TMPL_DIR = os.path.join(HERE, "triqr_templates")

SIG_LEN = 16
FLAG_ZLIB = 0x01
FLAG_SIG = 0x02

_AN = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "[:62]
_AI = {c: i for i, c in enumerate(_AN)}
MODE_AN = 1
MODE_A7 = 2
MODE_U8 = 3
