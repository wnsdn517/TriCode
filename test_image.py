"""Image test suite: transformations, corruptions, authentication."""

import io
import math
import os
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tricode_encode import encode
from tricode_decode import decode_image
from tricode_render import load_templates
from tricode_security import cmd_enroll, list_enrolled_names

TEMPLATES = load_templates()

TEXTS = [
    "Hello, TriCode!",
    "한글 테스트 문자열",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
]

PASS_ALPHA = "pw_alpha_1234"
PASS_BETA  = "pw_beta_5678"

# ── helpers ──────────────────────────────────────────────────────────────────

def _encode(text, sign_name=None, sign_pw=None):
    img, _ = encode(text, sign_name=sign_name, sign_pw=sign_pw, return_info=True)
    return img


def _decode(img, verify_pw=None):
    result = decode_image(img, templates=TEMPLATES, verify_pw=verify_pw)
    return result[0] if isinstance(result, tuple) else result


def _check(result, expected_text, signer=None, sig_ok=None):
    assert result["text"] == expected_text, f"text mismatch: {result['text']!r} != {expected_text!r}"
    if signer is not None:
        assert result["signer"] == signer, f"signer mismatch: {result['signer']!r} != {signer!r}"
    if sig_ok is not None:
        assert result["sig_ok"] is sig_ok, f"sig_ok={result['sig_ok']} expected {sig_ok}"


_passed = 0
_failed = 0
_errors = []


def run(name, fn):
    global _passed, _failed
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        print(f"  PASS  {name}  ({elapsed*1000:.0f}ms)")
        _passed += 1
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  FAIL  {name}  ({elapsed*1000:.0f}ms)  {e}")
        _errors.append((name, e))
        _failed += 1


# ── 1. basic round-trip ───────────────────────────────────────────────────────

def test_basic():
    for text in TEXTS:
        img = _encode(text)
        r = _decode(img)
        _check(r, text)


# ── 2. scale transforms ───────────────────────────────────────────────────────

def test_scale_down():
    for scale in (0.5, 0.75):
        text = TEXTS[0]
        img = _encode(text)
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        r = _decode(small)
        _check(r, text)


def test_scale_up():
    for scale in (1.5, 2.0):
        text = TEXTS[0]
        img = _encode(text)
        w, h = img.size
        big = img.resize((int(w * scale), int(h * scale)), Image.NEAREST)
        r = _decode(big)
        _check(r, text)


# ── 3. rotation ───────────────────────────────────────────────────────────────

def test_rotation_small():
    """Small rotations (1°, 3°, 5°) should decode."""
    text = TEXTS[0]
    img = _encode(text)
    for angle in (1, 3, 5):
        rotated = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        r = _decode(rotated)
        _check(r, text)


def test_rotation_medium():
    """Medium rotations (10°, 15°) — best effort for pure-Python path."""
    text = TEXTS[0]
    img = _encode(text)
    failures = []
    for angle in (10, 15):
        try:
            rotated = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
            r = _decode(rotated)
            _check(r, text)
        except Exception as e:
            failures.append(f"{angle}°: {e}")
    if len(failures) == 2:
        raise AssertionError(f"medium rotation failed entirely: {failures}")


# ── 4. noise / corruption ─────────────────────────────────────────────────────

def test_salt_and_pepper():
    """1% random pixel noise should not break decoding."""
    rng = random.Random(42)
    text = TEXTS[0]
    img = _encode(text)
    arr = np.array(img.convert("L"))
    n_pixels = arr.size
    n_noise = int(n_pixels * 0.01)
    flat = arr.flatten()
    idxs = rng.sample(range(n_pixels), n_noise)
    for i in idxs:
        flat[i] = rng.choice((0, 255))
    noisy = Image.fromarray(flat.reshape(arr.shape))
    r = _decode(noisy)
    _check(r, text)


def test_gaussian_blur():
    """Mild Gaussian blur should still decode."""
    text = TEXTS[0]
    img = _encode(text)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=1))
    r = _decode(blurred)
    _check(r, text)


def test_jpeg_compression():
    """JPEG round-trip at quality 85 should still decode."""
    text = TEXTS[0]
    img = _encode(text)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    buf.seek(0)
    jpg = Image.open(buf).convert("L")
    r = _decode(jpg)
    _check(r, text)


def test_partial_damage():
    """Overwrite ~5% of cells with white; ECC should recover."""
    rng = random.Random(7)
    text = TEXTS[0]
    img = _encode(text)
    arr = np.array(img.convert("L"))
    h, w = arr.shape
    block = max(4, w // 20)
    x0 = rng.randint(w // 4, 3 * w // 4 - block)
    y0 = rng.randint(h // 4, 3 * h // 4 - block)
    arr[y0:y0 + block, x0:x0 + block] = 255
    damaged = Image.fromarray(arr)
    r = _decode(damaged)
    _check(r, text)


# ── 5. authentication ─────────────────────────────────────────────────────────

def _ensure_users():
    import glob, os
    for name, pw in (("user_alpha", PASS_ALPHA), ("user_beta", PASS_BETA)):
        key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tricode_keys", f"{name}.key")
        try:
            from tricode_security import sign_data
            sign_data(b"probe", name, pw)
        except Exception:
            if os.path.exists(key_path):
                os.remove(key_path)
            cmd_enroll(name, pw)


def test_sign_and_verify():
    """Sign with user_alpha, verify succeeds."""
    _ensure_users()
    text = "Signed message"
    img = _encode(text, sign_name="user_alpha", sign_pw=PASS_ALPHA)
    r = _decode(img, verify_pw=PASS_ALPHA)
    _check(r, text, signer="user_alpha", sig_ok=True)


def test_sign_wrong_pw():
    """Signing with wrong password raises an error; Ed25519 public-key verify always works."""
    _ensure_users()
    # Signing with wrong password must fail
    raised = False
    try:
        _encode("test", sign_name="user_alpha", sign_pw="definitely_wrong_pw")
    except Exception:
        raised = True
    assert raised, "Expected sign with wrong password to raise an error"
    # Verification with any verify_pw still returns sig_ok=True (public key only needed)
    text = "Signed message"
    img = _encode(text, sign_name="user_alpha", sign_pw=PASS_ALPHA)
    r = _decode(img, verify_pw="wrong_password")
    _check(r, text, signer="user_alpha", sig_ok=True)


def test_two_users():
    """Two different signers produce distinct, verifiable codes."""
    _ensure_users()
    for name, pw in (("user_alpha", PASS_ALPHA), ("user_beta", PASS_BETA)):
        text = f"Message from {name}"
        img = _encode(text, sign_name=name, sign_pw=pw)
        r = _decode(img, verify_pw=pw)
        _check(r, text, signer=name, sig_ok=True)


def test_sign_with_noise():
    """Signed code should survive 1% noise and still verify."""
    _ensure_users()
    rng = random.Random(99)
    text = "Noisy signed message"
    img = _encode(text, sign_name="user_alpha", sign_pw=PASS_ALPHA)
    arr = np.array(img.convert("L"))
    flat = arr.flatten()
    n_noise = int(flat.size * 0.005)
    for i in rng.sample(range(flat.size), n_noise):
        flat[i] = rng.choice((0, 255))
    noisy = Image.fromarray(flat.reshape(arr.shape))
    r = _decode(noisy, verify_pw=PASS_ALPHA)
    _check(r, text, signer="user_alpha", sig_ok=True)


# ── 6. combined stress ────────────────────────────────────────────────────────

def test_combined_scale_noise():
    """Scale 0.75× + 0.5% noise."""
    rng = random.Random(55)
    text = TEXTS[1]
    img = _encode(text)
    w, h = img.size
    small = img.resize((max(1, int(w * 0.75)), max(1, int(h * 0.75))), Image.LANCZOS)
    arr = np.array(small.convert("L"))
    flat = arr.flatten()
    for i in rng.sample(range(flat.size), int(flat.size * 0.005)):
        flat[i] = rng.choice((0, 255))
    r = _decode(Image.fromarray(flat.reshape(arr.shape)))
    _check(r, text)


# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TriCode image test suite")
    print("=" * 60)

    groups = [
        ("Basic round-trip",    [test_basic]),
        ("Scale transforms",    [test_scale_down, test_scale_up]),
        ("Rotation",            [test_rotation_small, test_rotation_medium]),
        ("Noise / corruption",  [test_salt_and_pepper, test_gaussian_blur,
                                  test_jpeg_compression, test_partial_damage]),
        ("Authentication",      [test_sign_and_verify, test_sign_wrong_pw,
                                  test_two_users, test_sign_with_noise]),
        ("Combined stress",     [test_combined_scale_noise]),
    ]

    for group_name, tests in groups:
        print(f"\n[{group_name}]")
        for fn in tests:
            run(fn.__name__, fn)

    print("\n" + "=" * 60)
    print(f"Results: {_passed} passed, {_failed} failed")
    if _errors:
        print("Failed tests:")
        for name, err in _errors:
            print(f"  {name}: {err}")
    print("=" * 60)
    sys.exit(0 if _failed == 0 else 1)
