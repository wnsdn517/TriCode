"""Enrollment and signing helpers — Ed25519 (PyNaCl required).

Sign with private key (password-protected); verify with stored public key (no password).
"""

import base64
import hashlib
import hmac
import json
import os
import secrets

try:
    import nacl.signing
    import nacl.exceptions
except ImportError:
    raise ImportError("PyNaCl is required: pip install pynacl")

from tricode_common import KEYS_DIR, SERVER_DB, SIG_LEN_ED25519


def _db_load():
    return json.load(open(SERVER_DB)) if os.path.exists(SERVER_DB) else {}


def _db_save(d):
    json.dump(d, open(SERVER_DB, "w"), indent=2)


def list_enrolled_names():
    return list(_db_load().keys())


def _key_path(name):
    return os.path.join(KEYS_DIR, f"{name}.key")


def _derive_keys(password: str, salt: bytes) -> tuple[bytes, bytes]:
    """enc_key(32B) + mac_key(32B) — single PBKDF2 call."""
    km = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, 64)
    return km[:32], km[32:]


def _save_key(name: str, seed: bytes, password: str):
    """Encrypt and persist a 32-byte Ed25519 seed using Encrypt-then-MAC."""
    os.makedirs(KEYS_DIR, exist_ok=True)
    salt = secrets.token_bytes(16)
    enc_key, mac_key = _derive_keys(password, salt)
    ct = bytes(a ^ b for a, b in zip(seed, enc_key))
    tag = hmac.new(mac_key, ct, hashlib.sha256).digest()
    open(_key_path(name), "wb").write(salt + ct + tag)


def _load_key(name: str, password: str) -> bytes:
    pp = _key_path(name)
    if not os.path.exists(pp):
        raise FileNotFoundError(f"키 파일 없음: {pp}")
    raw = open(pp, "rb").read()
    if len(raw) < 80:
        raise ValueError("키 파일이 손상되었습니다. 재등록 필요.")
    salt, ct, tag = raw[:16], raw[16:48], raw[48:80]
    enc_key, mac_key = _derive_keys(password, salt)
    expected_tag = hmac.new(mac_key, ct, hashlib.sha256).digest()
    if not hmac.compare_digest(expected_tag, tag):
        raise ValueError("키 파일 인증 실패 — 비밀번호 오류 또는 파일 변조.")
    return bytes(a ^ b for a, b in zip(ct, enc_key))


def cmd_enroll(name: str, password: str):
    if os.path.exists(_key_path(name)):
        print(f"이미 존재: {name}")
        return
    db = _db_load()
    signing_key = nacl.signing.SigningKey.generate()
    seed = bytes(signing_key)
    pubkey = bytes(signing_key.verify_key)
    _save_key(name, seed, password)
    db[name] = {
        "algo": "ed25519",
        "pubkey": base64.b64encode(pubkey).decode(),
        "key_file": _key_path(name),
    }
    _db_save(db)
    print(f"[enroll] '{name}'  Ed25519  공개키: {base64.b64encode(pubkey).decode()}")
    print(f"  키파일: {_key_path(name)}  (개인키는 비번으로 보호)")


def sign_data(data: bytes, name: str, password: str) -> bytes:
    """Sign data with the named Ed25519 key. Returns 64-byte signature."""
    seed = _load_key(name, password)
    return bytes(nacl.signing.SigningKey(seed).sign(data).signature)


def verify_data(data: bytes, sig: bytes, name: str, password: str = None) -> bool:
    """Verify Ed25519 signature using stored public key (no password needed)."""
    try:
        db = _db_load()
        pubkey_b64 = db.get(name, {}).get("pubkey")
        if not pubkey_b64:
            return False
        vk = nacl.signing.VerifyKey(base64.b64decode(pubkey_b64))
        vk.verify(data, sig)
        return True
    except Exception:
        return False
