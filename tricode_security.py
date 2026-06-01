"""Enrollment and HMAC signing helpers."""

import base64
import hashlib
import hmac
import json
import os
import secrets

try:
    from cryptography.fernet import Fernet

    _FERNET = True
except ImportError:
    _FERNET = False

from tricode_common import KEYS_DIR, SERVER_DB, SIG_LEN


def _db_load():
    return json.load(open(SERVER_DB)) if os.path.exists(SERVER_DB) else {}


def _db_save(d):
    json.dump(d, open(SERVER_DB, "w"), indent=2)


def list_enrolled_names():
    return list(_db_load().keys())


def _key_path(name):
    return os.path.join(KEYS_DIR, f"{name}.key")


def _pw_to_fernet_key(password: str, salt: bytes) -> bytes:
    raw = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, 32)
    return base64.urlsafe_b64encode(raw)


def _save_key(name: str, master_key: bytes, password: str):
    os.makedirs(KEYS_DIR, exist_ok=True)
    salt = secrets.token_bytes(16)
    if _FERNET:
        fk = _pw_to_fernet_key(password, salt)
        token = Fernet(fk).encrypt(master_key)
        open(_key_path(name), "wb").write(salt + token)
    else:
        ks = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, len(master_key))
        ct = bytes(a ^ b for a, b in zip(master_key, ks))
        open(_key_path(name), "wb").write(salt + ct)


def _load_key(name: str, password: str) -> bytes:
    pp = _key_path(name)
    if not os.path.exists(pp):
        raise FileNotFoundError(f"키 파일 없음: {pp}")
    raw = open(pp, "rb").read()
    salt, payload = raw[:16], raw[16:]
    if _FERNET:
        fk = _pw_to_fernet_key(password, salt)
        return Fernet(fk).decrypt(payload)
    key_len = 32
    ks = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, key_len)
    return bytes(a ^ b for a, b in zip(payload, ks))


def cmd_enroll(name: str, password: str):
    if os.path.exists(_key_path(name)):
        print(f"이미 존재: {name}")
        return
    master_key = secrets.token_bytes(32)
    _save_key(name, master_key, password)
    db = _db_load()
    db[name] = {"algo": "hmac-sha256-16", "key_file": _key_path(name)}
    _db_save(db)
    print(f"[enroll] '{name}'  키:{_key_path(name)}  (HMAC-SHA256/16B)")


def _sign_hmac(data: bytes, name: str, password: str) -> bytes:
    key = _load_key(name, password)
    return hmac.new(key, data, hashlib.sha256).digest()[:SIG_LEN]


def _verify_hmac(data: bytes, sig: bytes, name: str, password: str) -> bool:
    try:
        key = _load_key(name, password)
        expected = hmac.new(key, data, hashlib.sha256).digest()[:SIG_LEN]
        return hmac.compare_digest(expected, sig)
    except Exception:
        return False
