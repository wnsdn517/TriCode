"""Enrollment and signing helpers.

Algorithm selection:
  - PyNaCl available → Ed25519 (public-key signature, no password needed to verify)
  - PyNaCl missing   → HMAC-SHA256/16B fallback (both sign & verify need password)
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
    _NACL = True
except ImportError:
    _NACL = False

try:
    from cryptography.fernet import Fernet
    _FERNET = True
except BaseException:
    _FERNET = False

from tricode_common import KEYS_DIR, SERVER_DB, SIG_LEN, SIG_LEN_ED25519


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


def _derive_keys(password: str, salt: bytes) -> tuple[bytes, bytes]:
    """enc_key(32B) + mac_key(32B) — single PBKDF2 call."""
    km = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, 64)
    return km[:32], km[32:]


def _save_key(name: str, key_bytes: bytes, password: str):
    """Encrypt and persist a 32-byte key (Ed25519 seed or HMAC master key)."""
    os.makedirs(KEYS_DIR, exist_ok=True)
    salt = secrets.token_bytes(16)
    if _FERNET:
        fk = _pw_to_fernet_key(password, salt)
        token = Fernet(fk).encrypt(key_bytes)
        open(_key_path(name), "wb").write(salt + token)
    else:
        enc_key, mac_key = _derive_keys(password, salt)
        ct = bytes(a ^ b for a, b in zip(key_bytes, enc_key))
        tag = hmac.new(mac_key, ct, hashlib.sha256).digest()
        open(_key_path(name), "wb").write(salt + ct + tag)


def _load_key(name: str, password: str) -> bytes:
    pp = _key_path(name)
    if not os.path.exists(pp):
        raise FileNotFoundError(f"키 파일 없음: {pp}")
    raw = open(pp, "rb").read()
    salt = raw[:16]
    if _FERNET:
        return Fernet(_pw_to_fernet_key(password, salt)).decrypt(raw[16:])
    if len(raw) < 80:
        raise ValueError("키 파일이 손상되었습니다 (너무 짧음). 재등록 필요.")
    ct, tag = raw[16:48], raw[48:80]
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
    if _NACL:
        signing_key = nacl.signing.SigningKey.generate()
        seed = bytes(signing_key)                      # 32B private seed
        pubkey = bytes(signing_key.verify_key)         # 32B public key
        _save_key(name, seed, password)
        db[name] = {
            "algo": "ed25519",
            "pubkey": base64.b64encode(pubkey).decode(),
            "key_file": _key_path(name),
        }
        _db_save(db)
        print(f"[enroll] '{name}'  Ed25519  공개키: {base64.b64encode(pubkey).decode()}")
        print(f"  키파일: {_key_path(name)}  (개인키는 비번으로 보호)")
    else:
        master_key = secrets.token_bytes(32)
        _save_key(name, master_key, password)
        db[name] = {"algo": "hmac-sha256-16", "key_file": _key_path(name)}
        _db_save(db)
        print(f"[enroll] '{name}'  HMAC-SHA256/16B  (PyNaCl 없음 — pip install pynacl 권장)")


def sign_data(data: bytes, name: str, password: str) -> bytes:
    """Sign data with the named key. Returns signature bytes."""
    db = _db_load()
    algo = db.get(name, {}).get("algo", "hmac-sha256-16")
    key = _load_key(name, password)
    if algo == "ed25519" and _NACL:
        return bytes(nacl.signing.SigningKey(key).sign(data).signature)  # 64B
    return hmac.new(key, data, hashlib.sha256).digest()[:SIG_LEN]       # 16B


def verify_data(data: bytes, sig: bytes, name: str, password: str = None) -> bool:
    """Verify signature. Ed25519: no password needed. HMAC: password required."""
    try:
        db = _db_load()
        algo = db.get(name, {}).get("algo", "hmac-sha256-16")
        if algo == "ed25519" and _NACL:
            pubkey_b64 = db[name].get("pubkey")
            if not pubkey_b64:
                return False
            vk = nacl.signing.VerifyKey(base64.b64decode(pubkey_b64))
            vk.verify(data, sig)
            return True
        # HMAC fallback — still needs password
        if not password:
            return None  # can't verify without password
        key = _load_key(name, password)
        expected = hmac.new(key, data, hashlib.sha256).digest()[:SIG_LEN]
        return hmac.compare_digest(expected, sig)
    except Exception:
        return False
