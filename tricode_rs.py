"""
GF(2^8) Reed-Solomon — 순수 Python (BM + Forney)
생성 다항식: x^8 + x^4 + x^3 + x^2 + 1  (0x11d)
"""
from functools import lru_cache

PRIM=0x11d; GF=256
EXP=[0]*(GF*2); LOG=[0]*GF
_x=1
for _i in range(GF-1):
    EXP[_i]=_x; LOG[_x]=_i
    _x<<=1
    if _x&GF: _x^=PRIM
for _i in range(GF-1,GF*2): EXP[_i]=EXP[_i-(GF-1)]
del _x,_i

_MUL = [
    bytes((EXP[(LOG[a] + LOG[b]) % (GF - 1)] if a and b else 0) for b in range(GF))
    for a in range(GF)
]

def gm(a,b): return _MUL[a][b]
def gi(a):   return EXP[(GF-1)-LOG[a]]
def gp(a,e): return EXP[(LOG[a]*e)%(GF-1)] if a else 0

def pe(p,x):
    r=0
    for c in p: r=gm(r,x)^c
    return r

def pm(a,b):
    r=[0]*(len(a)+len(b)-1)
    for i,u in enumerate(a):
        for j,v in enumerate(b): r[i+j]^=gm(u,v)
    return r

@lru_cache(maxsize=None)
def _gen(ns):
    g=[1]
    for i in range(ns): g=pm(g,[1,gp(2,i)])
    return g

def rs_encode(data:bytes, nsym:int)->bytes:
    g=_gen(nsym); m=list(data)+[0]*nsym
    for i in range(len(data)):
        c=m[i]
        if c:
            for j,gv in enumerate(g[1:],1): m[i+j]^=gm(gv,c)
    return bytes(data)+bytes(m[len(data):])

def _syndromes(msg,ns):
    return [pe(msg,gp(2,i)) for i in range(ns)]

def _bm(S):
    n=len(S); C=[1,0]; B=[1,0]; L=0; m=1; b=1
    for i in range(n):
        d=S[i]
        for j in range(1,L+1): d^=gm(C[j],S[i-j])
        if d==0: m+=1; continue
        T=C[:]
        c=gm(d,gi(b))
        C+=[0]*(max(0,len(B)+m-len(C)))
        for j,bv in enumerate(B): C[m+j]^=gm(c,bv)
        if 2*L<=i: L=i+1-L; B=T; b=d; m=1
        else: m+=1
    return C[:L+1]

def _chien(loc,n):
    return [n-1-i for i in range(n) if pe(loc,gp(2,i))==0]

def _forney(S,loc,pos,n,ns):
    omega_full=[0]*(ns+len(loc)-1)
    for i,a in enumerate(S[:ns]):
        for j,b in enumerate(loc): omega_full[i+j]^=gm(a,b)
    omega=omega_full[:ns]
    loc_prime=[loc[i] if i%2==1 else 0 for i in range(1,len(loc))]
    coef_pos=[n-1-p for p in pos]
    X=[gp(2,cp) for cp in coef_pos]
    mags=[]
    for Xi in X:
        Xi_inv=gi(Xi)
        om=pe(omega[::-1],Xi_inv)
        lp=pe(loc_prime[::-1] if loc_prime else [1],Xi_inv)
        if lp==0: return None
        mags.append(gm(gm(Xi,om),gi(lp)))
    return mags

class ReedSolomonError(Exception): pass

def rs_decode(data:bytes, nsym:int)->bytes:
    msg=list(data)
    S=_syndromes(msg,nsym)
    if max(S)==0: return bytes(msg[:-nsym])
    loc=_bm(S)
    pos=_chien(loc,len(msg))
    if not pos: raise ReedSolomonError("오류 위치 탐색 실패")
    mags=_forney(S,loc,pos,len(msg),nsym)
    if mags is None: raise ReedSolomonError("Forney 계산 실패")
    for p,mg in zip(pos,mags): msg[p]^=mg
    if max(_syndromes(msg,nsym))!=0: raise ReedSolomonError("복구 실패 — 오류 초과")
    return bytes(msg[:-nsym])

def rs_decode_erasure(data: bytes, nsym: int, era_pos: list) -> bytes:
    """Erasure-only 복구. 최대 nsym개 erasure 복구 가능 (일반 오류의 2배)."""
    msg = list(data)
    n = len(msg)
    S = _syndromes(msg, nsym)
    if max(S) == 0:
        return bytes(msg[:-nsym])
    sigma = [1]
    for k in era_pos:
        sigma = pm(sigma, [1, gp(2, n-1-k)])
    omega_raw = [0]*(nsym + len(sigma))
    for i, s in enumerate(S):
        for j, sg in enumerate(sigma):
            if i+j < len(omega_raw): omega_raw[i+j] ^= gm(s, sg)
    omega = omega_raw[:nsym]
    sig_d = [sigma[k] if k%2==1 else 0 for k in range(1, len(sigma))]
    for k in era_pos:
        Xk = gp(2, n-1-k); Xk_inv = gi(Xk)
        om = pe(omega[::-1], Xk_inv)
        sp = pe(sig_d[::-1] if sig_d else [1], Xk_inv)
        if sp == 0: continue
        msg[k] ^= gm(gm(Xk, om), gi(sp))
    if max(_syndromes(msg, nsym)) != 0:
        raise ReedSolomonError("Erasure 복구 실패 — 오류 초과")
    return bytes(msg[:-nsym])


def rs_encode_multiblock(data: bytes, nsym: int) -> bytes:
    """QR 스타일 다중 블록 RS 인코딩. 총 바이트가 255 초과 시 자동 분할 후 인터리빙."""
    from tricode_common import rs_block_plan
    plan = rs_block_plan(len(data), nsym)
    if len(plan) == 1:
        nd_b, nsym_b = plan[0]
        return rs_encode(data, nsym_b)
    data_parts, ecc_parts = [], []
    offset = 0
    for nd_b, nsym_b in plan:
        encoded = rs_encode(data[offset:offset + nd_b], nsym_b)
        data_parts.append(encoded[:nd_b])
        ecc_parts.append(encoded[nd_b:])
        offset += nd_b
    result = bytearray()
    for i in range(max(len(p) for p in data_parts)):
        for p in data_parts:
            if i < len(p):
                result.append(p[i])
    for i in range(max(len(p) for p in ecc_parts)):
        for p in ecc_parts:
            if i < len(p):
                result.append(p[i])
    return bytes(result)


def rs_decode_multiblock(data: bytes, nd: int, nsym: int) -> bytes:
    """다중 블록 RS 디코딩. 인터리빙 해제 후 각 블록 독립 복호."""
    from tricode_common import rs_block_plan
    plan = rs_block_plan(nd, nsym)
    if len(plan) == 1:
        nd_b, nsym_b = plan[0]
        return rs_decode(data, nsym_b)
    n_blocks = len(plan)
    data_sizes = [nd_b for nd_b, _ in plan]
    ecc_sizes = [nsym_b for _, nsym_b in plan]
    data_blocks = [bytearray() for _ in range(n_blocks)]
    ecc_blocks = [bytearray() for _ in range(n_blocks)]
    idx = 0
    for i in range(max(data_sizes)):
        for bi in range(n_blocks):
            if i < data_sizes[bi]:
                data_blocks[bi].append(data[idx] if idx < len(data) else 0)
                idx += 1
    for i in range(max(ecc_sizes)):
        for bi in range(n_blocks):
            if i < ecc_sizes[bi]:
                ecc_blocks[bi].append(data[idx] if idx < len(data) else 0)
                idx += 1
    result = bytearray()
    for bi in range(n_blocks):
        result.extend(rs_decode(bytes(data_blocks[bi]) + bytes(ecc_blocks[bi]), ecc_sizes[bi]))
    return bytes(result)


def rs_decode_erasure_multiblock(data: bytes, nd: int, nsym: int, era_interleaved: list) -> bytes:
    """다중 블록 erasure 복호. 인터리빙 좌표계의 erasure 위치를 블록별로 매핑."""
    from tricode_common import rs_block_plan
    plan = rs_block_plan(nd, nsym)
    if len(plan) == 1:
        nd_b, nsym_b = plan[0]
        return rs_decode_erasure(data, nsym_b, era_interleaved)
    n_blocks = len(plan)
    data_sizes = [nd_b for nd_b, _ in plan]
    ecc_sizes = [nsym_b for _, nsym_b in plan]
    pos_map: dict[int, tuple[int, int]] = {}
    idx = 0
    for i in range(max(data_sizes)):
        for bi in range(n_blocks):
            if i < data_sizes[bi]:
                pos_map[idx] = (bi, i)
                idx += 1
    for i in range(max(ecc_sizes)):
        for bi in range(n_blocks):
            if i < ecc_sizes[bi]:
                pos_map[idx] = (bi, data_sizes[bi] + i)
                idx += 1
    era_by_block: list[list[int]] = [[] for _ in range(n_blocks)]
    for p in era_interleaved:
        if p in pos_map:
            bi, local_p = pos_map[p]
            era_by_block[bi].append(local_p)
    data_blocks = [bytearray() for _ in range(n_blocks)]
    ecc_blocks = [bytearray() for _ in range(n_blocks)]
    idx = 0
    for i in range(max(data_sizes)):
        for bi in range(n_blocks):
            if i < data_sizes[bi]:
                data_blocks[bi].append(data[idx] if idx < len(data) else 0)
                idx += 1
    for i in range(max(ecc_sizes)):
        for bi in range(n_blocks):
            if i < ecc_sizes[bi]:
                ecc_blocks[bi].append(data[idx] if idx < len(data) else 0)
                idx += 1
    result = bytearray()
    for bi in range(n_blocks):
        block = bytes(data_blocks[bi]) + bytes(ecc_blocks[bi])
        era = era_by_block[bi]
        if era and len(era) <= ecc_sizes[bi]:
            try:
                result.extend(rs_decode_erasure(block, ecc_sizes[bi], era))
                continue
            except ReedSolomonError:
                pass
        result.extend(rs_decode(block, ecc_sizes[bi]))
    return bytes(result)
