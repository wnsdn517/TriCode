"""
GF(2^8) Reed-Solomon — 순수 Python (BM + Forney)
생성 다항식: x^8 + x^4 + x^3 + x^2 + 1  (0x11d)
"""
PRIM=0x11d; GF=256
EXP=[0]*(GF*2); LOG=[0]*GF
_x=1
for _i in range(GF-1):
    EXP[_i]=_x; LOG[_x]=_i
    _x<<=1
    if _x&GF: _x^=PRIM
for _i in range(GF-1,GF*2): EXP[_i]=EXP[_i-(GF-1)]
del _x,_i

def gm(a,b): return EXP[(LOG[a]+LOG[b])%(GF-1)] if a and b else 0
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
    """
    Erasure-only 복구 (위치를 알고 있는 경우).
    최대 nsym개 erasure 복구 가능 (일반 오류의 2배).
    """
    msg = list(data)
    n = len(msg)
    S = _syndromes(msg, nsym)
    if max(S) == 0:
        return bytes(msg[:-nsym])
    # erasure locator sigma
    sigma = [1]
    for k in era_pos:
        sigma = pm(sigma, [1, gp(2, n-1-k)])
    # omega = S * sigma mod x^nsym
    omega_raw = [0]*(nsym + len(sigma))
    for i, s in enumerate(S):
        for j, sg in enumerate(sigma):
            if i+j < len(omega_raw): omega_raw[i+j] ^= gm(s, sg)
    omega = omega_raw[:nsym]
    # formal derivative of sigma
    sig_d = [sigma[k] if k%2==1 else 0 for k in range(1, len(sigma))]
    # Forney magnitudes
    for k in era_pos:
        Xk = gp(2, n-1-k); Xk_inv = gi(Xk)
        om = pe(omega[::-1], Xk_inv)
        sp = pe(sig_d[::-1] if sig_d else [1], Xk_inv)
        if sp == 0: continue
        msg[k] ^= gm(gm(Xk, om), gi(sp))
    if max(_syndromes(msg, nsym)) != 0:
        raise ReedSolomonError("Erasure 복구 실패 — 오류 초과")
    return bytes(msg[:-nsym])
