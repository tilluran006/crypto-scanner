"""Breakout Pattern Setup [WillyAlgoTrader] v1.7.2 - deterministic Python port (default inputs).
Converging channel (triangle/wedge) fitted to confirmed swing pivots; signal when a bar CLOSES
outside the channel. SL = opposite boundary; TP1/TP2/TP3 = 1/3, 2/3, 1x measured move;
SL moves to entry after TP1. Same logic as the backtested port (results/h10_*).
Volume/RSI only grade strength (Strong/Medium/Weak); they never filter signals.
"""
import numpy as np

PIV = 5; MIN_T = 2; MAXCH = 120; CONV_MIN = 0.02; TOUCH_TOL = 0.15; DEV_MAX = 0.3; MINW = 0.5
MAX_PIVOTS = 60; RESCAN = 10; MAX_SCAN = 15; MAXW_MULT = 10.0; VOL_CONTR = 0.85; TIMEOUT = 3 * MAXCH


def _rma(x, n):
    out = np.full(len(x), np.nan); s = np.nan
    for i, v in enumerate(x):
        if i == n - 1: s = np.mean(x[:n])
        elif i >= n: s = (s * (n - 1) + v) / n
        out[i] = s
    return out


def _line(x1, y1, x2, y2, x):
    dx = x2 - x1
    return y1 + (y2 - y1) * (x - x1) / dx if abs(dx) > 1e-10 else y1


def signals(T, O, H, L, C, V):
    """Run the indicator over closed bars. Returns list of signal dicts (all history),
    each with its live status as of the last bar."""
    N = len(C)
    if N < 150:
        return []
    pc = np.concatenate([[C[0]], C[:-1]])
    tr = np.maximum(H - L, np.maximum(abs(H - pc), abs(L - pc)))
    atr = _rma(tr, 20)
    d = np.diff(C, prepend=C[0]); up = _rma(np.maximum(d, 0), 14); dn = _rma(np.maximum(-d, 0), 14)
    rsi = 100 - 100 / (1 + up / np.where(dn == 0, 1e-12, dn))
    vsma = np.convolve(V, np.ones(20) / 20, mode="full")[:N]; vsma[:19] = np.nan
    cumV = np.cumsum(V)
    WARM = max(PIV * 2 + MAXCH, 50)
    hiP, hiB, loP, loB = [], [], [], []
    chan = None; trade = None; out = []

    for bi in range(N):
        a20 = atr[bi] if not np.isnan(atr[bi]) else 0.0
        j = bi - PIV; newpiv = False
        if j - PIV >= 0:
            w = H[j - PIV: j + PIV + 1]
            if H[j] >= w[:PIV].max() and H[j] > w[PIV+1:].max():   # TradingView ta.pivothigh tie rule (verified)
                hiP.insert(0, H[j]); hiB.insert(0, j); newpiv = True
            w = L[j - PIV: j + PIV + 1]
            if L[j] <= w[:PIV].min() and L[j] < w[PIV+1:].min():   # TradingView ta.pivotlow tie rule (verified)
                loP.insert(0, L[j]); loB.insert(0, j); newpiv = True
        del hiP[MAX_PIVOTS:], hiB[MAX_PIVOTS:], loP[MAX_PIVOTS:], loB[MAX_PIVOTS:]
        warmed = bi >= WARM

        if (newpiv or bi % RESCAN == 0) and chan is None and trade is None and warmed \
                and len(hiP) >= MIN_T and len(loP) >= MIN_T and a20 > 0:
            rng_h = range(min(len(hiP) - 1, MAX_SCAN) + 1); rng_l = range(min(len(loP) - 1, MAX_SCAN) + 1)
            hX = [hiB[i] for i in rng_h if bi - hiB[i] <= MAXCH]; hY = [hiP[i] for i in rng_h if bi - hiB[i] <= MAXCH]
            lX = [loB[i] for i in rng_l if bi - loB[i] <= MAXCH]; lY = [loP[i] for i in rng_l if bi - loB[i] <= MAXCH]
            devTol = a20 * DEV_MAX; touchTol = a20 * TOUCH_TOL

            def best(X, Y, sign):
                bt = 0; res = None
                for a in range(len(X) - 1):
                    for b in range(a + 1, len(X)):
                        if abs(X[b] - X[a]) < 1: continue
                        ok = True; t = 0
                        for k in range(len(X)):
                            diff = Y[k] - _line(X[a], Y[a], X[b], Y[b], X[k])
                            if (sign > 0 and diff > devTol) or (sign < 0 and diff < -devTol):
                                ok = False; break
                            if abs(diff) <= touchTol: t += 1
                        if ok and t >= MIN_T and t > bt:
                            bt = t; res = (X[a], Y[a], X[b], Y[b], t)
                return res

            bh = best(hX, hY, +1) if len(hX) >= MIN_T else None
            bl = best(lX, lY, -1) if len(lX) >= MIN_T else None
            if bh and bl:
                hx1, hy1, hx2, hy2, ht = bh; lx1, ly1, lx2, ly2, lt = bl
                os_, oe = max(min(hx1, hx2), min(lx1, lx2)), min(max(hx1, hx2), max(lx1, lx2))
                maxH = 0.0
                if oe > os_:
                    ns = max(int((oe - os_) / 5.0), 2)
                    for s in range(ns + 1):
                        sx = os_ + (oe - os_) * s / ns
                        maxH = max(maxH, _line(hx1, hy1, hx2, hy2, sx) - _line(lx1, ly1, lx2, ly2, sx))
                else:
                    mx = (min(hx1, hx2) + max(hx1, hx2) + min(lx1, lx2) + max(lx1, lx2)) / 4
                    maxH = abs(_line(hx1, hy1, hx2, hy2, mx) - _line(lx1, ly1, lx2, ly2, mx))
                sX = min(hx1, hx2, lx1, lx2)
                uN = _line(hx1, hy1, hx2, hy2, bi); lN = _line(lx1, ly1, lx2, ly2, bi); wN = uN - lN
                wS = _line(hx1, hy1, hx2, hy2, sX) - _line(lx1, ly1, lx2, ly2, sX)
                conv = 1 - wN / wS if wS > 0 else 0.0
                inside = wN > 0 and bi >= 1 and lN <= C[bi - 1] <= uN
                pb = max(bi - int(sX), 1); spb = min(pb, bi); pre = min(spb, int(sX))
                vin = (cumV[bi] - (cumV[bi - spb] if bi - spb >= 0 else 0)) / spb
                vpre = ((cumV[bi - spb] if bi - spb >= 0 else 0) - (cumV[bi - spb - pre] if bi - spb - pre >= 0 else 0)) / pre if pre > 0 else vin
                vc = vin / vpre if vpre else 1.0
                if wN > 0 and conv >= CONV_MIN and MINW * a20 <= wN < MAXW_MULT * a20 and inside and vc < VOL_CONTR:
                    chan = dict(h=(hx1, hy1, hx2, hy2), l=(lx1, ly1, lx2, ly2), maxH=maxH, det=bi,
                                touches=(ht, lt), conv=conv)

        if chan is not None and trade is None and warmed:
            up_ = _line(*chan['h'], bi); lo_ = _line(*chan['l'], bi)
            side = +1 if C[bi] > up_ else -1 if C[bi] < lo_ else 0
            if side:
                brk = up_ if side > 0 else lo_; opp = lo_ if side > 0 else up_
                e = C[bi]; tgt = brk + side * chan['maxH']; full = abs(tgt - e)
                # strength grade (label only, as in the script)
                body = abs(C[bi] - O[bi]); rngc = H[bi] - L[bi]; br = body / rngc if rngc > 0 else 0
                mid = (O[bi] + C[bi]) / 2
                pen = min(abs(C[bi] - brk) / a20, 2.0) / 2.0 if a20 > 0 else 0.5
                bc = 1.0 if ((mid > brk) if side > 0 else (mid < brk)) else 0.3
                vok = (V[bi] > vsma[bi] * 1.2) if not np.isnan(vsma[bi]) else True
                mok = (rsi[bi] > 50) if side > 0 else (rsi[bi] < 50)
                score = (pen * 0.25 + br * 0.15 + bc * 0.15 + (1 if vok else 0.4) * 0.25 + (1 if mok else 0.4) * 0.20) * 100
                trade = dict(side=side, bar=bi, t=int(T[bi]), entry=e, sl=opp, sl0=opp,
                             tps=[e + side * full / 3, e + side * full * 2 / 3, e + side * full],
                             hit=[False] * 3, status="ACTIVE", strength="Strong" if score >= 65 else "Medium" if score >= 35 else "Weak",
                             touches=chan['touches'], conv=chan['conv'])
                out.append(trade)
            elif bi - chan['det'] > MAXCH:
                chan = None

        if trade is not None and bi > trade['bar']:
            s = trade['side']; closed = False
            if (L[bi] <= trade['sl']) if s > 0 else (H[bi] >= trade['sl']):
                trade['status'] = "BE EXIT" if trade['hit'][0] else "SL HIT"; closed = True
            else:
                for k in range(3):
                    if not trade['hit'][k] and ((H[bi] >= trade['tps'][k]) if s > 0 else (L[bi] <= trade['tps'][k])):
                        trade['hit'][k] = True
                        if k == 0: trade['sl'] = trade['entry']
                if trade['hit'][2]:
                    trade['status'] = "TP3 HIT"; closed = True
                elif trade['hit'][1]:
                    trade['status'] = "TP2 HIT (SL at BE)"
                elif trade['hit'][0]:
                    trade['status'] = "TP1 HIT (SL at BE)"
                if not closed and bi - trade['bar'] > TIMEOUT:
                    trade['status'] = "TIMEOUT"; closed = True
            if closed:
                trade = None; chan = None
    return out
