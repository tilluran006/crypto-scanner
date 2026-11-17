"""Crypto scanner: daily picker (top 10 by normal daily range) + breakout/breakdown signals
(WillyAlgoTrader BPS logic) on 4H / 1H / 5m across the top-200 Binance USDT spot pairs.
Deterministic: same candles in -> same list out. No trades are placed. No API keys needed for data.

Usage:  python scanner.py            (sends to Telegram if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set,
                                       otherwise prints and writes out/latest.txt)
        python scanner.py --dry-run  (never sends)
"""
import json, os, sys, time, html, datetime as dt, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from bps import signals as bps_signals
from day_picker import features, rank   # validated picker logic (2-yr backtest)

BASE = "https://data-api.binance.vision/api/v3"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SLOTS_IST = [8, 14, 20]                       # scheduled scan hours (IST)
UNIVERSE_N = 200
MIN_QVOL = 2_000_000                          # coins below this 24h volume are marked "thin"
TFS = {"4h": 500, "1h": 500, "5m": 500}       # bars fetched per timeframe
FIVE_MIN_FRESH_MIN = 60                       # 5m signals older than this are stale and dropped

# Backtest evidence shown with every message (from the Gate-2 research, 0.2% fees)
EVIDENCE = {
    "4h": "backtest: slightly positive, unproven (+0.05 to +0.08R/trade, 11 coins, 9 yrs)",
    "1h": "backtest: NOT TESTED",
    "5m": "backtest: LOSES after fees (-0.3 to -0.7R/trade) - info only",
}

STABLE = {"USDC","FDUSD","TUSD","DAI","USDP","EUR","EURI","GBP","AUD","BRL","TRY","USD1","USDE","XUSD",
          "AEUR","PAXG","XAUT","WBTC","WBETH","BUSD","UST","USTC","FRAX","LUSD","GUSD","SUSD","PYUSD",
          "RLUSD","BFUSD","USDS","USDY","RWUSD","U"}
TOKENIZED = {"NVDAB","SOXLB","SPCXB","TSLAB","AAPLB","GOOGLB","AMZNB","METAB","MSTRB","COINB","HOODB",
             "CRCLB","SPYB","QQQB","TQQQB","NFLXB","AMDB","PLTRB","GMEB","INTCB","ORCLB","LLYB","UNHB",
             "MRVLB","AVGOB","IWMB","GLDB","MCDB","ABTB","KOB","PFEB","JNJB","WMTB","BABAB","JPMB","VB",
             "MAB","DFDVB","UPXIB","BMNRB","SBETB","STKEB","CEPOB"}


def get(path, retries=4):
    url = BASE + path
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "crypto-scanner/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def universe():
    info = get("/exchangeInfo?permissions=SPOT")
    # Binance tokenized stocks/ETFs (e.g. MSFTB, SOXSB) carry permission group TRD_GRP_261 - exclude them
    trading = {s["symbol"]: s["baseAsset"] for s in info["symbols"]
               if s["status"] == "TRADING" and s["quoteAsset"] == "USDT"
               and not any("TRD_GRP_261" in ps for ps in s.get("permissionSets", []))}
    tick = get("/ticker/24hr")
    rows = []
    for t in tick:
        sym = t["symbol"]; base = trading.get(sym)
        if not base or not sym.isascii(): continue
        if base in STABLE or base in TOKENIZED: continue
        if any(base.endswith(x) and len(base) > len(x) + 1 for x in ("UP", "DOWN", "BULL", "BEAR")): continue
        rows.append((sym, float(t["quoteVolume"]), float(t["lastPrice"]), float(t["priceChangePercent"])))
    rows.sort(key=lambda r: -r[1])
    return rows[:UNIVERSE_N]


def klines(sym, interval, limit):
    k = get(f"/klines?symbol={urllib.parse.quote(sym)}&interval={interval}&limit={limit}")
    now_ms = int(time.time() * 1000)
    k = [r for r in k if int(r[6]) < now_ms]          # closed bars only
    if not k: return None
    a = np.array([[float(r[i]) for i in (0, 1, 2, 3, 4, 5)] for r in k])   # col 5 = base volume (= Pine 'volume')
    return a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]


def window_start(now_ist):
    """Start of the 'new since last scan' window = previous scheduled slot (minus 30 min margin)."""
    today = now_ist.replace(minute=0, second=0, microsecond=0)
    slots = sorted([today.replace(hour=h) - dt.timedelta(days=d) for d in (0, 1) for h in SLOTS_IST])
    cur = max(s for s in slots if s <= now_ist + dt.timedelta(minutes=45))
    prev = max(s for s in slots if s < cur)
    return prev - dt.timedelta(minutes=30)


def fmt_px(x):
    if x >= 100: return f"{x:,.2f}"
    if x >= 1: return f"{x:.4f}"
    if x >= 0.01: return f"{x:.5f}"
    return f"{x:.8f}".rstrip("0")


def scan():
    now = dt.datetime.now(IST)
    wstart = window_start(now)
    uni = universe()
    syms = [u[0] for u in uni]
    meta = {u[0]: u for u in uni}

    # ---------- daily picker (validated features()/rank() from day_picker.py) ----------
    def picker_row(sym):
        try:
            k = get(f"/klines?symbol={urllib.parse.quote(sym)}&interval=1d&limit=61")
        except Exception:
            return None
        now_ms = time.time() * 1000
        bars = [(float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[7])) for b in k if b[6] < now_ms]
        if len(bars) < 51: return None
        f = features(bars[-60:])
        if f["vol7"] < MIN_QVOL: return None
        return dict(sym=sym, **f)
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(picker_row, syms) if r]
    picks = rank(rows)[:10]
    today_open_ms = int(now.astimezone(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    for p in picks:
        sym, qv, last, chg = meta[p["sym"]]
        k = klines(p["sym"], "1h", 30)
        th = tl = to = last
        if k is not None:
            T, O, H, L, C, Q = k
            m = T >= today_open_ms
            if m.any():
                th, tl, to = H[m].max(), L[m].min(), O[m][0]
        p.update(last=last, chg24=chg, adr=p["atr14"] * 100,
                 today_rng=(max(th, last) - min(tl, last)) / to * 100 if to else 0,
                 today_move=(last / to - 1) * 100 if to else 0)

    # ---------- breakout signals ----------
    def tf_signals(arg):
        sym, tf = arg
        try:
            k = klines(sym, tf, TFS[tf])
        except Exception:
            return []
        if k is None: return []
        T, O, H, L, C, Q = k
        sigs = bps_signals(T, O, H, L, C, Q)
        bar_ms = {"4h": 4 * 3600e3, "1h": 3600e3, "5m": 300e3}[tf]
        out = []
        for s in sigs:
            close_ms = s["t"] + bar_ms
            close_ist = dt.datetime.fromtimestamp(close_ms / 1000, IST)
            if tf == "5m":
                if now - close_ist > dt.timedelta(minutes=FIVE_MIN_FRESH_MIN): continue
            elif close_ist < wstart:
                continue
            if s["status"] in ("SL HIT", "TP3 HIT", "BE EXIT", "TIMEOUT"):
                continue
            last = float(C[-1]); e = s["entry"]; side = s["side"]
            late = (last - s["tps"][0]) * side >= 0           # already past TP1 -> don't chase
            risk = abs(e - s["sl0"]) / e * 100
            out.append(dict(sym=sym, tf=tf, side="LONG" if side > 0 else "SHORT", when=close_ist,
                            entry=e, sl=s["sl0"], tps=s["tps"], risk=risk, last=last,
                            rr1=abs(s["tps"][0] - e) / abs(e - s["sl0"]) if e != s["sl0"] else 0,
                            rr3=abs(s["tps"][2] - e) / abs(e - s["sl0"]) if e != s["sl0"] else 0,
                            status=s["status"], strength=s["strength"], late=late,
                            from_entry=(last / e - 1) * 100 * side,
                            thin=meta[sym][1] < MIN_QVOL))
        return out
    jobs = [(s, tf) for tf in TFS for s in syms]
    with ThreadPoolExecutor(8) as ex:
        sigs = [x for lst in ex.map(tf_signals, jobs) for x in lst]

    btc = meta.get("BTCUSDT")
    return dict(now=now, wstart=wstart, n_uni=len(uni), n_liquid=sum(1 for u in uni if u[1] >= MIN_QVOL),
                picks=picks, sigs=sigs, btc=btc)


def render(r):
    e = html.escape
    L = []
    L.append(f"<b>📊 Crypto scan — {r['now'].strftime('%d %b %Y, %H:%M')} IST</b>")
    if r["btc"]:
        L.append(f"BTC {fmt_px(r['btc'][2])} ({r['btc'][3]:+.1f}% 24h) · universe {r['n_uni']} pairs ({r['n_liquid']} with ≥$2M volume)")
    L.append("")
    L.append("<b>🎯 Daily picker — coins most likely to move today</b>")
    L.append("<i>Ranks expected RANGE (validated), not direction.</i>")
    for i, p in enumerate(r["picks"], 1):
        pct = p["today_rng"] / p["adr"] * 100 if p["adr"] else 0
        flag = " ⚠️ already moved ≥ normal range — don't chase" if pct >= 100 else ""
        L.append(f"{i}. <b>{e(p['sym'])}</b> {fmt_px(p['last'])} | normal range {p['adr']:.1f}%/day | "
                 f"today {p['today_rng']:.1f}% ({pct:.0f}% of normal), {p['today_move']:+.1f}% from open | "
                 f"yday L {fmt_px(p['prev_low'])} / H {fmt_px(p['prev_high'])} | 20d L {fmt_px(p['low20'])} / H {fmt_px(p['high20'])} | "
                 f"trend {p['trend']}" + (" | abnormal last day" if p['abnormal'] else "") + flag)
    L.append("")
    tf_names = {"4h": "4H", "1h": "1H", "5m": "5m"}
    for tf in ("4h", "1h", "5m"):
        s = sorted([x for x in r["sigs"] if x["tf"] == tf], key=lambda x: x["when"], reverse=True)
        window = f"last {FIVE_MIN_FRESH_MIN} min" if tf == "5m" else f"since {r['wstart'].strftime('%d %b %H:%M')}"
        L.append(f"<b>📈 Breakouts {tf_names[tf]}</b> ({window}) — <i>{e(EVIDENCE[tf])}</i>")
        if not s:
            L.append("  none")
        for x in s[:15]:
            icon = "🟢" if x["side"] == "LONG" else "🔴"
            tags = []
            if x["late"]: tags.append("⚠️ past TP1 — late")
            if x["thin"]: tags.append("thin volume")
            if x["status"] != "ACTIVE": tags.append(x["status"])
            L.append(f"{icon} <b>{e(x['sym'])}</b> {x['side']} [{x['strength']}] {x['when'].strftime('%d %b %H:%M')} | "
                     f"entry {fmt_px(x['entry'])} SL {fmt_px(x['sl'])} (−{x['risk']:.1f}%) | "
                     f"TP1 {fmt_px(x['tps'][0])} TP2 {fmt_px(x['tps'][1])} TP3 {fmt_px(x['tps'][2])} | "
                     f"R:R TP1 1:{x['rr1']:.2f}, TP3 1:{x['rr3']:.2f} | now {fmt_px(x['last'])} ({x['from_entry']:+.1f}%)"
                     + (f" | {', '.join(tags)}" if tags else ""))
        if len(s) > 15:
            L.append(f"  … +{len(s) - 15} more")
        L.append("")
    L.append("<i>Signals are rule-based candidates, not advice. Fees ~0.2% round trip; risk a fixed small % per trade.</i>")
    return "\n".join(L)


def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN"); chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False, "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3900:
            chunks.append(cur); cur = ""
        cur += line + "\n"
    if cur: chunks.append(cur)
    for c in chunks:
        data = urllib.parse.urlencode({"chat_id": chat, "text": c, "parse_mode": "HTML",
                                       "disable_web_page_preview": "true"}).encode()
        for i in range(3):
            try:
                urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=30)
                break
            except Exception as ex:
                if i == 2:
                    return False, f"telegram error: {type(ex).__name__}"   # never print the token/URL
                time.sleep(2)
    return True, f"sent {len(chunks)} message(s)"


if __name__ == "__main__":
    t0 = time.time()
    res = scan()
    text = render(res)
    os.makedirs("out", exist_ok=True)
    open("out/latest.txt", "w").write(text)
    if "--dry-run" in sys.argv:
        ok, msg = False, "dry run"
    else:
        ok, msg = send_telegram(text)
    print(text)
    print(f"\n[scanner] {len(res['sigs'])} signals, {len(res['picks'])} picks, {time.time() - t0:.0f}s, telegram: {msg}")
