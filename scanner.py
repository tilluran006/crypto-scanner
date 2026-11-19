"""Crypto scanner v2 (shorts-first): 4H BPS breakdown signals (primary plan) + 4H breakouts (info only)
+ a volume-surge list (where the action is), across the top-200 Binance USDT spot pairs.
Deterministic: same candles in -> same list out. No trades are placed. No API keys needed for data.

Why shorts first (backtest 19 Nov 2026, 4H, 22 coins incl. mid/small/micro caps, Nov 2020 - Nov 2026, 0.1% fees):
  shorts +0.11R/trade (1,052 trades) vs random shorts with the same stop/target sizes -0.01R (edge t ~3.0);
  with max 3 open at once: +0.13R/trade, ~7 trades/month, worst drawdown -17R, longest losing streak 10.
  Longs on the same pool: +0.02R (nothing). Breakout-bar volume and build-up volume did NOT improve results.
  Last 12 months were weak (shorts ~+0.06R). Unproven, small edge.

Usage:  python scanner.py            (sends to Telegram if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set,
                                       otherwise prints and writes out/latest.txt)
        python scanner.py --dry-run  (never sends)
"""
import json, os, sys, time, html, statistics, datetime as dt, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from bps import signals as bps_signals

BASE = "https://data-api.binance.vision/api/v3"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SLOTS_IST = [8, 14, 20]                       # scheduled scan hours (IST)
UNIVERSE_N = 200
MIN_QVOL = 2_000_000                          # 24h quote volume below this is marked "thin"
BARS_4H = 500
RISK_PCT = 1.0                                # account risk per trade, for the size column
MAX_OPEN = 3
SURGE_N = 10

LARGE_CAPS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT",
              "LTCUSDT", "AVAXUSDT", "DOTUSDT", "TRXUSDT", "TONUSDT", "BCHUSDT", "SUIUSDT", "NEARUSDT"}
EVID_SHORT = "backtest 2020-26: +0.13R/trade with max 3 open, beat random shorts (t~2); last 12m weak. Unproven."
EVID_LONG = "backtest 2020-26: ~0R/trade (+0.02R) - info only, no edge"
EVID_SURGE = "untested - shows where volume is, not direction"

STABLE = {"USDC","FDUSD","TUSD","DAI","USDP","EUR","EURI","GBP","AUD","BRL","TRY","USD1","USDE","XUSD",
          "AEUR","PAXG","XAUT","WBTC","WBETH","BUSD","UST","USTC","FRAX","LUSD","GUSD","SUSD","PYUSD",
          "RLUSD","BFUSD","USDS","USDY","RWUSD","U"}


def get(path, retries=4):
    url = BASE + path
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "crypto-scanner/2.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def universe():
    info = get("/exchangeInfo?permissions=SPOT")
    # Binance tokenized stocks/ETFs carry permission group TRD_GRP_261 - exclude them
    trading = {s["symbol"]: s["baseAsset"] for s in info["symbols"]
               if s["status"] == "TRADING" and s["quoteAsset"] == "USDT"
               and not any("TRD_GRP_261" in ps for ps in s.get("permissionSets", []))}
    rows = []
    for t in get("/ticker/24hr"):
        sym = t["symbol"]; base = trading.get(sym)
        if not base or not sym.isascii() or base in STABLE: continue
        if any(base.endswith(x) and len(base) > len(x) + 1 for x in ("UP", "DOWN", "BULL", "BEAR")): continue
        rows.append((sym, float(t["quoteVolume"]), float(t["lastPrice"]), float(t["priceChangePercent"])))
    rows.sort(key=lambda r: -r[1])
    return rows[:UNIVERSE_N]


def klines(sym, interval, limit):
    k = get(f"/klines?symbol={urllib.parse.quote(sym)}&interval={interval}&limit={limit}")
    now_ms = int(time.time() * 1000)
    k = [r for r in k if int(r[6]) < now_ms]          # closed bars only
    if not k: return None
    a = np.array([[float(r[i]) for i in (0, 1, 2, 3, 4, 5, 7)] for r in k])   # col 5 base vol (Pine 'volume'), col 6 quote vol
    return a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6]


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


def surge_row(sym):
    """Last 24h / 4h quote volume vs the coin's own normal (previous 14 days / previous 7 days of 4h blocks)."""
    try:
        k = klines(sym, "1h", 361)
    except Exception:
        return None
    if k is None or len(k[0]) < 360: return None
    T, O, H, L, C, V, Q = (x[-360:] for x in k)
    v24 = Q[-24:].sum()
    base24 = statistics.median([Q[-24 * (i + 1) - 24:-24 * (i + 1)].sum() for i in range(14)])
    if v24 < MIN_QVOL or base24 <= 0: return None     # must be liquid NOW
    base4 = statistics.median([Q[-4 * (i + 1) - 4:-4 * (i + 1)].sum() for i in range(42)])
    hi7, lo7 = H[-168:].max(), L[-168:].min()
    return dict(sym=sym, rvol24=v24 / base24, rvol4=Q[-4:].sum() / base4 if base4 else 0.0,
                chg24=(C[-1] / C[-25] - 1) * 100, last=float(C[-1]),
                pos7=(C[-1] - lo7) / (hi7 - lo7) * 100 if hi7 > lo7 else 50.0)


def scan():
    now = dt.datetime.now(IST)
    wstart = window_start(now)
    uni = universe()
    syms = [u[0] for u in uni]
    meta = {u[0]: u for u in uni}

    def tf_signals(sym):
        try:
            k = klines(sym, "4h", BARS_4H)
        except Exception:
            return []
        if k is None: return []
        T, O, H, L, C, V, Q = k
        out = []
        for s in bps_signals(T, O, H, L, C, V):
            close_ist = dt.datetime.fromtimestamp((s["t"] + 4 * 3600e3) / 1000, IST)
            if close_ist < wstart: continue
            if s["status"] in ("SL HIT", "TP3 HIT", "BE EXIT", "TIMEOUT"): continue
            last = float(C[-1]); e = s["entry"]; side = s["side"]
            risk = abs(e - s["sl0"]) / e * 100
            out.append(dict(sym=sym, side="LONG" if side > 0 else "SHORT", when=close_ist,
                            entry=e, sl=s["sl0"], tps=s["tps"], risk=risk, last=last,
                            rr=[abs(t - e) / abs(e - s["sl0"]) if e != s["sl0"] else 0 for t in s["tps"]],
                            status=s["status"], strength=s["strength"],
                            late=(last - s["tps"][0]) * side >= 0,                  # already past TP1 -> don't chase
                            beyond_sl=(last - s["sl0"]) * side <= 0,
                            size=RISK_PCT / risk * 100 if risk else 0,              # position notional, % of account
                            from_entry=(last / e - 1) * 100 * side,
                            large=sym in LARGE_CAPS, thin=meta[sym][1] < MIN_QVOL))
        return out

    with ThreadPoolExecutor(8) as ex:
        sigs = [x for lst in ex.map(tf_signals, syms) for x in lst]
        surge = sorted([r for r in ex.map(surge_row, syms) if r], key=lambda r: -r["rvol24"])[:SURGE_N]
    return dict(now=now, wstart=wstart, n_uni=len(uni), n_liquid=sum(1 for u in uni if u[1] >= MIN_QVOL),
                sigs=sigs, surge=surge, btc=meta.get("BTCUSDT"))


def render(r):
    e = html.escape
    L = [f"<b>📊 Crypto scan — {r['now'].strftime('%d %b %Y, %H:%M')} IST</b>"]
    if r["btc"]:
        L.append(f"BTC {fmt_px(r['btc'][2])} ({r['btc'][3]:+.1f}% 24h) · universe {r['n_uni']} pairs ({r['n_liquid']} with ≥$2M volume)")
    L.append("")
    since = r["wstart"].strftime("%d %b %H:%M")

    def sig_line(x):
        icon = "🔴" if x["side"] == "SHORT" else "🟢"
        tags = []
        if x["large"]: tags.append("tested large cap")
        if x["late"]: tags.append("⚠️ past TP1 — late, don't chase")
        if x["beyond_sl"]: tags.append("⚠️ price beyond SL — invalid")
        if x["thin"]: tags.append("thin volume")
        if x["status"] != "ACTIVE": tags.append(x["status"])
        return (f"{icon} <b>{e(x['sym'])}</b> {x['side']} [{x['strength']}] closed {x['when'].strftime('%d %b %H:%M')} | "
                f"entry ≈ next 4H open (signal close {fmt_px(x['entry'])}) | SL {fmt_px(x['sl'])} (−{x['risk']:.1f}%) | "
                f"TP1 {fmt_px(x['tps'][0])} ({x['rr'][0]:.1f}R) TP2 {fmt_px(x['tps'][1])} ({x['rr'][1]:.1f}R) "
                f"TP3 {fmt_px(x['tps'][2])} ({x['rr'][2]:.1f}R) | size for {RISK_PCT:.0f}% risk ≈ {x['size']:.0f}% of account | "
                f"now {fmt_px(x['last'])} ({x['from_entry']:+.1f}%)" + (f" | {', '.join(tags)}" if tags else ""))

    order = lambda x: (x["late"] or x["beyond_sl"], not x["large"], -x["when"].timestamp())
    shorts = sorted([x for x in r["sigs"] if x["side"] == "SHORT"], key=order)
    longs = sorted([x for x in r["sigs"] if x["side"] == "LONG"], key=order)

    L.append(f"<b>🔴 4H breakdowns — SHORT (primary plan)</b> (since {since})")
    L.append(f"<i>{e(EVID_SHORT)}</i>")
    L.append(f"<i>Plan: short at next 4H open, SL as given, ⅓ off at each TP, SL → entry after TP1. "
             f"Max {MAX_OPEN} open at once — skip if your slots are full.</i>")
    L += [sig_line(x) for x in shorts[:15]] or ["  none"]
    if len(shorts) > 15: L.append(f"  … +{len(shorts) - 15} more")
    L.append("")

    L.append(f"<b>🟢 4H breakouts — LONG</b> (since {since}) — <i>{e(EVID_LONG)}</i>")
    L += [sig_line(x) for x in longs[:8]] or ["  none"]
    if len(longs) > 8: L.append(f"  … +{len(longs) - 8} more")
    L.append("")

    L.append(f"<b>🔊 Volume surge — trading far above their own normal</b> — <i>{e(EVID_SURGE)}</i>")
    for i, s in enumerate(r["surge"], 1):
        accel = " ↑ still accelerating" if s["rvol4"] > s["rvol24"] else ""
        L.append(f"{i}. <b>{e(s['sym'])}</b> {fmt_px(s['last'])} | 24h vol {s['rvol24']:.1f}x normal, last 4h {s['rvol4']:.1f}x{accel} | "
                 f"{s['chg24']:+.1f}% 24h | {s['pos7']:.0f}% of 7-day range")
    if not r["surge"]: L.append("  none")
    L.append("")
    L.append("<i>Rule-based candidates, not advice. Spot data; funding and squeezes not modelled. "
             f"Risk a fixed {RISK_PCT:.0f}% per trade.</i>")
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
    ok, msg = (False, "dry run") if "--dry-run" in sys.argv else send_telegram(text)
    print(text)
    ns = sum(1 for x in res["sigs"] if x["side"] == "SHORT"); nl = len(res["sigs"]) - ns
    print(f"\n[scanner] 4H shorts {ns}, 4H longs {nl}, surge {len(res['surge'])}, {time.time() - t0:.0f}s, telegram: {msg}")
