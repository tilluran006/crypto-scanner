"""Daily crypto picker: which liquid coins are most likely to MOVE tomorrow (range, not direction).

Ranks tradable Binance USDT coins by their typical daily range over the last 14 closed days. In a 2-year
backtest that ranking predicted the next day's range well (rank correlation about +0.6 both years) but not
the next day's direction. So the output is a shortlist with context and key levels; direction is your call.

Usage: python day_picker.py [--top 200] [--min-vol 2e6] [--show 10] [--symbols BTC,ETH]
Not financial advice.
"""
import argparse, concurrent.futures as cf, datetime as dt, json, statistics, sys, time
import requests

SPOT = "https://data-api.binance.vision"
GECKO = "https://api.coingecko.com/api/v3/coins/markets"
STABLES = {"USDT", "USDC", "FDUSD", "TUSD", "USDP", "DAI", "USDE", "PYUSD", "BUSD", "EUR", "EURI", "AEUR", "USD1",
           "RLUSD", "XUSD", "BFUSD", "USDS", "GUSD", "FRAX", "LUSD", "USDD", "USDG", "USDF", "USD0", "EURC", "USDY",
           "OUSD", "GHO", "U", "STABLE"}
NON_CRYPTO = {"PAXG", "XAUT", "KAU", "KAG", "WBTC", "WETH", "STETH", "WSTETH", "WEETH", "CBBTC", "WBETH", "BETH",
              "RETH", "METH", "BNSOL", "JITOSOL", "LBTC", "SOLVBTC", "SUSDE", "SUSDS", "USDT0"}
BAD_NAME = ("wrapped", "staked", "bridged", "usd", "gold", "fund", "treasury", "tokenized", "xstock", "bstock",
            " etf", "preferred")
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
S = requests.Session()


def get(url, **params):
    for attempt in range(5):
        r = S.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429):
            time.sleep(int(r.headers.get("Retry-After", 0) or 10 * (attempt + 1)))
            continue
        time.sleep(1 + attempt)
    r.raise_for_status()


def features(bars):
    """bars: closed daily bars, oldest first, each (open, high, low, close, quote_volume); needs >= 51.
    Everything uses only these bars, so the backtest and the live run compute the same numbers."""
    o = [b[0] for b in bars]; h = [b[1] for b in bars]; l = [b[2] for b in bars]
    c = [b[3] for b in bars]; qv = [b[4] for b in bars]
    rng = [(h[i] - l[i]) / o[i] for i in range(len(bars))]
    atr14 = sum(rng[-14:]) / 14                                   # typical daily range, % of price
    rvol = qv[-1] / statistics.median(qv[-21:-1])                 # last day's volume vs its 20-day median
    ma20, ma50 = sum(c[-20:]) / 20, sum(c[-50:]) / 50
    return dict(atr14=atr14, range_last=rng[-1], rvol=rvol, vol7=statistics.median(qv[-7:]),
                close=c[-1], ma20=ma20, ma50=ma50,
                trend=("up" if c[-1] > ma20 > ma50 else "down" if c[-1] < ma20 < ma50 else "mixed"),
                prev_high=h[-1], prev_low=l[-1], high20=max(h[-20:]), low20=min(l[-20:]),
                abnormal=rng[-1] > 4 * statistics.median(rng[-30:]))


def rank(rows):
    """Sort by expected range (typical daily range), largest first. Abnormal one-day spikes go last."""
    return sorted(rows, key=lambda r: (r["abnormal"], -r["atr14"]))


def universe(top, max_spread=0.0005):
    info = get(SPOT + "/api/v3/exchangeInfo")["symbols"]
    syms = {s["symbol"]: s for s in info if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"}
    tick = {s: float(next((f["tickSize"] for f in v["filters"] if f["filterType"] == "PRICE_FILTER"), 0)) for s, v in syms.items()}
    book = {b["symbol"]: b for b in get(SPOT + "/api/v3/ticker/bookTicker")}
    t24 = {t["symbol"]: t for t in get(SPOT + "/api/v3/ticker/24hr")}
    coins = []
    for page in range(1, min(4, (int(top * 1.6) - 1) // 250 + 2) + 1):
        coins += get(GECKO, vs_currency="usd", order="market_cap_desc", per_page=250, page=page)
        time.sleep(2)
    out, dropped, seen = [], {"not on Binance": 0, "stable/wrapped/stock": 0, "spread": 0}, set()
    for c in coins:
        sym, name = c["symbol"].upper(), c["name"].lower()
        if sym in STABLES or sym in NON_CRYPTO or any(w in name for w in BAD_NAME) or \
                (c.get("current_price") and abs(c["current_price"] - 1) < 0.03):
            dropped["stable/wrapped/stock"] += 1; continue
        pair = sym + "USDT"
        if pair not in syms or pair in seen or pair not in book or pair not in t24:
            dropped["not on Binance"] += 1; continue
        ask, bid = float(book[pair]["askPrice"]), float(book[pair]["bidPrice"])
        if ask <= 0: continue
        spread = 0.0 if ask - bid <= tick[pair] * 1.01 else (ask - bid) / ask
        if spread > max_spread:
            dropped["spread"] += 1; continue
        seen.add(pair)
        out.append(dict(symbol=pair, price=float(t24[pair]["lastPrice"]), chg24=float(t24[pair]["priceChangePercent"]),
                        spread=spread))
        if len(out) >= top: break
    return out, dropped


def daily_bars(sym, n=60):
    raw = get(SPOT + "/api/v3/klines", symbol=sym, interval="1d", limit=n + 1)
    now = time.time() * 1000
    return [(float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[7])) for b in raw if b[6] < now], \
        (max(b[6] for b in raw if b[6] < now) + 1 if raw else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--min-vol", type=float, default=2e6)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--symbols")
    a = ap.parse_args()
    run = dt.datetime.now(dt.timezone.utc)
    if a.symbols:
        t24 = {t["symbol"]: t for t in get(SPOT + "/api/v3/ticker/24hr")}
        uni = []
        for s in a.symbols.split(","):
            p = s.strip().upper(); p = p if p.endswith("USDT") else p + "USDT"
            uni.append(dict(symbol=p, price=float(t24[p]["lastPrice"]), chg24=float(t24[p]["priceChangePercent"]), spread=None))
        dropped = {}
    else:
        uni, dropped = universe(min(a.top, 500))

    def work(u):
        try:
            bars, last_close = daily_bars(u["symbol"])
            if len(bars) < 51: return None, "under 51 days of history"
            f = features(bars[-60:])
            if f["vol7"] < a.min_vol: return None, "low volume"
            return dict(u, **f, last_close_ms=last_close), None
        except Exception as e:  # noqa: BLE001
            return None, f"error {type(e).__name__}"

    bb, _ = daily_bars("BTCUSDT")
    btc = features(bb[-60:])
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(work, uni))
    rows = [r for r, _ in res if r]
    skipped = {}
    for _, why in res:
        if why: skipped[why] = skipped.get(why, 0) + 1
    picks = rank(rows)
    bar = dt.datetime.fromtimestamp(max(r["last_close_ms"] for r in rows) / 1000, dt.timezone.utc) if rows else None

    print(f"\nDaily picker | run {run:%Y-%m-%d %H:%M} UTC / {run.astimezone(IST):%H:%M} IST"
          + (f" | based on daily candles to {bar:%d %b %H:%M} UTC ({bar.astimezone(IST):%H:%M} IST)" if bar else ""))
    print(f"Universe {len(uni)} (dropped: {dropped}) | ranked {len(rows)} | skipped {skipped}")
    print(f"BTC: {btc['close']:.0f}, trend {btc['trend']} (close vs 20/50-day averages), "
          f"typical daily range {100 * btc['atr14']:.1f}%\n")
    print("Ranked by EXPECTED RANGE (typical daily range, last 14 days). It predicts how much a coin moves, not which way.\n")
    hdr = f"{'#':>2} {'symbol':12} {'price':>11} {'24h%':>6} {'exp range':>9} {'vol $M':>7} {'vol x':>6} {'trend':>6}  {'prev day low-high':>23}  {'20-day low-high':>23}"
    print(hdr)
    for i, r in enumerate(picks[: a.show], 1):
        fmt = lambda v: f"{v:.6g}"
        print(f"{i:>2} {r['symbol']:12} {fmt(r['price']):>11} {r['chg24']:>+6.1f} {100 * r['atr14']:>8.1f}% {r['vol7'] / 1e6:>7.1f} "
              f"{r['rvol']:>5.1f}x {r['trend']:>6}  {fmt(r['prev_low']) + ' - ' + fmt(r['prev_high']):>23}  "
              f"{fmt(r['low20']) + ' - ' + fmt(r['high20']):>23}" + ("  (abnormal last day)" if r["abnormal"] else ""))
    print("\nexp range = typical daily high-low range as % of price | vol x = last day's volume vs its 20-day median"
          "\ntrend = close vs 20- and 50-day averages (up: above both, rising order; down: below both; else mixed)")
    fn = f"day_picks_{run:%Y%m%d_%H%M}.json"
    with open(fn, "w") as fh:
        json.dump(dict(as_of=run.isoformat(), candles_to=bar.isoformat() if bar else None, btc_trend=btc["trend"],
                       picks=[{k: v for k, v in r.items() if k not in ("last_close_ms",)} for r in picks]), fh, indent=1, default=str)
    print(f"\nSaved {fn}. A shortlist of coins likely to move, not trade signals. Not financial advice.")


if __name__ == "__main__":
    sys.exit(main())
