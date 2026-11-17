# crypto-scanner

Three-times-daily Binance scan (08:00, 14:00, 20:00 IST) sent to Telegram. Research tool, not advice. No trades are placed and no exchange API keys are used.

## What each message contains

1. **Daily picker (top 10).** These are the liquid coins (≥ $2M 24h volume) with the largest normal daily range, measured as the 14-day mean of (high − low) / open. Each line shows:
   - today's range so far and how much of a normal day that is
   - move since the UTC open
   - yesterday's high and low
   - whether price is above or below its 20-day average
   - a ⚠️ no-chase flag if the coin has already moved a full normal range

   Validated in research: it predicts how much a coin will move (rank correlation ≈ +0.6), not which direction.
2. **Breakout / breakdown signals** from a deterministic port of *Breakout Pattern Setup [WillyAlgoTrader] v1.7.2* with its default inputs, run on 4H, 1H and 5m across the top 200 USDT pairs by volume. Stablecoins, leveraged tokens and Binance stock tokens are excluded. Each signal gives:
   - entry (breakout close) and SL (the opposite channel line)
   - TP1, TP2 and TP3 at ⅓, ⅔ and 1× the measured move
   - R:R to TP1 and to TP3
   - strength label, live status (ACTIVE / TP1 HIT …), current price, and tags for "late" (already past TP1) and "thin volume"
   - Freshness: 4H and 1H show signals since the previous scheduled scan; 5m shows only the last 60 minutes.

## Evidence (from the research, 0.2% fees)

| Timeframe | Backtest result |
|---|---|
| 4H | +0.08R per trade over 1,489 trades on 11 coins and 9 years. Unproven: date-clustered t = 1.7, last 2 years ≈ +0.04R |
| 1H | Not tested |
| 5m | −0.3 to −0.7R per trade. Fees exceed any edge |

## Verification

- The live port reproduces the backtest port exactly (ETH, BTC and SOL 4H: identical signal lists).
- It reproduces a TradingView chart exactly: ENAUSDT 5m, 17 Nov 2026 08:30 IST, Long, entry 0.1145, SL 0.1129, TP 0.1154 / 0.1163 / 0.1173. Pivot tie rule: ≥ left, > right.

## Run

```
pip install numpy requests
python scanner.py --dry-run     # print only
python scanner.py               # send if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set
```

Secrets are read from environment variables only. They are never written to files, logs or this repo.

Network access needed: `data-api.binance.vision`, `api.telegram.org`.
