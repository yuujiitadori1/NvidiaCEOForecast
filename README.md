# Polymarket "Next CEO of NVIDIA?" — 4-hour-ahead quantile forecaster

Entry for make.fun competition `polymarket-nvidia-ceo-forecast-01`: a Python 3.13
program (standard library only, no network) that reads one case file
(`/input/<id>.in`) and writes `/output/<id>.out` with, for each of the event's
trading markets and each of the 240 minutes after the checkpoint, five quantiles
(5%, 25%, 50%, 75%, 95%) of that minute's Yes price.

## Run

```sh
python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR` (else `/input`), writes `$MAKEFUN_OUTPUT_DIR`
(else `/output`). One run handles every `*.in` file it finds, in name order.

## Model

Polymarket Yes prices are step functions: about 88% of 1-minute returns are
exactly zero, then a tick, then now and then a jump (in the practice, Colette
Kress went 0.3535 -> 0.59 in one minute). A normal random walk with one sigma
is the wrong shape for that: its inner quantiles are far too wide and its
tails too thin. This entry is empirical and jump-aware:

1. **Empirical move quantiles.** For each market, take the last 1080 minutes
   of prices and, at anchor horizons 1, 2, 3, 5, 8, 13, 21, 34, 55, 90, 150,
   240 minutes, the empirical quantiles of the H-minute moves (overlapping
   windows). This captures the true shape directly: a large atom at zero,
   tiny inner quantiles, fat asymmetric tails.
2. **Pooled prior.** The same quantiles pooled over all markets of the case
   (far more data, far less noise), blended 10% own / 90% pooled. The pooled
   quantiles are scaled by the market's own recent activity: the RMS of its
   1-minute returns over the last 120 minutes, relative to the pooled RMS,
   floored at 0.2 and capped at 1.5 — a market that has not moved in two hours
   still carries jump risk, and one that just jumped should not get an
   unbounded scale.
3. **Interpolation.** Each quantile is interpolated in log(horizon) between
   anchors.
4. **Median = the checkpoint price (persistence).** The empirical median move
   is noise: recent trends mean-revert (the correlation of the last hour's
   move with the next four hours' is about -0.36 on the practice), and forcing
   the median to the last price scores better than any drift. The same goes
   for contrarian drift: tested, it hurts.
5. Clamp to [0, 1], round to 6 decimals, enforce non-decreasing rows while
   preserving the median (a noisy 25%/75% quantile with the wrong sign must
   not drag the median off the checkpoint price).

No randomness, no clock, no dict-order dependence: two runs on the same case
write byte-identical output.

## Validation (practice, scored against the real Polymarket answers)

Answers were fetched from Polymarket's public CLOB API
(`prices-history`, fidelity 1) for all 24 markets over 2026-11-12T06:00Z to
2026-11-13T06:00Z, gridded the house way, and scored exactly like the
harness (mean pinball loss over every case/market/minute/quantile). The
scorer was first validated by reproducing the house baseline's published
practice loss exactly: **0.363589** (median error 0.767503, coverage 0.949642).

| entry | mean pinball loss |
|---|---|
| house baseline | 0.363589 |
| this entry (v1, no scale floor/cap) | 0.279868 |
| this entry (final) | **0.274859** |

Final per-case losses: p00 0.464309, p01 0.301721, p02 0.363937, p03 0.257783,
p04 0.214098, p05 0.110218 — every case beats its baseline
(0.536138, 0.380352, 0.408408, 0.368599, 0.298794, 0.238232). Median error
0.767503, 90% coverage 0.938151. Runtime ~0.21 s per case (limit 8 s);
all outputs valid JSON, right shape, monotone, in [0, 1], under 1 MiB,
byte-identical across runs.

Caveats: the practice is no guide to the final (the rules say so, and the
tuning was kept to a few robust parameters in a flat region rather than the
argmin). If the event is decided before a checkpoint, that case scores
nothing; the program still writes a valid forecast for it.
