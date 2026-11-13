"""polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).

Forecasts each market's Yes price at every minute of the next 4 hours as five
quantiles, from the minute-by-minute prices up to the checkpoint.

Model (empirical, jump-aware -- Polymarket prices are step functions: ~88% of
1-minute returns are exactly zero, then a tick, then now and then a jump, so a
normal random walk with a single sigma is the wrong shape: its inner quantiles
are far too wide and its tails too thin):

  * For each market, take the last W = 1080 minutes of prices and, at anchor
    horizons H = 1, 2, 3, 5, 8, 13, 21, 34, 55, 90, 150, 240, the empirical
    quantiles of the H-minute moves (overlapping windows).
  * Do the same pooled over all markets of the case (far more data, far less
    noise), and blend: 10% the market's own quantiles, 90% the pooled
    quantiles scaled by the market's own recent activity -- the RMS of its
    1-minute returns over the last 120 minutes, relative to the pooled RMS,
    floored at 0.2 and capped at 1.5 (a dead book can still reprice in a
    minute; a market that just jumped should not get an unbounded scale).
  * Interpolate each quantile in log(horizon) between anchors.
  * The median is the checkpoint price itself (persistence): the empirical
    median move is noise -- recent trends mean-revert, and forcing the median
    to the last price scores better on the practice than any drift.
  * Clamp to [0, 1], round to 6 decimals, enforce non-decreasing rows.

No network, no randomness, no time dependence: two runs on the same case
write byte-identical output. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Anchor horizons (minutes) at which empirical move quantiles are measured.
ANCHORS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 90, 150, 240]
# Minutes of history used for the empirical move quantiles.
WINDOW = 1080
# Weight of a market's own quantiles vs the pooled (cross-market) ones.
OWN_WEIGHT = 0.10
# Minutes of history used for a market's recent-activity scale.
SCALE_WINDOW = 120
# A market that has not moved at all in the last 2 hours still carries jump
# risk (a dead book can reprice in a minute), and a market that just jumped
# should not get an unbounded scale: floor and cap the activity ratio. The
# practice score improves with both (0.2799 -> 0.2750).
SCALE_FLOOR = 0.2
SCALE_CAP = 1.5


def quantile_sorted(s, q):
    """q-quantile of a sorted list, linear interpolation (deterministic)."""
    n = len(s)
    if n == 0:
        return 0.0
    i = q * (n - 1)
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def anchor_quantiles(moves_by_h):
    """Empirical quantiles of the moves at each anchor horizon."""
    return {
        h: [quantile_sorted(sorted(mv), q) for q in QUANTILES]
        for h, mv in moves_by_h.items()
    }


def interp_quantiles(aq, h):
    """Interpolate each quantile in log(h) between the anchor horizons."""
    if h <= ANCHORS[0]:
        return list(aq[ANCHORS[0]])
    if h >= ANCHORS[-1]:
        return list(aq[ANCHORS[-1]])
    for i in range(len(ANCHORS) - 1):
        a, b = ANCHORS[i], ANCHORS[i + 1]
        if a <= h <= b:
            t = (math.log(h) - math.log(a)) / (math.log(b) - math.log(a))
            return [aq[a][j] + (aq[b][j] - aq[a][j]) * t for j in range(5)]
    return list(aq[ANCHORS[-1]])


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1000000.0) / 1000000.0 + 0.0


def forecast_case(case):
    """Forecast one case: list of {"id": ..., "q": [[5 numbers] x horizon]}."""
    horizon = int(case.get("horizon_minutes", 240))
    markets = case["markets"]

    # Per-market and pooled H-minute moves over the last WINDOW minutes.
    pooled = {h: [] for h in ANCHORS}
    own_moves = []
    scales = []
    for m in markets:
        p = m["prices"]
        pw = p[-WINDOW:] if len(p) > WINDOW else p
        n = len(pw)
        om = {}
        for h in ANCHORS:
            mv = [pw[i + h] - pw[i] for i in range(n - h)] if n > h else [0.0]
            om[h] = mv
            pooled[h].extend(mv)
        own_moves.append(om)
        sw = p[-SCALE_WINDOW:] if len(p) > SCALE_WINDOW else p
        rets = [sw[i + 1] - sw[i] for i in range(len(sw) - 1)]
        scales.append(math.sqrt(sum(r * r for r in rets) / max(1, len(rets))))
    pooled_scale = math.sqrt(sum(s * s for s in scales) / max(1, len(scales)))
    pooled_aq = anchor_quantiles(pooled)
    own_aq = [anchor_quantiles(om) for om in own_moves]

    forecasts = []
    for mi, m in enumerate(markets):
        p = m["prices"]
        x = p[-1] if p else 0.5
        sf = scales[mi] / pooled_scale if pooled_scale > 0 else 1.0
        sf = min(SCALE_CAP, max(SCALE_FLOOR, sf))
        rows = []
        for h in range(1, horizon + 1):
            qo = interp_quantiles(own_aq[mi], h)
            qp = interp_quantiles(pooled_aq, h)
            row = []
            for j in range(5):
                if j == 2:
                    # Median: persistence at the checkpoint price.
                    qv = 0.0
                else:
                    qv = OWN_WEIGHT * qo[j] + (1.0 - OWN_WEIGHT) * qp[j] * sf
                row.append(price6(x + qv))
            # Enforce non-decreasing within the row, preserving the median
            # (the most important quantile): a noisy 25%/75% quantile whose
            # empirical sign is wrong must not drag the median off the
            # checkpoint price. Clamp the lower quantiles down to the median
            # and the upper quantiles up to it instead.
            if row[1] > row[2]:
                row[1] = row[2]
            if row[0] > row[1]:
                row[0] = row[1]
            if row[3] < row[2]:
                row[3] = row[2]
            if row[4] < row[3]:
                row[4] = row[3]
            rows.append(row)
        forecasts.append({"id": m["id"], "q": rows})
    return forecasts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        cid = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        forecasts = forecast_case(case)
        out = json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n"
        with open(os.path.join(OUTPUT_DIR, cid + ".out"), "w", encoding="utf-8") as f:
            f.write(out)


if __name__ == "__main__":
    main()
