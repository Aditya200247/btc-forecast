import json
import time
import numpy as np
from model import fetch_klines, predict_range, evaluate


SYMBOL        = "BTCUSDT"
INTERVAL      = "1h"
N_PREDICTIONS = 720
WARMUP        = 200          # extra history stabilises EWMA + empirical correction
FETCH_LIMIT   = N_PREDICTIONS + WARMUP + 10
OUTPUT_FILE   = "backtest_results.jsonl"


def run_backtest() -> dict:
    print(f"Fetching {FETCH_LIMIT} bars of {SYMBOL} {INTERVAL} data ...")
    timestamps, closes = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=FETCH_LIMIT)
    print(f"Got {len(closes)} bars.")

    predictions = []
    available   = len(closes) - WARMUP - 1
    n           = min(N_PREDICTIONS, available)

    print(f"Running walk-forward backtest on {n} bars (no peeking!) ...\n")

    for i in range(n):
        bar_idx    = WARMUP + i         # the "current" bar we predict FROM
        target_idx = bar_idx + 1        # the bar we're predicting

        # No peeking: only pass data up to (and including) bar_idx
        history      = closes[: bar_idx + 1]
        actual_price = closes[target_idx]

        result = predict_range(history)
        if result is None:
            continue

        lower, upper = result
        inside = lower <= actual_price <= upper

        predictions.append({
            "timestamp":     timestamps[bar_idx],
            "current_price": closes[bar_idx],
            "lower_95":      round(lower, 2),
            "upper_95":      round(upper, 2),
            "actual_price":  actual_price,
            "inside":        inside,
            "width":         round(upper - lower, 2),
        })

        if (i + 1) % 100 == 0 or (i + 1) == n:
            cov_so_far = np.mean([p["inside"] for p in predictions])
            print(f"  {i+1:4d}/{n}  coverage so far: {cov_so_far:.3f}")

    with open(OUTPUT_FILE, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"\nSaved {len(predictions)} predictions -> {OUTPUT_FILE}")

    metrics = evaluate(predictions)

    print("\n" + "=" * 45)
    print("  BACKTEST RESULTS")
    print("=" * 45)
    print(f"  Predictions:        {metrics['n']}")
    print(f"  Coverage (target 0.95):  {metrics['coverage_95']:.4f}")
    print(f"  Avg width:          ${metrics['mean_width']:,.2f}")
    print(f"  Mean Winkler score: {metrics['mean_winkler_95']:,.2f}  (lower = better)")
    print("=" * 45)

    return metrics


if __name__ == "__main__":
    t0 = time.time()
    metrics = run_backtest()
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"\nFor submission form:")
    print(f"  coverage_95      = {metrics['coverage_95']:.4f}")
    print(f"  mean_winkler_95  = {metrics['mean_winkler_95']:.2f}")
