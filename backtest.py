import json
import time
import numpy as np
from model import fetch_klines, predict_range, evaluate

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
N_PREDICTIONS = 720
WARMUP = 168
OUTPUT_FILE = "backtest_results.jsonl"


def run_backtest():
    timestamps, closes = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=N_PREDICTIONS + WARMUP + 10)
    print(f"Fetched {len(closes)} bars\n")

    predictions = []
    n = min(N_PREDICTIONS, len(closes) - WARMUP - 1)

    for i in range(n):
        bar_idx = WARMUP + i
        history = closes[:bar_idx + 1]
        actual_price = closes[bar_idx + 1]

        result = predict_range(history)
        if result is None:
            continue

        lower, upper = result
        predictions.append({
            "timestamp": timestamps[bar_idx],
            "current_price": closes[bar_idx],
            "lower_95": round(lower, 2),
            "upper_95": round(upper, 2),
            "actual_price": actual_price,
            "inside": lower <= actual_price <= upper,
            "width": round(upper - lower, 2),
        })

        if (i + 1) % 100 == 0 or (i + 1) == n:
            cov = np.mean([p["inside"] for p in predictions])
            print(f"  {i+1}/{n}  coverage: {cov:.3f}")

    with open(OUTPUT_FILE, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    metrics = evaluate(predictions)

    print(f"\n{'='*40}")
    print(f"  Predictions : {metrics['n']}")
    print(f"  Coverage    : {metrics['coverage_95']:.4f}  (target 0.95)")
    print(f"  Avg width   : ${metrics['mean_width']:,.2f}")
    print(f"  Winkler     : {metrics['mean_winkler_95']:,.2f}")
    print(f"{'='*40}")

    return metrics


if __name__ == "__main__":
    t0 = time.time()
    m = run_backtest()
    print(f"\nFinished in {time.time() - t0:.1f}s")
    print(f"\nSubmission numbers:")
    print(f"  coverage_95     = {m['coverage_95']:.4f}")
    print(f"  mean_winkler_95 = {m['mean_winkler_95']:.2f}")
