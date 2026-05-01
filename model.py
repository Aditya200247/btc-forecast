import numpy as np
from scipy import stats
import requests


def fetch_klines(symbol="BTCUSDT", interval="1h", limit=500):
    url = "https://data-api.binance.vision/api/v3/klines"
    resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    resp.raise_for_status()
    rows = resp.json()
    timestamps = [int(r[0]) for r in rows]
    closes = [float(r[4]) for r in rows]
    return timestamps, closes


def predict_range(closes, lookback=168, vol_window=24, n_sims=10000, confidence=0.95, seed=None):
    prices = np.asarray(closes[-lookback:] if len(closes) >= lookback else closes, dtype=float)

    if len(prices) < 10:
        return None

    log_ret = np.diff(np.log(prices))

    df, loc, _ = stats.t.fit(log_ret)
    df = max(df, 2.01)

    recent = log_ret[-vol_window:] if len(log_ret) >= vol_window else log_ret
    recent_std = np.std(recent, ddof=1)

    # scale simulated vol to match recent window (volatility clustering)
    t_scale = recent_std * np.sqrt((df - 2) / df)

    rng = np.random.default_rng(seed)
    sim_returns = stats.t.rvs(df, loc=loc, scale=t_scale, size=n_sims, random_state=rng)
    sim_prices = prices[-1] * np.exp(sim_returns)

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(sim_prices, alpha * 100))
    upper = float(np.percentile(sim_prices, (1 - alpha) * 100))
    return lower, upper


def winkler_score(lower, upper, actual, alpha=0.05):
    width = upper - lower
    if lower <= actual <= upper:
        return width
    elif actual < lower:
        return width + (2 / alpha) * (lower - actual)
    else:
        return width + (2 / alpha) * (actual - upper)


def evaluate(predictions):
    inside = [p["lower_95"] <= p["actual_price"] <= p["upper_95"] for p in predictions]
    widths = [p["upper_95"] - p["lower_95"] for p in predictions]
    winkler = [winkler_score(p["lower_95"], p["upper_95"], p["actual_price"]) for p in predictions]

    return {
        "coverage_95": float(np.mean(inside)),
        "mean_width": float(np.mean(widths)),
        "mean_winkler_95": float(np.mean(winkler)),
        "n": len(predictions),
    }
