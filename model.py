"""
model.py -- GBM + Student-t forecaster for BTCUSDT 1-hour bars.

Improvements over the naive baseline:
  1. EWMA volatility  -- captures volatility clustering (calm->narrow, volatile->wide)
  2. Correct t-scale  -- stats.t.fit() MLE scale used directly, no re-normalisation
  3. Empirical tail correction -- partial (40%) adjustment from observed quantile ratios
     so coverage stays near the nominal 95% level even when the t-distribution
     under-estimates BTC fat tails
  4. Vol floor  -- prevents absurdly narrow ranges in calm-regime patches
"""

import numpy as np
from scipy import stats
import requests


# -- Data fetching -------------------------------------------------------------

def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500):
    """
    Pull `limit` closed 1-hour bars from Binance's geo-unrestricted mirror.
    Returns (timestamps_ms_list, close_prices_list).
    """
    url = "https://data-api.binance.vision/api/v3/klines"
    resp = requests.get(
        url,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    rows = resp.json()
    timestamps = [int(r[0]) for r in rows]
    closes = [float(r[4]) for r in rows]
    return timestamps, closes


# -- Core prediction -----------------------------------------------------------

def predict_range(
    closes,
    lookback: int = 300,
    ewma_lambda: float = 0.94,
    short_window: int = 24,
    n_sims: int = 20_000,
    confidence: float = 0.95,
    seed=None,
):
    """
    Predict the `confidence`-level price range for the NEXT bar.

    Parameters
    ----------
    closes       : list/array of close prices, ordered oldest->newest.
                   The prediction is for the bar AFTER closes[-1].
    lookback     : recent bars for distribution fit (300 ~= 12.5 days)
    ewma_lambda  : RiskMetrics-style EWMA decay factor (0.94)
    short_window : rolling-std window blended with EWMA (24 hours = 1 day)
    n_sims       : Monte-Carlo paths
    confidence   : nominal coverage level (0.95 => 2.5th/97.5th percentile)
    seed         : optional RNG seed for reproducibility

    Returns
    -------
    (lower, upper) price bounds, or None if insufficient data.
    """
    prices = np.asarray(
        closes[-lookback:] if len(closes) >= lookback else closes, dtype=float
    )

    if len(prices) < 30:
        return None

    log_ret = np.diff(np.log(prices))  # log-returns

    # -- 1. Fit Student-t distribution ----------------------------------------
    # stats.t.fit returns (df, loc, scale).
    # The returned scale is the correct MLE dispersion -- do NOT apply
    # sqrt((df-2)/df) to it. That formula shrinks scale and causes under-coverage.
    df, loc, _ = stats.t.fit(log_ret)
    df = max(df, 2.5)

    # -- 2. EWMA variance (volatility clustering) ------------------------------
    lam = ewma_lambda
    ewma_var = np.empty(len(log_ret))
    ewma_var[0] = log_ret[0] ** 2
    for k in range(1, len(log_ret)):
        ewma_var[k] = lam * ewma_var[k - 1] + (1.0 - lam) * log_ret[k] ** 2
    ewma_std = float(np.sqrt(ewma_var[-1]))

    # -- 3. Blended vol: EWMA (fast) + 24h rolling (stable) -------------------
    recent    = log_ret[-short_window:] if len(log_ret) >= short_window else log_ret
    short_std = float(np.std(recent, ddof=1))
    long_std  = float(np.std(log_ret, ddof=1))

    # 65% EWMA + 35% 24-bar rolling; floor at 35% of lookback-window std
    blended_vol = 0.65 * ewma_std + 0.35 * short_std
    vol = max(blended_vol, long_std * 0.35)

    # -- 4. Partial empirical tail correction ----------------------------------
    # Compare theoretical t-quantiles (with the blended vol) to the empirical
    # quantiles of the actual log-return distribution. Apply a damped correction
    # (40% toward empirical, 60% staying with the t-fit) to avoid over-widening.
    alpha  = (1.0 - confidence) / 2.0   # 0.025 for 95% CI
    emp_lo = float(np.percentile(log_ret, alpha * 100.0))
    emp_hi = float(np.percentile(log_ret, (1.0 - alpha) * 100.0))

    t_lo = float(stats.t.ppf(alpha,       df, loc=0.0, scale=vol))
    t_hi = float(stats.t.ppf(1.0 - alpha, df, loc=0.0, scale=vol))

    if t_lo != 0 and t_hi != 0:
        ratio_lo = abs(emp_lo) / abs(t_lo)
        ratio_hi = abs(emp_hi) / abs(t_hi)
        raw_ratio = (ratio_lo + ratio_hi) / 2.0
        # 40% pull toward empirical, 60% stay with t-fit
        correction = 0.40 * raw_ratio + 0.60 * 1.0
        correction = max(correction, 0.70)
        correction = min(correction, 2.0)
    else:
        correction = 1.0

    # -- 5. Monte-Carlo simulation ---------------------------------------------
    rng = np.random.default_rng(seed)
    sim_returns = stats.t.rvs(
        df, loc=loc, scale=vol * correction, size=n_sims, random_state=rng
    )

    lo_ret = float(np.percentile(sim_returns, alpha * 100.0))
    hi_ret = float(np.percentile(sim_returns, (1.0 - alpha) * 100.0))

    current_price = float(prices[-1])
    lower = current_price * np.exp(lo_ret)
    upper = current_price * np.exp(hi_ret)

    return float(lower), float(upper)


# -- Scoring helpers -----------------------------------------------------------

def winkler_score(lower: float, upper: float, actual: float, alpha: float = 0.05) -> float:
    """
    Winkler interval score.
      - Inside  : width
      - Outside : width + (2/alpha) x miss distance
    Lower is better.
    """
    width = upper - lower
    if lower <= actual <= upper:
        return width
    elif actual < lower:
        return width + (2.0 / alpha) * (lower - actual)
    else:
        return width + (2.0 / alpha) * (actual - upper)


def evaluate(predictions: list) -> dict:
    """
    Compute backtest summary statistics from a list of prediction dicts.
    Each dict must have: lower_95, upper_95, actual_price.
    """
    inside  = [p["lower_95"] <= p["actual_price"] <= p["upper_95"] for p in predictions]
    widths  = [p["upper_95"] - p["lower_95"] for p in predictions]
    winkler = [
        winkler_score(p["lower_95"], p["upper_95"], p["actual_price"])
        for p in predictions
    ]
    return {
        "coverage_95":     float(np.mean(inside)),
        "mean_width":      float(np.mean(widths)),
        "mean_winkler_95": float(np.mean(winkler)),
        "n":               len(predictions),
    }
