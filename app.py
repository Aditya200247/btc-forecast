import json
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import fetch_klines, predict_range, evaluate, winkler_score

DB_FILE = "predictions.db"
BACKTEST_FILE = "backtest_results.jsonl"
BARS_FOR_CHART = 50
REFRESH_SECONDS = 60


def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def init_db():
    with get_conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                made_at         TEXT NOT NULL,
                current_bar_ts  INTEGER NOT NULL UNIQUE,
                target_bar_ts   INTEGER NOT NULL,
                current_price   REAL NOT NULL,
                lower_95        REAL NOT NULL,
                upper_95        REAL NOT NULL,
                actual_price    REAL,
                inside          INTEGER
            )
        """)


def save_prediction(rec):
    with get_conn() as c:
        c.execute("""
            INSERT OR IGNORE INTO predictions
              (made_at, current_bar_ts, target_bar_ts, current_price, lower_95, upper_95)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            rec["made_at"], rec["current_bar_ts"], rec["target_bar_ts"],
            rec["current_price"], rec["lower_95"], rec["upper_95"],
        ))


def fill_actuals(timestamps, closes):
    ts_map = dict(zip(timestamps, closes))
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, target_bar_ts, lower_95, upper_95 FROM predictions WHERE actual_price IS NULL"
        ).fetchall()
        for row_id, target_ts, lower, upper in rows:
            if target_ts in ts_map:
                actual = ts_map[target_ts]
                c.execute(
                    "UPDATE predictions SET actual_price=?, inside=? WHERE id=?",
                    (actual, int(lower <= actual <= upper), row_id),
                )


def load_history():
    with get_conn() as c:
        return pd.read_sql("SELECT * FROM predictions ORDER BY current_bar_ts DESC LIMIT 200", c)


@st.cache_data(ttl=3600)
def load_backtest_metrics():
    if not os.path.exists(BACKTEST_FILE):
        return None
    preds = [json.loads(line) for line in open(BACKTEST_FILE) if line.strip()]
    return evaluate(preds) if preds else None


def build_chart(timestamps, closes, lower, upper):
    ts = timestamps[-BARS_FOR_CHART:]
    cls = closes[-BARS_FOR_CHART:]
    dts = [datetime.fromtimestamp(t / 1000, tz=timezone.utc) for t in ts]
    dt_next = datetime.fromtimestamp((ts[-1] + 3_600_000) / 1000, tz=timezone.utc)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dts, y=cls, mode="lines", name="BTC Close",
        line=dict(color="#F7931A", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=[dts[-1], dt_next, dt_next, dts[-1]],
        y=[upper, upper, lower, lower],
        fill="toself",
        fillcolor="rgba(99,202,255,0.20)",
        line=dict(color="rgba(99,202,255,0.60)", width=1, dash="dot"),
        name="95% range",
    ))

    for val, label in [(upper, f"▲ ${upper:,.0f}"), (lower, f"▼ ${lower:,.0f}")]:
        fig.add_annotation(
            x=dt_next, y=val, text=label, showarrow=False,
            xanchor="left", font=dict(color="#63CAFF", size=12), xshift=6,
        )

    fig.update_layout(
        title="BTCUSDT — Last 50 Bars + Next-Hour Forecast",
        xaxis_title="Time (UTC)", yaxis_title="Price (USDT)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=80, t=50, b=10), height=400,
    )
    return fig


def main():
    st.set_page_config(page_title="BTC Forecast", page_icon="₿", layout="wide")
    init_db()

    st.markdown(f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">', unsafe_allow_html=True)
    st.title("₿ BTC/USDT — Next-Hour Forecast")
    st.caption(f"Refreshes every {REFRESH_SECONDS}s · GBM + Student-t · 95% confidence")

    with st.spinner("Fetching data from Binance…"):
        try:
            timestamps, closes = fetch_klines(limit=500)
        except Exception as e:
            st.error(f"Binance API error: {e}")
            st.stop()

    current_price = closes[-1]
    current_ts = timestamps[-1]
    target_ts = current_ts + 3_600_000

    result = predict_range(closes)
    if result is None:
        st.error("Not enough data.")
        st.stop()

    lower, upper = result
    mid = (lower + upper) / 2
    width = upper - lower

    save_prediction({
        "made_at": datetime.now(tz=timezone.utc).isoformat(),
        "current_bar_ts": current_ts,
        "target_bar_ts": target_ts,
        "current_price": current_price,
        "lower_95": round(lower, 2),
        "upper_95": round(upper, 2),
    })
    fill_actuals(timestamps, closes)

    bt = load_backtest_metrics()

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BTC Price", f"${current_price:,.2f}")
    c2.metric("Predicted Low", f"${lower:,.2f}")
    c3.metric("Predicted High", f"${upper:,.2f}")
    c4.metric("Range Width", f"${width:,.2f}")
    if bt:
        c5.metric("Backtest Coverage", f"{bt['coverage_95']:.3f}",
                  delta=f"{bt['coverage_95'] - 0.95:+.3f} vs 0.95")
    else:
        c5.metric("Backtest Coverage", "—")

    if bt:
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Width (backtest)", f"${bt['mean_width']:,.2f}")
        m2.metric("Mean Winkler Score", f"{bt['mean_winkler_95']:,.2f}")
        m3.metric("Total Predictions", f"{bt['n']}")

    st.markdown("---")
    st.info(f"📊 Next-hour range: **${lower:,.2f} – ${upper:,.2f}** · mid ${mid:,.2f} · width ${width:,.2f}")

    st.plotly_chart(build_chart(timestamps, closes, lower, upper), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Prediction History")

    history = load_history()
    if history.empty:
        st.info("History builds up with each visit.")
    else:
        def fmt_ts(ms):
            try:
                return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                return "—"

        display = history.copy()
        display["Time"] = display["current_bar_ts"].apply(fmt_ts)
        display["Target Hour"] = display["target_bar_ts"].apply(fmt_ts)
        display["Current Price"] = display["current_price"].apply(lambda x: f"${x:,.2f}")
        display["Lower 95%"] = display["lower_95"].apply(lambda x: f"${x:,.2f}")
        display["Upper 95%"] = display["upper_95"].apply(lambda x: f"${x:,.2f}")
        display["Actual"] = display["actual_price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "⏳ pending")
        display["Hit"] = display["inside"].apply(lambda x: "✅" if x == 1 else ("❌" if x == 0 else "—"))

        st.dataframe(
            display[["Time", "Target Hour", "Current Price", "Lower 95%", "Upper 95%", "Actual", "Hit"]],
            use_container_width=True, hide_index=True,
        )

        resolved = history[history["inside"].notna()]
        if not resolved.empty:
            live_cov = resolved["inside"].mean()
            live_wink = np.mean([
                winkler_score(r["lower_95"], r["upper_95"], r["actual_price"])
                for _, r in resolved.iterrows()
            ])
            r1, r2, r3 = st.columns(3)
            r1.metric("Live Coverage", f"{live_cov:.3f}")
            r2.metric("Live Winkler", f"{live_wink:,.2f}")
            r3.metric("Resolved", f"{len(resolved)}")

    st.caption("Data: Binance · Model: GBM + Student-t with rolling volatility")


if __name__ == "__main__":
    main()
