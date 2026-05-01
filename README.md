---
title: BTC Forecast
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.33.0"
app_file: app.py
pinned: false
---
# BTC Forecast — AlphaI × Polaris Challenge

Predicts the 95% confidence price range for BTC/USDT one hour ahead using GBM with Student-t fat tails and rolling volatility clustering.

## File guide

| File | Where it runs | What it does |
|---|---|---|
| `model.py` | everywhere | Core GBM model — shared by all other files |
| `backtest.py` | Colab / local | Part A — 30-day walk-forward backtest |
| `app.py` | Streamlit Cloud | Parts B & C — live dashboard + history |
| `BTC_Forecast_Colab.ipynb` | Google Colab | Ready-to-run notebook for Part A |
| `backtest_results.jsonl` | generated | Output of backtest — commit to repo |
| `requirements.txt` | Streamlit Cloud | Python dependencies |

## How to run

### Part A — Backtest (Google Colab)
1. Upload `BTC_Forecast_Colab.ipynb` to Colab
2. Change `YOUR_USERNAME` in Cell 2 to your GitHub username
3. Run all cells (takes ~2 mins)
4. Copy `coverage_95` and `mean_winkler_95` for the submission form
5. Download `backtest_results.jsonl` and upload it to this repo

### Parts B & C — Dashboard (Streamlit Cloud)
1. Push all files to a public GitHub repo
2. Go to share.streamlit.io → Create app
3. Select this repo, branch `main`, file `app.py`
4. Deploy → get your public URL

### Local development
```bash
pip install -r requirements.txt
streamlit run app.py
```
