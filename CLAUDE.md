# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DML_G8 is a 3-class stock volatility classification system for Chinese stocks (and some US stocks). It fetches OHLCV data, classifies daily volatility as LOW/MEDIUM/HIGH using statistical thresholds, and trains an XGBoost multi-class classifier using technical indicators and news sentiment features.

## Environment Setup

Uses `uv` for dependency management:

```bash
uv venv venv_name
# Activate (Windows): venv_name\Scripts\activate
# Activate (Unix): source venv_name/bin/activate
uv add -r requirements.txt
```

Python 3.13 required (see `.python-version`).

## Running the Pipeline

The project runs as a sequential data pipeline — each step produces outputs consumed by the next:

```bash
# 1. Fetch/cache stock data (Yahoo Finance)
python data_util/fetcher/fetcher_yf.py

# 2. Fetch/cache Chinese stock data (Tushare)
python data_util/fetcher/fetcher_ts.py

# 3. Classify volatility for all cached stocks
python data_util/data_processor/volatility_classifier.py

# 4. Score news sentiment with FinBERT (requires GPU for reasonable speed)
python data_util/process_news/calculate_semantic_score/process_raw_news_sentiment.py

# 5. Build training dataset (combines volatility labels, OHLCV, news)
python data_util/data_processor/create_xgboost_dataset.py

# 6. Train XGBoost model (Optuna tuning + early stopping) and produce evaluation plots
python xgboost/xgb.py
```

To download news data (required for news features):
```bash
pip install gdown
gdown 1ydXVbgVsQiTmCbJyh1BNf38gWRMqEEjr
tar -xf news.zip
```

## Architecture

### Data Layer — `data_util/`

**Fetchers** (`data_util/fetcher/`):
- `fetcher_yf.py`: `Ticker_Day` class wrapping Yahoo Finance data with persistent pickle caching in `data_util/cache/`. Supports `moving_average()`, `macd()`, `rsi()`, `subset()`, `merge()`.
- `fetcher_ts.py`: `TSFetcher` wrapping Tushare API for Chinese stocks. Auto-detects exchange suffix (.SZ/.BJ/.SH). Returns `Ticker_Day`-compatible objects. Same cache directory.
- Cache is keyed by ticker symbol (`{ticker}.pkl`). Fetching is incremental — only missing date ranges are downloaded.

**Volatility Classifier** (`data_util/data_processor/volatility_classifier.py`):
- `VolatilityClassifier` classifies each trading day as HIGH or LOW volatility using two criteria:
  1. Relative Amplitude Ratio: `(High - Low) / Close` vs threshold (default 2%)
  2. Z-Score: deviation from rolling historical mean vs threshold (default 1.5)
- Outputs `VolatilityRecord` dataclasses with a `confidence` score used for sample weighting in XGBoost.
- Processes all 998 cached stock files (~1.56M records total).

**Dataset Builder** (`data_util/data_processor/create_xgboost_dataset.py`):
- Reads raw stock cache from `data_for_process/cache_raw_stock/china_stock/`, volatility labels from `data_util/cache_output/`, and news from `data_for_process/news_daily_stock/`.
- Combines all into `data_for_process/xgboost_dataset_china_stock/xgboost_training_dataset.pkl`.
- Uses a local dummy `Ticker_Day` class to deserialize pickled objects without importing the fetcher module.

**News Processing** (`data_util/process_news/`):
- `read_daily_stock_news.py`: Fetches news from Tushare (Sina source), stores as pickle in `data_for_process/raw_news/`.
- `calculate_semantic_score/process_raw_news_sentiment.py`: Uses **FinBERT** (`ProsusAI/finbert`) with GPU support to score sentiment. Producer/consumer threading: CPU loads and matches news to tickers via `stock_name_mapping.pkl`, GPU runs inference in batches of 32. Outputs `NewsData` records to `data_for_process/news_daily_stock/{year}/{month}/{date}.pkl`.
- `match_stock_name/build_stock_name_mapping.py`: Builds `stock_name_mapping.pkl` mapping symbols to company names (including former names) for text matching.

### Model Layer — `xgboost/`

`xgb.py` trains a 3-class XGBoost classifier (LOW/MEDIUM/HIGH) with 5 features:
1. `intraday_range`
2. `volume_change_rate`
3. `rolling_historical_volatility`
4. `prev_day_news_count`
5. `prev_day_avg_news_sentiment`

All features are computed from day `t-1` to predict the label at day `t` (no leakage). Split: 80% train / 10% validation / 10% test (stratified). Sample weights = class-balance weights × confidence scores, normalized to mean=1.

Hyperparameters are tuned with **Optuna** (100 Bayesian trials, maximising macro F1 on validation set). Final model uses early stopping (patience=30). Outputs plots (`roc_curves.png`, `confusion_matrix_{train,validation,test}.png`, `feature_importance.png`) to the **current working directory** and saves the model to `xgboost/volatility_classifier_model.pkl`.

## Branch Strategy

- Personal branches named by team member initials (AH, BW, JF, MT, NW, RZ)
- Do not merge to `main` without team discussion
- To sync with main: `git merge origin main`

## Key Notes

- The Tushare API key is hardcoded in `fetcher_ts.py` — avoid committing alternative keys.
- Cache files in `data_util/cache/` are large pickle files; don't commit them.
- There are no automated tests — modules are verified by running them directly.
- `main.py` is currently a placeholder.

## Git commits

- Never mention claude code in any commit message under any circumstances
- Never commit unless given explicit user permission