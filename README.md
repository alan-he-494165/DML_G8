# README
This is the space for DML iexplore Group 8 code.

The project utilises ```uv``` to manage virtual python environment. Upon installing ```uv```, the environment shall be set up
1. Setup your own uv venv, named for example as ```venv_name```
```
uv venv venv_name
```
2. Upon activation of your environment (see manual of uv, can be different for UNIX-based system and DOS-based system), run 
```
uv add -r requirements.txt
```
Please do set up virtual environments accordingly to avoid dependency version control problem. Let me know if there is any incompatibility that I am not aware of.

3. Please work within your own branch as a starting point (named accordingly with your initials given in the team allocation sheet) to avoid version confliction. We can merge the code later. \Do backup the code actively!
   a) How to switch branch? For example using AH branch
   ```
   git checkout AH
   ```
   b) If your branch is outdated against ```main```, you may want to run
   ```
   git merge origin main
   ```
   c) Do NOT merge anything into branch unless you are superconfident at early-stage of the project. Do NOT merge anything into branch without discussion with others at late stage of the project

4. Useful websites
   · uv:[https://docs.astral.sh/uv/] \
   · XGBoost: [https://xgboost.readthedocs.io/en/stable/] \
   · Pandas: [https://pandas.pydata.org/docs/user_guide/index.html] \
   · yfinance: [https://ranaroussi.github.io/yfinance/] \
   · Tushare: [https://tushare.pro/document/2]

## fetcher_yf Module

A stock data fetcher with persistent caching. Fetches OHLCV data from Yahoo Finance and caches it locally to avoid repeated downloads.

### Usage
```python
from fetcher.fetcher_yf import Ticker_Day

# Fetch AAPL data (downloads and caches)
data = Ticker_Day.from_yf('AAPL', '2023-01-01', '2023-12-31')

# Access data
print(data.date)    # List of dates
print(data.high)    # High prices
print(data.low)     # Low prices
print(data.open)    # Open prices
print(data.close)   # Close prices
print(data.volume)  # Volume

# Force refresh (ignore cache)
data = Ticker_Day.from_yf('AAPL', '2023-01-01', '2023-12-31', force_refresh=True)
```

### Technical Indicators
```python
# Moving Average (default window=20)
ma = data.moving_average(window=20)

# MACD (returns macd_line, signal_line, histogram)
macd_line, signal_line, histogram = data.macd(short_window=12, long_window=26, signal_window=9)

# RSI (default period=14)
rsi = data.rsi(period=14)
```

### Utility Methods
```python
# Get subset of data for a date range
subset_data = data.subset('2023-06-01', '2023-06-30')

# Merge data from another Ticker_Day object
data.merge(other_data)
```

### Cache Behavior
- Cache stored in `cache/` directory as `{ticker}.pkl`
- Only fetches missing date ranges from Yahoo Finance
- Automatically merges new data with cached data

## fetcher_ts Module

A stock data fetcher for Chinese stocks using Tushare API. Fetches OHLCV data and caches it locally, automatically trying different exchanges (Shenzhen, Beijing, Shanghai).

### Usage
```python
from fetcher.fetcher_ts import TSFetcher

# Fetch Chinese stock data (e.g., Ping An Bank)
data = TSFetcher.from_ts('000001', '2023-01-01', '2023-12-31')

# Access data (same interface as fetcher_yf)
print(data.date)    # List of dates
print(data.high)    # High prices
print(data.low)     # Low prices
print(data.open)    # Open prices
print(data.close)   # Close prices
print(data.volume)  # Volume

# Force refresh (ignore cache)
data = TSFetcher.from_ts('000001', '2023-01-01', '2023-12-31', force_refresh=True)

# Use custom API key
data = TSFetcher.from_ts('000001', '2023-01-01', '2023-12-31', api_key='your_api_key')
```

### Features
- Automatically tries multiple exchanges (.SZ, .BJ, .SH) to find valid data
- Returns `Ticker_Day` objects compatible with fetcher_yf
- Built-in retry mechanism for API failures
- Cache stored in `cache/` directory as `{symbol}.pkl`

