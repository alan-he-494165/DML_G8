import pandas as pd
import sys
from pathlib import Path
import time
import yfinance as yf

sys.path.append(str(Path(__file__).parent.parent))
from fetcher.fetcher_yf import Ticker_Day

df = pd.read_excel("SP500.xlsx")
stock = df["Ticker"].to_list()

print(f"Total stocks to fetch: {len(stock)}")

count = 0

for i, symbol in enumerate(stock):
    try:
        data = Ticker_Day.from_yf(symbol, "2020-01-01", "2025-12-31", force_refresh=True)
        count += 1
        print(f"{i}. Fetched data for: {symbol}, total successful: {count}")
        time.sleep(0.5)
    except:
        print(f"{i}. Failed to fetch data for: {symbol}")
        time.sleep(0.5)