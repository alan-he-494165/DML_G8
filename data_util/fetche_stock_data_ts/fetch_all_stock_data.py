import pandas as pd
import os
import sys
import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from fetcher import TSFetcher

fetcher = TSFetcher()

stocks_file = os.path.join(os.path.dirname(__file__), 'zhongzheng500.txt')
stocks_list = []
with open(stocks_file, 'r', encoding='utf-8') as f:
    for line in f:
        symbol = line.strip()
        if symbol:
            stocks_list.append(symbol)

print(f"Loading {len(stocks_list)} stocks from Tushare...")

total_stocks = len(stocks_list)
success_count = 0
failed_stocks = []

for idx, symbol in enumerate(stocks_list, 1):
    try:
        ticker_day = fetcher.fetch_data(symbol, '20181231', '20251231')
        
        if ticker_day and len(ticker_day.date) > 0:
            success_count += 1
            print(f"[OK] {symbol} Success ({len(ticker_day.date)} records)")
        else:
            failed_stocks.append((symbol, "No data"))
    
    except Exception as e:
        failed_stocks.append((symbol, str(e)))
    
    if idx % 10 == 0:
        time.sleep(1)

print("\n" + "="*50)
print(f"Total stocks: {total_stocks}")
print(f"Success: {success_count}")
print(f"Failed: {len(failed_stocks)}")
print("="*50)

if failed_stocks:
    print("\nFailed stocks:")
    for symbol, reason in failed_stocks:
        print(f"  {symbol}: {reason}")

cache_dir = os.path.join(parent_dir, 'cache')
stock_data_files = len([f for f in os.listdir(cache_dir) if f.endswith('.pkl')])
print(f"\nFiles in cache directory: {stock_data_files}")
print(f"\nFiles in cache directory: {stock_data_files}")