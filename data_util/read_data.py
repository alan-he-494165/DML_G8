import pickle
from pathlib import Path
import sys

# Add the fetcher directory to sys.path so pickle can find 'fetcher_yf' module
sys.path.insert(0, str(Path(__file__).parent / 'fetcher'))

# Import the module so pickle can find the classes
from fetcher_yf import Ticker_Day

# Define the cache directory
cache_dir = Path('cache')

# Read a specific pickle file
pkl_file = cache_dir / '000001.pkl'
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# Read AAPL data
aapl_file = cache_dir / '000001.pkl'
with open(aapl_file, 'rb') as f:
    aapl_data = pickle.load(f)

# Display first 5 lines of 000001 data
print("000001 First 5 Lines:")
print(f"Symbol: {aapl_data.symbol}")
print(f"\n{'Date':<12} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<15}")
print("-" * 70)

for i in range(min(5, len(aapl_data.date))):
    print(f"{str(aapl_data.date[i]):<12} {aapl_data.open[i]:<10.2f} {aapl_data.high[i]:<10.2f} {aapl_data.low[i]:<10.2f} {aapl_data.close[i]:<10.2f} {aapl_data.volume[i]:<15}")