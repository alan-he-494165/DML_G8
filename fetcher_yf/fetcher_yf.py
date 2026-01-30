import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')

class Ticker_Day:
    def __init__(self, symbol=None, date=None, high=None, low=None, open=None, close=None, volume=None):
        self.symbol = symbol
        self.high = high if high is not None else []
        self.low = low if low is not None else []
        self.open = open if open is not None else []
        self.close = close if close is not None else []
        self.volume = volume if volume is not None else []
        self.date = date if date is not None else []
        self.date_range = (min(self.date), max(self.date)) if self.date else (None, None)

    def merge(self, other):
        """Merge data from another Ticker_Day, removing duplicates and sorting by date."""
        if other is None or not other.date:
            return

        # Combine into a dict keyed by date to remove duplicates
        data_dict = {}
        for i, d in enumerate(self.date):
            data_dict[d] = (self.high[i], self.low[i], self.open[i], self.close[i], self.volume[i])
        for i, d in enumerate(other.date):
            data_dict[d] = (other.high[i], other.low[i], other.open[i], other.close[i], other.volume[i])

        # Sort by date and rebuild lists
        sorted_dates = sorted(data_dict.keys())
        self.date = sorted_dates
        self.high = [data_dict[d][0] for d in sorted_dates]
        self.low = [data_dict[d][1] for d in sorted_dates]
        self.open = [data_dict[d][2] for d in sorted_dates]
        self.close = [data_dict[d][3] for d in sorted_dates]
        self.volume = [data_dict[d][4] for d in sorted_dates]
        self.date_range = (min(self.date), max(self.date)) if self.date else (None, None)

    def subset(self, start_date, end_date):
        """Return a new Ticker_Day with only dates in the given range."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        indices = [i for i, d in enumerate(self.date) if start <= d <= end]

        return Ticker_Day(
            symbol=self.symbol,
            date=[self.date[i] for i in indices],
            high=[self.high[i] for i in indices],
            low=[self.low[i] for i in indices],
            open=[self.open[i] for i in indices],
            close=[self.close[i] for i in indices],
            volume=[self.volume[i] for i in indices]
        )

    @classmethod
    def from_yf(cls, symbol, start_date, end_date, force_refresh=False):
        return YFFetcher().fetch_data(symbol, start_date, end_date, force_refresh=force_refresh)

class YFFetcher:
    def __init__(self):
        self.dates_avail = None
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _get_cache_path(self, ticker):
        return os.path.join(CACHE_DIR, f'{ticker}.pkl')

    def _load_cache(self, ticker):
        path = self._get_cache_path(ticker)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_cache(self, ticker, ticker_day):
        path = self._get_cache_path(ticker)
        with open(path, 'wb') as f:
            pickle.dump(ticker_day, f)

    def _fetch_online(self, ticker, start_date, end_date):
        """Fetch data from yfinance for the given range."""
        data = yf.download(ticker, start=start_date, end=end_date)
        if data is None or data.empty:
            return None
        data = data.droplevel('Ticker', axis=1)
        return Ticker_Day(
            symbol=ticker,
            high=data['High'].tolist(),
            low=data['Low'].tolist(),
            open=data['Open'].tolist(),
            close=data['Close'].tolist(),
            volume=data['Volume'].tolist(),
            date=data.index.tolist()
        )

    def fetch_data(self, ticker, start_date, end_date, force_refresh=False):
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Load cached data
        cached = self._load_cache(ticker)

        if cached is None or force_refresh:
            # No cache - fetch entire range
            result = self._fetch_online(ticker, start_date, end_date)
            if result:
                self._save_cache(ticker, result)
            return result

        cache_start, cache_end = cached.date_range

        # Check if cache fully covers the requested range
        if cache_start <= start and cache_end >= end:
            return cached.subset(start_date, end_date)

        # Missing
        if start < cache_start:
            before_data = self._fetch_online(ticker, start_date, str(cache_start.date()))
            cached.merge(before_data)

        if end > cache_end:
            after_data = self._fetch_online(ticker, str(cache_end.date()), end_date)
            cached.merge(after_data)

        # Save updated cache
        self._save_cache(ticker, cached)

        return cached.subset(start_date, end_date)

test = Ticker_Day.from_yf('AAPL', '2023-01-01', '2023-01-10')
print(test.high)