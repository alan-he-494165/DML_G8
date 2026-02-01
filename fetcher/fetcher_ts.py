import tushare as ts
import pandas as pd
import pickle
import os
import time

from .fetcher_yf import Ticker_Day

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')


class TSFetcher:
    """
    Tushare Stock Data Fetcher
    Fetches stock data from Tushare API and caches it locally as Ticker_Day objects
    Similar to YFFetcher but for Chinese stocks from Tushare
    """
    
    def __init__(self, api_key=None):
        """
        Initialize TSFetcher
        
        PARAMETERS
        ----------
        api_key : str
            Tushare API key. If None, uses default key.
        """
        self.api_key = api_key or '31c93f930abfd98726b28a7d984e5d56163f69c1f040ba97d6d4436a'
        self.pro = ts.pro_api(self.api_key)
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _get_cache_path(self, ticker):
        """Get the cache file path for a ticker symbol"""
        return os.path.join(CACHE_DIR, f'{ticker}.pkl')

    def _load_cache(self, ticker):
        """Load cached ticker data from pickle file"""
        path = self._get_cache_path(ticker)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_cache(self, ticker, ticker_day):
        """Save ticker data to cache"""
        path = self._get_cache_path(ticker)
        with open(path, 'wb') as f:
            pickle.dump(ticker_day, f)

    def _fetch_online(self, ts_code, start_date, end_date):
        """
        Fetch data from Tushare API for the given date range
        
        PARAMETERS
        ----------
        ts_code : str
            Tushare stock code (e.g., '000001.SZ')
        start_date : str
            Start date (format: YYYYMMDD or YYYY-MM-DD)
        end_date : str
            End date (format: YYYYMMDD or YYYY-MM-DD)
            
        RETURNS
        -------
        Ticker_Day or None
            Ticker_Day object with the fetched data, or None if no data available
        """
        # Convert date format from YYYY-MM-DD to YYYYMMDD if needed
        if isinstance(start_date, str) and '-' in start_date:
            start_date = start_date.replace('-', '')
        if isinstance(end_date, str) and '-' in end_date:
            end_date = end_date.replace('-', '')
        
        try:
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields=[
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol"
                ]
            )
        except Exception as e:
            return None
        
        if df is None or len(df) == 0:
            return None
        
        # Extract symbol from ts_code
        symbol = ts_code.split('.')[0]
        
        # Convert trade_date to timestamps
        dates = [pd.Timestamp(str(date_str)) for date_str in df['trade_date']]
        
        return Ticker_Day(
            symbol=symbol,
            date=dates,
            open=df['open'].tolist(),
            high=df['high'].tolist(),
            low=df['low'].tolist(),
            close=df['close'].tolist(),
            volume=df['vol'].tolist()
        )

    def fetch_data(self, symbol, start_date, end_date, force_refresh=False):
        """
        Fetch stock data for a symbol, trying different exchanges if needed
        
        PARAMETERS
        ----------
        symbol : str
            Stock symbol (without exchange suffix, e.g., '000001')
        start_date : str
            Start date
        end_date : str
            End date
        force_refresh : bool
            If True, fetch fresh data even if cached data exists
            
        RETURNS
        -------
        Ticker_Day or None
            Ticker_Day object with the fetched data
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Try to load cached data first
        cached = self._load_cache(symbol)
        
        if cached is not None and not force_refresh:
            cache_start, cache_end = cached.date_range
            # Check if cache fully covers the requested range
            if cache_start <= start and cache_end >= end:
                return cached.subset(start_date, end_date)
        
        # Not in cache or force refresh - try to fetch from API
        result = self._fetch_data_with_exchanges(symbol, start_date, end_date)
        
        if result:
            self._save_cache(symbol, result)
            return result
        
        # If we have cached data but it doesn't cover the full range, return what we have
        if cached:
            return cached.subset(start_date, end_date)
        
        return None

    def _fetch_data_with_exchanges(self, symbol, start_date, end_date):
        """
        Try to fetch data using different exchange suffixes
        
        PARAMETERS
        ----------
        symbol : str
            Stock symbol without exchange suffix
        start_date : str
            Start date
        end_date : str
            End date
            
        RETURNS
        -------
        Ticker_Day or None
            First successful result from any exchange, or None if all fail
        """
        # Try all three exchanges: .SZ (Shenzhen), .BJ (Beijing), .SH (Shanghai)
        for exchange in ['.SZ', '.BJ', '.SH']:
            ts_code = symbol + exchange
            try:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result = self._fetch_online(ts_code, start_date, end_date)
                        if result and len(result.date) > 0:
                            return result
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                        else:
                            break
            except Exception as e:
                continue
        
        return None

    @classmethod
    def from_ts(cls, symbol, start_date, end_date, force_refresh=False, api_key=None):
        """
        Fetch data from Tushare using TSFetcher
        
        PARAMETERS
        ----------
        symbol : str
            Stock symbol
        start_date : str
            Start date
        end_date : str
            End date
        force_refresh : bool
            If True, fetch fresh data even if cached
        api_key : str
            Tushare API key
            
        RETURNS
        -------
        Ticker_Day or None
            Ticker_Day object with the fetched data
        """
        return cls(api_key=api_key).fetch_data(symbol, start_date, end_date, force_refresh=force_refresh)
