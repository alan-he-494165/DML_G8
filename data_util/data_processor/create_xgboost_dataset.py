"""
XGBoost Training Dataset Creator
Creates training features from stock data for binary volatility prediction

Features:
- Volume Change Rate: (volume_today - volume_yesterday) / volume_yesterday
- Intraday Range: (high - low) / close
- Rolling Historical Volatility: std of daily returns over N days
- Confidence: from volatility classification model (as sample weight)

Output: Training dataset with features and labels for XGBoost
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# Add parent directory to path for fetcher module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Define a dummy Ticker_Day class to handle pickle deserialization
class Ticker_Day:
    """Dummy class for unpickling Ticker_Day objects without fetcher module"""
    def __init__(self, symbol=None, date=None, high=None, low=None, open=None, close=None, volume=None):
        self.symbol = symbol
        self.date = date if date is not None else []
        self.high = high if high is not None else []
        self.low = low if low is not None else []
        self.open = open if open is not None else []
        self.close = close if close is not None else []
        self.volume = volume if volume is not None else []

# Register the dummy class in sys.modules to handle unpickling
if 'fetcher.fetcher_yf' not in sys.modules:
    import types
    fetcher_module = types.ModuleType('fetcher.fetcher_yf')
    fetcher_module.Ticker_Day = Ticker_Day
    sys.modules['fetcher.fetcher_yf'] = fetcher_module
    sys.modules['fetcher'] = types.ModuleType('fetcher')
    sys.modules['fetcher'].fetcher_yf = fetcher_module

CACHE_DIR = os.path.join('data_for_process', 'cache_raw_stock', 'china_stock')  # Adjust if needed
CACHE_OUTPUT_DIR = os.path.join('data_for_process', 'cache_output', 'china_stock')  # Adjust if needed
OUTPUT_DIR = os.path.join('data_for_process', 'xgboost_dataset_china_stock')  # Adjust if needed

# Configuration
ROLLING_WINDOW = 20  # 20-day rolling volatility window
MIN_VALID_RECORDS = 50  # minimum records needed per stock


@dataclass
class VolatilityRecord:
    """Data structure for volatility classification output"""
    date: 'pd.Timestamp'
    open_price: float
    symbol: str
    is_high_volatility: bool
    relative_amplitude_ratio: float
    z_score: float
    confidence: float


class XGBoostDatasetCreator:
    """
    Creates XGBoost training dataset from stock data and volatility labels
    """
    
    def __init__(self, rolling_window: int = ROLLING_WINDOW):
        """
        Initialize dataset creator
        
        Args:
            rolling_window: window size for rolling volatility calculation
        """
        self.rolling_window = rolling_window
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def load_volatility_labels(self, symbol: str) -> List[VolatilityRecord]:
        """
        Load volatility classification records for a symbol
        
        Args:
            symbol: stock symbol
            
        Returns:
            List of VolatilityRecord objects
        """
        pkl_file = os.path.join(CACHE_OUTPUT_DIR, f"{symbol}.pkl")
        try:
            with open(pkl_file, 'rb') as f:
                records = pickle.load(f)
            return records
        except Exception as e:
            print(f"Error loading volatility labels for {symbol}: {e}")
            return []
    
    def load_stock_data(self, symbol: str):
        """
        Load stock price and volume data
        
        Args:
            symbol: stock symbol
            
        Returns:
            Ticker_Day object or None if error
        """
        pkl_file = os.path.join(CACHE_DIR, f"{symbol}.pkl")
        try:
            with open(pkl_file, 'rb') as f:
                ticker_data = pickle.load(f)
            return ticker_data
        except Exception as e:
            print(f"Error loading stock data for {symbol}: {e}")
            return None
    
    def calculate_features(self, ticker_data, volatility_records: List[VolatilityRecord]) -> pd.DataFrame:
        """
        Calculate features for XGBoost training
        
        Features from day t-1 are used to predict the label on day t.
        This prevents data leakage - we never use today's data to predict today's label.
        
        Args:
            ticker_data: Ticker_Day object
            volatility_records: list of VolatilityRecord objects
            
        Returns:
            DataFrame with calculated features
        """
        if not ticker_data or not volatility_records:
            return None
        
        # Create mapping from date to volatility record
        vol_map = {r.date: r for r in volatility_records}
        
        # Convert to pandas Series for easier calculation
        dates = pd.Series(ticker_data.date)
        opens = pd.Series(ticker_data.open) if ticker_data.open else None
        highs = pd.Series(ticker_data.high)
        lows = pd.Series(ticker_data.low)
        closes = pd.Series(ticker_data.close)
        volumes = pd.Series(ticker_data.volume) if ticker_data.volume else None
        
        features_list = []
        
        # Start from i=1 because we need i-1 to compute features for predicting label at i
        for i in range(1, len(ticker_data.date)):
            # Target date: today (day i)
            target_date = ticker_data.date[i]
            
            # Get volatility label for TODAY (day i)
            if target_date not in vol_map:
                continue
            
            vol_record = vol_map[target_date]
            
            row = {
                'date': target_date,
                'symbol': ticker_data.symbol,
                'close_today': closes[i],
            }
            
            # ===== FEATURES CALCULATED FROM DAY t-1 (YESTERDAY) =====
            # This prevents data leakage - we only use historical data
            
            # Feature 1: Intraday Range from YESTERDAY (t-1)
            # Use yesterday's high/low to predict today's volatility
            intraday_range_prev = (highs[i-1] - lows[i-1]) / closes[i-1] if closes[i-1] != 0 else 0
            row['intraday_range'] = intraday_range_prev
            
            # Feature 2: Volume Change Rate from YESTERDAY (t-1 vs t-2)
            if volumes is not None and i > 1 and volumes[i-2] != 0:
                volume_change_rate = (volumes[i-1] - volumes[i-2]) / volumes[i-2]
                row['volume_change_rate'] = volume_change_rate
            else:
                row['volume_change_rate'] = np.nan
            
            # Feature 3: Rolling Historical Volatility up to YESTERDAY
            # Calculate using returns from (i-1-rolling_window) to (i-1)
            if i - 1 >= self.rolling_window:
                returns = []
                for j in range(i - 1 - self.rolling_window, i - 1):
                    if closes[j] != 0:
                        ret = (closes[j+1] - closes[j]) / closes[j]
                        returns.append(ret)
                
                if returns:
                    rolling_volatility = np.std(returns)
                    row['rolling_historical_volatility'] = rolling_volatility
                else:
                    row['rolling_historical_volatility'] = np.nan
            else:
                row['rolling_historical_volatility'] = np.nan
            
            # ===== TARGET: TODAY'S VOLATILITY LABEL (DAY t) =====
            # Label: is_high_volatility (binary: 1 or 0) for TODAY
            row['target'] = int(vol_record.is_high_volatility)
            
            # Sample weight: confidence score
            row['confidence'] = vol_record.confidence
            
            # Additional useful columns
            row['relative_amplitude_ratio'] = vol_record.relative_amplitude_ratio
            row['z_score'] = vol_record.z_score
            
            features_list.append(row)
        
        if not features_list:
            return None
        
        return pd.DataFrame(features_list)
    
    def create_dataset(self) -> pd.DataFrame:
        """
        Create complete dataset from all stocks
        
        Returns:
            DataFrame with all features and labels
        """
        all_data = []
        cache_files = sorted(Path(CACHE_DIR).glob('*.pkl'))
        total_files = len(cache_files)
        
        print(f"Total cache files found: {total_files}")
        
        if total_files == 0:
            print(f"ERROR: No files found in {CACHE_DIR}/")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Absolute cache path: {os.path.abspath(CACHE_DIR)}")
            return None
        
        successful = 0
        skipped = 0
        
        for idx, pkl_file in enumerate(cache_files):
            symbol = pkl_file.stem
            
            # Load data
            ticker_data = self.load_stock_data(symbol)
            volatility_records = self.load_volatility_labels(symbol)
            
            if not ticker_data or not volatility_records:
                skipped += 1
                continue
            
            try:
                # Calculate features for this stock
                stock_features = self.calculate_features(ticker_data, volatility_records)
                
                if stock_features is not None and len(stock_features) > MIN_VALID_RECORDS:
                    all_data.append(stock_features)
                    successful += 1
                    print(f"[{idx+1}/{total_files}] {symbol}: {len(stock_features)} records ✓")
                else:
                    if stock_features is not None:
                        print(f"[{idx+1}/{total_files}] {symbol}: insufficient records ({len(stock_features)})")
                    else:
                        print(f"[{idx+1}/{total_files}] {symbol}: no features calculated")
                    skipped += 1
            except Exception as e:
                print(f"[{idx+1}/{total_files}] {symbol}: ERROR - {e}")
                skipped += 1
        
        print(f"\nProcessing summary: {successful} successful, {skipped} skipped")
        
        # Combine all data
        if not all_data:
            print("No data collected!")
            return None
        
        dataset = pd.concat(all_data, ignore_index=True)
        return dataset
    
    def save_dataset(self, dataset: pd.DataFrame):
        """
        Save dataset in multiple formats
        
        Args:
            dataset: DataFrame with features and labels
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save as pickle (more efficient for large datasets)
        pkl_path = os.path.join(OUTPUT_DIR, 'xgboost_training_dataset.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved: {pkl_path}")
    
    def generate_summary_statistics(self, dataset: pd.DataFrame):
        """
        Generate summary statistics for the dataset
        
        Args:
            dataset: DataFrame with features and labels
        """
        print("\n" + "="*70)
        print("XGBOOST TRAINING DATASET SUMMARY")
        print("="*70 + "\n")
        
        print(f"Total samples: {len(dataset)}")
        print(f"Total unique stocks: {dataset['symbol'].nunique()}")
        print(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")
        
        print(f"\nTarget Distribution:")
        target_counts = dataset['target'].value_counts()
        print(f"  High Volatility (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(dataset)*100:.2f}%)")
        print(f"  Low Volatility (0):  {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(dataset)*100:.2f}%)")
        
        print(f"\nFeature Statistics:")
        feature_cols = ['intraday_range', 'volume_change_rate', 'rolling_historical_volatility']
        
        for col in feature_cols:
            valid_data = dataset[col].dropna()
            if len(valid_data) > 0:
                print(f"\n  {col}:")
                print(f"    Valid samples: {len(valid_data)} ({len(valid_data)/len(dataset)*100:.2f}%)")
                print(f"    Mean: {valid_data.mean():.6f}")
                print(f"    Std: {valid_data.std():.6f}")
                print(f"    Min: {valid_data.min():.6f}")
                print(f"    Max: {valid_data.max():.6f}")
                print(f"    Median: {valid_data.median():.6f}")
            else:
                print(f"\n  {col}: No valid data")
        
        print(f"\nConfidence Score Distribution:")
        print(f"  Mean: {dataset['confidence'].mean():.4f}")
        print(f"  Min: {dataset['confidence'].min():.4f}")
        print(f"  Max: {dataset['confidence'].max():.4f}")
        print(f"  Median: {dataset['confidence'].median():.4f}")
        
        # Data quality
        print(f"\nData Quality (Missing Values):")
        for col in feature_cols:
            missing_pct = dataset[col].isna().sum() / len(dataset) * 100
            print(f"  {col}: {missing_pct:.2f}%")
        
        print("\n" + "="*70)


def main():
    """Main pipeline"""
    print("\n" + "="*70)
    print("XGBOOST TRAINING DATASET CREATION".center(70))
    print("="*70 + "\n")
    
    print("DATA LEAKAGE PREVENTION:")
    print("-" * 70)
    print("Features:  Calculated from day (t-1)")
    print("Target:    Volatility label for day (t)")
    print("")
    print("Example: Use yesterday's data to predict today's volatility")
    print("         This avoids using today's price movements to predict")
    print("         today's volatility (which would be data leakage)")
    print("-" * 70 + "\n")
    
    # Initialize creator
    creator = XGBoostDatasetCreator(rolling_window=ROLLING_WINDOW)
    
    # Create dataset
    print("Creating dataset from stock data and volatility labels...")
    print("-" * 70)
    dataset = creator.create_dataset()
    
    if dataset is None or len(dataset) == 0:
        print("Failed to create dataset!")
        return
    
    print("-" * 70)
    print(f"Total records collected: {len(dataset)}\n")
    
    # Remove rows with NaN in required features (for XGBoost)
    print("Cleaning dataset (removing NaN values in required features)...")
    initial_rows = len(dataset)
    dataset = dataset.dropna(subset=['intraday_range', 'rolling_historical_volatility'])
    final_rows = len(dataset)
    removed = initial_rows - final_rows
    
    print(f"Removed {removed} rows with missing features")
    print(f"Final dataset size: {final_rows} samples\n")
    
    # For volume_change_rate, fill with 0 if NaN (since not always available)
    dataset['volume_change_rate'].fillna(0, inplace=True)
    
    # Generate statistics
    creator.generate_summary_statistics(dataset)
    
    # Keep required features plus target, confidence, symbol, and date
    dataset = dataset[
        [
            'symbol',
            'date',
            'rolling_historical_volatility',
            'volume_change_rate',
            'intraday_range',
            'target',
            'confidence',
        ]
    ].copy()

    # Save dataset
    print("\nSaving dataset...")
    print("-" * 70)
    creator.save_dataset(dataset)
    
    # Display sample rows
    print("\nSample rows from dataset:")
    print("-" * 70)
    sample_cols = [
        'symbol',
        'date',
        'rolling_historical_volatility',
        'volume_change_rate',
        'intraday_range',
        'target',
        'confidence',
    ]
    print(dataset[sample_cols].head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("DATASET CREATION COMPLETE")
    print("="*70 + "\n")
    
    print("Output Files:")
    print(f"  1. {OUTPUT_DIR}/xgboost_training_dataset.pkl")
    print("\nUsage in xgb.py:")
    print("  - Load PKL: pd.read_pickle('data_for_process/xgboost_training_dataset.pkl')")
    print("  - Train with weights: model.fit(X, y, sample_weight=weights)")


if __name__ == '__main__':
    main()
