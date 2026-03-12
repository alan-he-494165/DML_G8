"""
Trinary Volatility Classifier
Processes all stock data and generates volatility labels using z-score and relative amplitude ratio
Output: pickle files with Date, Open, Symbol, and volatility_trinary label (0=LOW, 1=MEDIUM, 2=HIGH)
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Add parent directory to path to import fetcher module
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

# Add parent directory to path to import fetcher module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Cache is at project root level, so go up 3 levels from data_processor
# Raw stock data: DML_G8/cache_raw_stock/china_stock
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cache_raw_stock', 'china_stock')

# Output directory: DML_G8/cache_output_trinary
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cache_output_trinary')


@dataclass
class VolatilityRecord:
    """Data structure for volatility classification output"""
    date: pd.Timestamp
    open_price: float
    symbol: str
    volatility_trinary: int  # 0=LOW, 1=MEDIUM, 2=HIGH
    relative_amplitude_ratio: float
    z_score: float
    confidence: float


class VolatilityClassifier:
    """
    Trinary volatility classifier using relative amplitude ratio and z-score
    Labels: 0=LOW, 1=MEDIUM, 2=HIGH
    """

    def __init__(self, high_zscore_threshold=0.5, high_ratio_threshold=0.03, extreme_ratio_threshold=0.05, extreme_zscore_threshold=2.0,
                 low_zscore_threshold=-0.5, low_ratio_threshold=0.015, extreme_low_ratio_threshold=0.0075, extreme_low_zscore_threshold=-1.0,
                 confidence_method='avg'):
        """
        Initialize classifier with fixed thresholds

        Args:
            high_zscore_threshold: z-score threshold for HIGH boundary (default: 0.5)
            high_ratio_threshold: relative ratio threshold for HIGH boundary (default: 0.03)
            extreme_ratio_threshold: extreme ratio threshold for automatic HIGH (default: 0.05)
            extreme_zscore_threshold: extreme z-score threshold for automatic HIGH (default: 2.0)
            low_zscore_threshold: z-score threshold for LOW boundary (default: -0.5)
            low_ratio_threshold: relative ratio threshold for LOW boundary (default: 0.015)
            extreme_low_ratio_threshold: extreme low ratio threshold for automatic LOW (default: 0.0075)
            extreme_low_zscore_threshold: extreme low z-score threshold for automatic LOW (default: -1.0)
            confidence_method: method for MEDIUM confidence calculation (default: 'min')
                - 'min': confidence = minimum distance to any boundary (conservative)
                - 'avg': confidence = average distance to all boundaries (higher values)

        Classification rules:
        - HIGH (2):   (z_score > high_zscore_threshold AND relative_ratio >= high_ratio_threshold)
                      OR relative_ratio >= extreme_ratio_threshold OR z_score > extreme_zscore_threshold
        - LOW (0):    (z_score <= low_zscore_threshold AND relative_ratio < low_ratio_threshold)
                      OR relative_ratio <= extreme_low_ratio_threshold OR z_score < extreme_low_zscore_threshold
        - MEDIUM (1): Everything else
        """
        # Fixed thresholds for classification
        self.high_zscore_threshold = high_zscore_threshold
        self.high_ratio_threshold = high_ratio_threshold
        self.extreme_ratio_threshold = extreme_ratio_threshold
        self.extreme_zscore_threshold = extreme_zscore_threshold
        self.low_zscore_threshold = low_zscore_threshold
        self.low_ratio_threshold = low_ratio_threshold
        self.extreme_low_ratio_threshold = extreme_low_ratio_threshold
        self.extreme_low_zscore_threshold = extreme_low_zscore_threshold
        self.confidence_method = confidence_method  # 'min' or 'avg'
    
    def classify_single_day(self, high: float, low: float, close: float, open_price: float,
                           historical_mean_amplitude: float, historical_std_amplitude: float) -> dict:
        """
        Classify volatility for a single trading day
        
        Args:
            high: highest price of the day
            low: lowest price of the day
            close: closing price of the day
            open_price: opening price of the day
            historical_mean_amplitude: mean of (high-low) from historical data
            historical_std_amplitude: std of (high-low) from historical data
            
        Returns:
            dict with volatility classification and metrics
        """
        # Calculate daily amplitude
        daily_amplitude = high - low
        
        # Calculate relative amplitude ratio
        if close == 0:
            relative_ratio = 0
        else:
            relative_ratio = daily_amplitude / close
        
        # Calculate z-score
        if historical_std_amplitude == 0:
            z_score = 0
        else:
            z_score = (daily_amplitude - historical_mean_amplitude) / historical_std_amplitude
        
        # Determine trinary volatility label using combined metrics with extreme value overrides
        # HIGH (2): (z_score > high_zscore_threshold AND relative_ratio >= high_ratio_threshold)
        #           OR relative_ratio >= extreme_ratio_threshold OR z_score > extreme_zscore_threshold
        # LOW (0):  (z_score <= low_zscore_threshold AND relative_ratio < low_ratio_threshold)
        #           OR relative_ratio <= extreme_low_ratio_threshold OR z_score < extreme_low_zscore_threshold
        # MEDIUM (1): Everything else
        high_zscore = z_score > self.high_zscore_threshold
        high_ratio = relative_ratio >= self.high_ratio_threshold
        low_zscore = z_score <= self.low_zscore_threshold
        low_ratio = relative_ratio < self.low_ratio_threshold

        if (high_zscore and high_ratio) or relative_ratio >= self.extreme_ratio_threshold or z_score > self.extreme_zscore_threshold:
            volatility_trinary = 2  # HIGH: both metrics agree OR extreme value
        elif (low_zscore and low_ratio) or relative_ratio <= self.extreme_low_ratio_threshold or z_score < self.extreme_low_zscore_threshold:
            volatility_trinary = 0  # LOW: both metrics agree OR extreme low value
        else:
            volatility_trinary = 1  # MEDIUM: mixed signals or moderate values

        # Calculate confidence (0-1) based on distance from classification boundaries
        # Uses class threshold attributes for all calculations
        z_conf = 0.0
        r_conf = 0.0

        if volatility_trinary == 2:  # HIGH
            # Measure how far into HIGH territory
            if z_score > self.extreme_zscore_threshold:
                # Extreme z-score: confidence from how far past extreme_zscore_threshold
                z_conf = min((z_score - self.extreme_zscore_threshold) / 2.0, 1.0)
            elif relative_ratio >= self.extreme_ratio_threshold:
                # Extreme ratio: confidence from how far past extreme_ratio_threshold
                r_conf = min((relative_ratio - self.extreme_ratio_threshold) / self.extreme_ratio_threshold, 1.0)
            else:
                # Base HIGH: distance from high_zscore_threshold AND high_ratio_threshold boundary
                z_conf = min(max(z_score - self.high_zscore_threshold, 0) / (self.extreme_zscore_threshold - self.high_zscore_threshold), 1.0)
                r_conf = min(max(relative_ratio - self.high_ratio_threshold, 0) / (self.extreme_ratio_threshold - self.high_ratio_threshold), 1.0)
            confidence = max(z_conf, r_conf)

        elif volatility_trinary == 0:  # LOW
            # Measure how far into LOW territory
            if z_score < self.extreme_low_zscore_threshold:
                # Extreme z-score: confidence from how far below extreme_low_zscore_threshold
                z_conf = min(abs(z_score - self.extreme_low_zscore_threshold), 1.0)
            elif relative_ratio <= self.extreme_low_ratio_threshold:
                # Extreme low ratio: confidence from how far below extreme_low_ratio_threshold
                r_conf = min((self.extreme_low_ratio_threshold - relative_ratio) / self.extreme_low_ratio_threshold, 1.0)
            else:
                # Base LOW: distance from low_zscore_threshold AND low_ratio_threshold boundary
                z_conf = min(max(self.low_zscore_threshold - z_score, 0) / abs(self.extreme_low_zscore_threshold - self.low_zscore_threshold), 1.0)
                r_conf = min(max(self.low_ratio_threshold - relative_ratio, 0) / (self.low_ratio_threshold - self.extreme_low_ratio_threshold), 1.0)
            confidence = max(z_conf, r_conf)

        else:  # MEDIUM
            # MEDIUM is between HIGH and LOW boundaries
            # Distance to each boundary (how much change before reclassification)

            # Distance to HIGH boundary (how much z/ratio needs to increase to trigger HIGH)
            z_to_high = self.high_zscore_threshold - z_score
            r_to_high = self.high_ratio_threshold - relative_ratio

            # Distance to LOW boundary (how much z/ratio needs to decrease to trigger LOW)
            z_to_low = z_score - self.low_zscore_threshold
            r_to_low = relative_ratio - self.low_ratio_threshold

            # Normalize by the range
            z_range = self.high_zscore_threshold - self.low_zscore_threshold
            r_range = self.high_ratio_threshold - self.low_ratio_threshold

            # Calculate normalized distances (all positive for MEDIUM points)
            z_dist_high = max(0, z_to_high) / z_range
            z_dist_low = max(0, z_to_low) / z_range
            r_dist_high = max(0, r_to_high) / r_range
            r_dist_low = max(0, r_to_low) / r_range

            # Use configured method for combining distances
            if self.confidence_method == 'avg':
                # Average distance: higher confidence, less sensitive to single boundary
                confidence = (z_dist_high + z_dist_low + r_dist_high + r_dist_low) / 4.0
            elif self.confidence_method == 'min':
                # Minimum distance (default): conservative, measures closest boundary
                confidence = min(z_dist_high, z_dist_low, r_dist_high, r_dist_low)
            else:
                raise ValueError(f"Invalid confidence_method: {self.confidence_method}. Use 'min' or 'avg'.")

        confidence = max(0.0, min(1.0, confidence))

        return {
            'daily_amplitude': daily_amplitude,
            'relative_amplitude_ratio': relative_ratio,
            'z_score': z_score,
            'volatility_trinary': volatility_trinary,
            'confidence': confidence
        }
    
    def process_ticker(self, symbol: str, ticker_data) -> List[VolatilityRecord]:
        """
        Process a single ticker and generate volatility records
        
        Args:
            symbol: stock symbol
            ticker_data: Ticker_Day object from cache
            
        Returns:
            List of VolatilityRecord objects
        """
        records = []
        
        if not ticker_data.date or len(ticker_data.date) == 0:
            return records
        
        # Calculate historical statistics
        amplitudes = np.array(ticker_data.high) - np.array(ticker_data.low)
        historical_mean = np.mean(amplitudes)
        historical_std = np.std(amplitudes)
        
        # Process each day
        for i in range(len(ticker_data.date)):
            try:
                classification = self.classify_single_day(
                    high=ticker_data.high[i],
                    low=ticker_data.low[i],
                    close=ticker_data.close[i],
                    open_price=ticker_data.open[i] if ticker_data.open and len(ticker_data.open) > i else 0,
                    historical_mean_amplitude=historical_mean,
                    historical_std_amplitude=historical_std
                )
                
                record = VolatilityRecord(
                    date=ticker_data.date[i],
                    open_price=ticker_data.open[i] if ticker_data.open and len(ticker_data.open) > i else 0,
                    symbol=symbol,
                    volatility_trinary=classification['volatility_trinary'],
                    relative_amplitude_ratio=classification['relative_amplitude_ratio'],
                    z_score=classification['z_score'],
                    confidence=classification['confidence']
                )
                records.append(record)
            except Exception as e:
                print(f"Error processing {symbol} on {ticker_data.date[i]}: {e}")
                continue
        
        return records
    
    def process_all_tickers(self, cache_dir: str = CACHE_DIR) -> dict:
        """
        Process all ticker files in cache directory
        
        Args:
            cache_dir: path to cache directory
            
        Returns:
            dict mapping symbol to list of VolatilityRecord objects
        """
        all_records = {}
        cache_files = sorted(Path(cache_dir).glob('*.pkl'))
        
        total_files = len(cache_files)
        for idx, pkl_file in enumerate(cache_files):
            try:
                with open(pkl_file, 'rb') as f:
                    ticker_data = pickle.load(f)
                    symbol = ticker_data.symbol
                    
                    records = self.process_ticker(symbol, ticker_data)
                    all_records[symbol] = records
                    
                    print(f"[{idx+1}/{total_files}] Processed {symbol}: {len(records)} records")
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
        
        return all_records
    
    def save_results(self, all_records: dict, output_dir: str = OUTPUT_DIR):
        """
        Save classification results to pickle files in output directory
        
        Args:
            all_records: dict mapping symbol to list of VolatilityRecord objects
            output_dir: path to output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, records in all_records.items():
            if not records:
                continue
            
            output_file = os.path.join(output_dir, f"{symbol}.pkl")
            try:
                with open(output_file, 'wb') as f:
                    pickle.dump(records, f)
                print(f"Saved: {symbol}.pkl with {len(records)} records")
            except Exception as e:
                print(f"Error saving {symbol}: {e}")
    
    def generate_summary_statistics(self, all_records: dict) -> pd.DataFrame:
        """
        Generate summary statistics across all processed data

        Args:
            all_records: dict mapping symbol to list of VolatilityRecord objects

        Returns:
            DataFrame with summary statistics
        """
        summary = []

        for symbol, records in all_records.items():
            if not records:
                continue

            # Count trinary labels
            low_vol_count = sum(1 for r in records if r.volatility_trinary == 0)
            medium_vol_count = sum(1 for r in records if r.volatility_trinary == 1)
            high_vol_count = sum(1 for r in records if r.volatility_trinary == 2)
            total_count = len(records)

            summary.append({
                'symbol': symbol,
                'total_records': total_count,
                'low_volatility_count': low_vol_count,
                'medium_volatility_count': medium_vol_count,
                'high_volatility_count': high_vol_count,
                'low_volatility_ratio': low_vol_count / total_count if total_count > 0 else 0,
                'medium_volatility_ratio': medium_vol_count / total_count if total_count > 0 else 0,
                'high_volatility_ratio': high_vol_count / total_count if total_count > 0 else 0,
                'avg_relative_ratio': np.mean([r.relative_amplitude_ratio for r in records]),
                'avg_z_score': np.mean([abs(r.z_score) for r in records]),
                'avg_confidence': np.mean([r.confidence for r in records])
            })

        return pd.DataFrame(summary)


def main():
    """Main processing pipeline"""
    print("\n" + "="*70)
    print("TRINARY VOLATILITY CLASSIFICATION PIPELINE".center(70))
    print("="*70 + "\n")

    # Initialize classifier
    print("Initializing classifier with fixed thresholds:")
    print("  - HIGH (2):   (z_score > 0.5 AND relative_ratio >= 3%)")
    print("                OR relative_ratio >= 5% OR z_score > 2.0")
    print("  - LOW (0):    (z_score <= -0.5 AND relative_ratio < 1.5%)")
    print("                OR relative_ratio <= 0.75% OR z_score < -1.0")
    print("  - MEDIUM (1): Everything else\n")

    print("Confidence calculation for MEDIUM records:")
    print("  - Method: 'min' = minimum distance to any boundary (conservative, avg ~0.19)")
    print("  - Use confidence_method='avg' for average distance (higher, avg ~0.51)\n")

    classifier = VolatilityClassifier()  # default: confidence_method='min'
    
    # Process all tickers
    print(f"Processing all tickers from cache directory: {CACHE_DIR}")
    print("-" * 70)
    all_records = classifier.process_all_tickers(CACHE_DIR)
    
    print("\n" + "-" * 70)
    print(f"Total stocks processed: {len(all_records)}")
    
    # Calculate total records
    total_records = sum(len(records) for records in all_records.values())
    print(f"Total trading records: {total_records}\n")
    
    # Save results
    print("Saving results to cache_output...")
    print("-" * 70)
    classifier.save_results(all_records, OUTPUT_DIR)
    
    print("\n" + "-" * 70)
    print("Generating summary statistics...")
    
    # Generate summary
    summary_df = classifier.generate_summary_statistics(all_records)

    if summary_df is None or len(summary_df) == 0:
        print("\nNo data to generate summary statistics.")
        print("\n" + "="*70)
        print("PROCESSING COMPLETE (NO DATA)")
        print("="*70 + "\n")
        return

    summary_df = summary_df.sort_values('high_volatility_ratio', ascending=False)

    # Print summary statistics
    print("\nTop 15 stocks by high volatility ratio:")
    print(summary_df[['symbol', 'total_records', 'high_volatility_ratio', 'medium_volatility_ratio', 'low_volatility_ratio']].head(15).to_string(index=False))

    print("\nBottom 15 stocks by high volatility ratio:")
    print(summary_df[['symbol', 'total_records', 'high_volatility_ratio', 'medium_volatility_ratio', 'low_volatility_ratio']].tail(15).to_string(index=False))

    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    overall_low_vol = summary_df['low_volatility_count'].sum()
    overall_medium_vol = summary_df['medium_volatility_count'].sum()
    overall_high_vol = summary_df['high_volatility_count'].sum()
    overall_total = summary_df['total_records'].sum()

    print(f"Low volatility records (0):    {overall_low_vol} ({overall_low_vol/overall_total*100:.2f}%)")
    print(f"Medium volatility records (1): {overall_medium_vol} ({overall_medium_vol/overall_total*100:.2f}%)")
    print(f"High volatility records (2):   {overall_high_vol} ({overall_high_vol/overall_total*100:.2f}%)")
    print(f"Total records: {overall_total}")
    
    print(f"\nAverage relative amplitude ratio: {summary_df['avg_relative_ratio'].mean():.4f} ({summary_df['avg_relative_ratio'].mean()*100:.2f}%)")
    print(f"Average Z-score magnitude: {summary_df['avg_z_score'].mean():.4f}")
    print(f"Average confidence: {summary_df['avg_confidence'].mean():.4f}")
    
    # Save summary
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'volatility_classification_summary.csv'), index=False)
    print(f"\nSummary saved to: volatility_classification_summary.csv")
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
