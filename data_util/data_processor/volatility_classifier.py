"""
Multi-class Volatility Classifier
Processes all stock data and generates volatility labels using z-score and relative amplitude ratio
Output: pickle files with Date, Open, Symbol, and volatility_level label (0=LOW, 1=MEDIUM, 2=HIGH)
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum
from typing import List


class VolatilityLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

# Add parent directory to path to import fetcher module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache_output')


@dataclass
class VolatilityRecord:
    """Data structure for volatility classification output"""
    date: pd.Timestamp
    open_price: float
    symbol: str
    volatility_level: VolatilityLevel  # 0=LOW, 1=MEDIUM, 2=HIGH
    relative_amplitude_ratio: float
    z_score: float
    confidence: float


class VolatilityClassifier:
    """
    Binary volatility classifier using relative amplitude ratio and z-score
    """
    
    def __init__(self, ratio_thresholds: tuple = (0.02, 0.04), z_thresholds: tuple = (1.5, 2.5)):
        """
        Initialize classifier with two thresholds per metric for 3-class output.

        Args:
            ratio_thresholds: (low_mid, mid_high) boundaries for (High-Low)/Close
                              default: (2%, 4%) → LOW ≤2%, MEDIUM 2-4%, HIGH >4%
            z_thresholds: (low_mid, mid_high) boundaries for |z-score|
                          default: (1.5, 2.5) → LOW ≤1.5, MEDIUM 1.5-2.5, HIGH >2.5
        """
        self.ratio_low, self.ratio_high = ratio_thresholds
        self.z_low, self.z_high = z_thresholds

    def _classify_level(self, value: float, low_thresh: float, high_thresh: float) -> VolatilityLevel:
        """Classify a single metric value into LOW/MEDIUM/HIGH."""
        if value > high_thresh:
            return VolatilityLevel.HIGH
        elif value > low_thresh:
            return VolatilityLevel.MEDIUM
        return VolatilityLevel.LOW
    
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

        # Classify each metric independently, take the more extreme level
        ratio_level = self._classify_level(relative_ratio, self.ratio_low, self.ratio_high)
        z_level = self._classify_level(abs(z_score), self.z_low, self.z_high)
        volatility_level = max(ratio_level, z_level)

        # Confidence (0-1): how far the sample is from its nearest threshold boundary.
        # Samples near a boundary get low confidence; clear cases get high confidence.
        ratio_min_dist = min(abs(relative_ratio - self.ratio_low), abs(relative_ratio - self.ratio_high))
        z_min_dist = min(abs(abs(z_score) - self.z_low), abs(abs(z_score) - self.z_high))
        ratio_confidence = min(ratio_min_dist / (self.ratio_low + 0.001), 1.0)
        z_confidence = min(z_min_dist / (self.z_low + 0.001), 1.0)
        confidence = max(ratio_confidence, z_confidence)

        return {
            'daily_amplitude': daily_amplitude,
            'relative_amplitude_ratio': relative_ratio,
            'z_score': z_score,
            'volatility_level': int(volatility_level),
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
                    volatility_level=VolatilityLevel(classification['volatility_level']),
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
            
            low_count  = sum(1 for r in records if r.volatility_level == VolatilityLevel.LOW)
            med_count  = sum(1 for r in records if r.volatility_level == VolatilityLevel.MEDIUM)
            high_count = sum(1 for r in records if r.volatility_level == VolatilityLevel.HIGH)
            total_count = len(records)

            summary.append({
                'symbol': symbol,
                'total_records': total_count,
                'low_count': low_count,
                'medium_count': med_count,
                'high_count': high_count,
                'low_ratio': low_count / total_count if total_count > 0 else 0,
                'medium_ratio': med_count / total_count if total_count > 0 else 0,
                'high_ratio': high_count / total_count if total_count > 0 else 0,
                'avg_relative_ratio': np.mean([r.relative_amplitude_ratio for r in records]),
                'avg_z_score': np.mean([abs(r.z_score) for r in records]),
                'avg_confidence': np.mean([r.confidence for r in records])
            })
        
        return pd.DataFrame(summary)


def main():
    """Main processing pipeline"""
    print("\n" + "="*70)
    print("BINARY VOLATILITY CLASSIFICIATION PIPELINE".center(70))
    print("="*70 + "\n")
    
    # Initialize classifier
    print("Initializing classifier with thresholds:")
    print(f"  - Relative amplitude ratio: LOW ≤2.0%, MEDIUM 2-4%, HIGH >4.0%")
    print(f"  - Z-score magnitude:        LOW ≤1.5,  MEDIUM 1.5-2.5, HIGH >2.5\n")

    classifier = VolatilityClassifier(
        ratio_thresholds=(0.02, 0.04),
        z_thresholds=(1.5, 2.5),
    )
    
    # Process all tickers
    print("Processing all tickers from cache...")
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
    summary_df = summary_df.sort_values('high_ratio', ascending=False)

    # Print summary statistics
    print("\nTop 15 stocks by high volatility ratio:")
    print(summary_df.head(15).to_string(index=False))

    print("\nBottom 15 stocks by high volatility ratio:")
    print(summary_df.tail(15).to_string(index=False))

    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    overall_low  = summary_df['low_count'].sum()
    overall_med  = summary_df['medium_count'].sum()
    overall_high = summary_df['high_count'].sum()
    overall_total = summary_df['total_records'].sum()

    print(f"Total records:            {overall_total}")
    print(f"  LOW  volatility:        {overall_low}  ({overall_low/overall_total:.2%})")
    print(f"  MEDIUM volatility:      {overall_med}  ({overall_med/overall_total:.2%})")
    print(f"  HIGH volatility:        {overall_high}  ({overall_high/overall_total:.2%})")
    
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
