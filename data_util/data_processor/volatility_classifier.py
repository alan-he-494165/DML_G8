"""
Binary Volatility Classifier
Processes all stock data and generates volatility labels using z-score and relative amplitude ratio
Output: pickle files with Date, Open, Symbol, and is_high_volatility label
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

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache_output')


@dataclass
class VolatilityRecord:
    """Data structure for volatility classification output"""
    date: pd.Timestamp
    open_price: float
    symbol: str
    is_high_volatility: bool  # 1 if HIGH, 0 if LOW
    relative_amplitude_ratio: float
    z_score: float
    confidence: float


class VolatilityClassifier:
    """
    Binary volatility classifier using relative amplitude ratio and z-score
    """
    
    def __init__(self, relative_ratio_threshold: float = 0.02, z_score_threshold: float = 1.5):
        """
        Initialize classifier with thresholds
        
        Args:
            relative_ratio_threshold: threshold for (High-Low)/Close (default: 2%)
            z_score_threshold: threshold for z-score magnitude (default: 1.5)
        """
        self.relative_ratio_threshold = relative_ratio_threshold
        self.z_score_threshold = z_score_threshold
    
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
        
        # Determine high volatility
        is_high_volatility = (relative_ratio > self.relative_ratio_threshold) or (abs(z_score) > self.z_score_threshold)
        
        # Calculate confidence (0-1)
        # Confidence increases with how far the metric is from threshold
        ratio_confidence = min(abs(relative_ratio - self.relative_ratio_threshold) / (self.relative_ratio_threshold + 0.001), 1.0)
        z_confidence = min(abs(z_score) / (self.z_score_threshold + 0.001), 1.0)
        confidence = max(ratio_confidence, z_confidence)
        
        return {
            'daily_amplitude': daily_amplitude,
            'relative_amplitude_ratio': relative_ratio,
            'z_score': z_score,
            'is_high_volatility': is_high_volatility,
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
                    is_high_volatility=classification['is_high_volatility'],
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
            
            high_vol_count = sum(1 for r in records if r.is_high_volatility)
            total_count = len(records)
            
            summary.append({
                'symbol': symbol,
                'total_records': total_count,
                'high_volatility_count': high_vol_count,
                'high_volatility_ratio': high_vol_count / total_count if total_count > 0 else 0,
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
    print(f"  - Relative amplitude ratio: > 2.0%")
    print(f"  - Z-score magnitude: > 1.5\n")
    
    classifier = VolatilityClassifier(
        relative_ratio_threshold=0.02,
        z_score_threshold=1.5
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
    summary_df = summary_df.sort_values('high_volatility_ratio', ascending=False)
    
    # Print summary statistics
    print("\nTop 15 stocks by high volatility ratio:")
    print(summary_df.head(15).to_string(index=False))
    
    print("\nBottom 15 stocks by high volatility ratio:")
    print(summary_df.tail(15).to_string(index=False))
    
    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    overall_high_vol = summary_df['high_volatility_count'].sum()
    overall_total = summary_df['total_records'].sum()
    overall_ratio = overall_high_vol / overall_total if overall_total > 0 else 0
    
    print(f"Total high volatility records: {overall_high_vol}")
    print(f"Total records: {overall_total}")
    print(f"High volatility ratio: {overall_ratio:.2%}")
    
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
