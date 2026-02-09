"""
Demo script to read and display volatility classification results
"""

import pickle
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

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

OUTPUT_DIR = 'cache_output'

def read_and_display_sample():
    """Load and display sample data from cached output"""
    
    output_files = list(Path(OUTPUT_DIR).glob('*.pkl'))
    
    if not output_files:
        print("No pickle files found in cache_output")
        return
    
    # Read first pickle file (AAPL)
    sample_file = None
    for f in output_files:
        if 'AAPL' in f.name:
            sample_file = f
            break
    
    if sample_file is None:
        sample_file = output_files[0]
    
    print(f"Loading sample data from: {sample_file.name}\n")
    
    with open(sample_file, 'rb') as f:
        records = pickle.load(f)
    
    print(f"Total records: {len(records)}")
    print(f"Data structure: VolatilityRecord\n")
    
    # Display first 10 records
    print("First 10 records:")
    print("=" * 120)
    print(f"{'Date':<12} {'Open':<10} {'Symbol':<10} {'Relative Ratio':<16} {'Z-Score':<12} {'High Vol':<12} {'Confidence':<12}")
    print("=" * 120)
    
    for i, record in enumerate(records[:10]):
        print(f"{str(record.date):<12} {record.open_price:<10.2f} {record.symbol:<10} {record.relative_amplitude_ratio:<16.4f} {record.z_score:<12.4f} {str(record.is_high_volatility):<12} {record.confidence:<12.4f}")
    
    print("\n" + "=" * 120)
    print("\nData structure breakdown:")
    print(f"  - date: {type(records[0].date).__name__}")
    print(f"  - open_price: {type(records[0].open_price).__name__}")
    print(f"  - symbol: {type(records[0].symbol).__name__}")
    print(f"  - is_high_volatility: {type(records[0].is_high_volatility).__name__}")
    print(f"  - relative_amplitude_ratio: {type(records[0].relative_amplitude_ratio).__name__}")
    print(f"  - z_score: {type(records[0].z_score).__name__}")
    print(f"  - confidence: {type(records[0].confidence).__name__}")
    
    # Statistics
    high_vol_count = sum(1 for r in records if r.is_high_volatility)
    print(f"\nStatistics for {records[0].symbol}:")
    print(f"  - Total records: {len(records)}")
    print(f"  - High volatility days: {high_vol_count} ({high_vol_count/len(records)*100:.2f}%)")
    print(f"  - Low volatility days: {len(records) - high_vol_count} ({(len(records)-high_vol_count)/len(records)*100:.2f}%)")
    print(f"  - Avg relative amplitude ratio: {sum(r.relative_amplitude_ratio for r in records) / len(records):.4f}")
    print(f"  - Avg absolute z-score: {sum(abs(r.z_score) for r in records) / len(records):.4f}")
    
    # Load summary CSV
    print("\n" + "=" * 120)
    print("Summary Statistics (Top 5 most volatile stocks):\n")
    
    summary_file = Path(OUTPUT_DIR) / 'volatility_classification_summary.csv'
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        print(summary_df.head().to_string(index=False))
        
        print("\n\nSummary Statistics (Top 5 least volatile stocks):\n")
        print(summary_df.tail().to_string(index=False))

if __name__ == '__main__':
    read_and_display_sample()
