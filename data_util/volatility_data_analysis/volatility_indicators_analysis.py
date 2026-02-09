"""
Volatility Indicators Analysis and Recommendations
====================================================

This script analyzes different volatility indicators for daily stock movements
and recommends the best metric for a binary volatility classification model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

CACHE_DIR = 'cache'

def analyze_daily_volatility_indicators():
    """Analyze different daily volatility indicators"""
    
    print("\n" + "="*70)
    print("VOLATILITY INDICATOR ANALYSIS FOR BINARY CLASSIFICATION")
    print("="*70)
    
    print("\n" + "OPTION 1: ABSOLUTE AMPLITUDE (High - Low)".center(70, "-"))
    print("""
Pros:
  - Simplest to calculate
  - Directly represents price range
  - Easy to understand
  
Cons:
  - Affected by absolute price level
  - High-price stocks naturally have larger amplitudes
  - Not comparable across different stocks
  
Example:
  - Stock A (price ~$300): High-Low = $50 (is this large?)
  - Stock B (price ~$50):  High-Low = $10 (is this large?)
  
Recommendation: NOT SUITABLE for binary model without normalization
""")

    print("\n" + "OPTION 2: RELATIVE AMPLITUDE RATIO (High-Low) / Close".center(70, "-"))
    print("""
Pros:
  - Normalizes by price level
  - Comparable across stocks
  - Represents percentage volatility
  - Range typically 0-10% for most stocks
  
Cons:
  - Slightly more complex
  - Division by zero edge case (rare)
  
Formula: (High - Low) / Close × 100%

Typical ranges from our analysis:
  - Low volatility stocks: 0.1% - 1%
  - Medium volatility stocks: 1% - 5%
  - High volatility stocks: 5%+
  
Recommendation: GOOD CHOICE for single day classification
""")

    print("\n" + "OPTION 3: Z-SCORE NORMALIZED AMPLITUDE".center(70, "-"))
    print("""
Uses: Historical statistics to normalize daily amplitude

Formula: Z-Score = (Daily_Amplitude - Historical_Mean_Amplitude) 
                   / Historical_Std_Amplitude

Pros:
  - Accounts for each stock's historical volatility pattern
  - Detects anomalies relative to that stock's typical behavior
  - Values > 1.5 indicate unusual volatility
  
Cons:
  - Requires historical baseline for each stock
  - More complex to implement
  
Interpretation:
  - Z-Score > 1.5: Large volatility for THIS stock
  - Z-Score > 2.0: Extremely large volatility
  - Z-Score < -1.5: Unusually small volatility
  
Recommendation: BEST CHOICE for detecting daily anomalies
""")

    print("\n" + "OPTION 4: PERCENTILE-BASED CLASSIFICATION".center(70, "-"))
    print("""
Uses: Historical percentile distribution

Steps:
  1. Calculate daily relative amplitude (High-Low) / Close
  2. Compare to historical distribution percentiles
  3. If > 75th percentile: HIGH volatility
  4. If < 25th percentile: LOW volatility
  5. Otherwise: MEDIUM volatility

Pros:
  - Intuitive (using percentiles)
  - Stock-specific baseline
  - Clear threshold definition
  
Cons:
  - Requires historical data
  - Less sensitive to genuine anomalies
  
Recommendation: GOOD for stable classification
""")

    print("\n" + "OPTION 5: COMBINED METRIC (RECOMMENDED)".center(70, "-"))
    print("""
Best approach: Use Multiple Indicators

Binary Classification Model Output: HIGH/LOW Volatility

Suggested Implementation:
  
  1. Primary Indicator: Relative Amplitude Ratio
     threshold = 0.02 (2%)  # Adjustable based on stock characteristics
     
  2. Secondary Indicator: Z-Score
     If |Z-Score| > 1.5: mark as HIGH volatility
     
  3. Ternary Enhancement (if needed later):
     HIGH:   (High-Low)/Close > 0.03 OR Z-Score > 1.5
     MEDIUM: (High-Low)/Close 0.01-0.03 OR -0.5 < Z-Score < 1.5
     LOW:    (High-Low)/Close < 0.01 OR Z-Score < -0.5

Advantages:
  ✓ Works across all stocks
  ✓ Accounts for individual stock characteristics
  ✓ Detects both unusually large AND unusually small movements
  ✓ Flexible and scalable
  ✓ Easy to interpret and adjust thresholds

Recommended Binary Model Output Format:
  {
    'date': pd.Timestamp,
    'symbol': str,
    'high_low_amplitude': float,
    'relative_amplitude_ratio': float,
    'historical_mean_amplitude': float,
    'z_score': float,
    'is_high_volatility': bool,  # 1 if HIGH, 0 if LOW
    'confidence': float,  # 0.0-1.0
  }
""")

    print("\n" + "="*70)
    print("IMPLEMENTATION RECOMMENDATION")
    print("="*70)
    print("""
For a production binary volatility classifier, I recommend:

PRIMARY METRIC: Relative Amplitude Ratio (High - Low) / Close

Implementation:
  - Calculate historical statistics for each stock
  - For each trading day:
    1. Calculate daily_amplitude = high - low
    2. Calculate relative_ratio = daily_amplitude / close
    3. Calculate z_score using historical mean/std
    4. Output: is_high_volatility = (relative_ratio > threshold) OR (z_score > 1.5)

Threshold tuning:
  - Conservative (fewer false positives): threshold = 0.025 (2.5%)
  - Moderate (balanced):                  threshold = 0.020 (2.0%)
  - Aggressive (fewer false negatives):   threshold = 0.015 (1.5%)

Expected distribution (from our data):
  - ~25-30% of days would be classified as HIGH volatility
  - ~70-75% of days would be classified as LOW volatility
""")

    print("\n" + "="*70)
    
def demonstrate_with_sample_data():
    """Show practical example with real data"""
    print("\nPRACTICAL EXAMPLE WITH REAL DATA")
    print("="*70 + "\n")
    
    # Load sample data
    cache_files = list(Path(CACHE_DIR).glob('*.pkl'))[:5]  # Load first 5 stocks as sample
    
    examples = []
    
    for pkl_file in cache_files:
        try:
            with open(pkl_file, 'rb') as f:
                ticker = pickle.load(f)
                
            if not ticker.high or len(ticker.high) < 30:
                continue
                
            # Calculate daily amplitudes
            amplitudes = np.array(ticker.high) - np.array(ticker.low)
            closes = np.array(ticker.close)
            relative_ratios = amplitudes / closes * 100  # Convert to percentage
            
            # Calculate statistics
            mean_amp = np.mean(amplitudes)
            std_amp = np.std(amplitudes)
            
            # Get last 5 days as examples
            for i in range(-5, 0):
                idx = i
                if -idx <= len(amplitudes):
                    daily_amp = amplitudes[idx]
                    relative_ratio = relative_ratios[idx]
                    z_score = (daily_amp - mean_amp) / (std_amp + 1e-8)
                    
                    # Determine classification
                    is_high_vol = (relative_ratio > 2.0) or (z_score > 1.5)
                    
                    examples.append({
                        'symbol': ticker.symbol,
                        'daily_amplitude': daily_amp,
                        'relative_ratio_%': relative_ratio,
                        'z_score': z_score,
                        'is_high_volatility': is_high_vol
                    })
        except:
            continue
    
    if examples:
        df_examples = pd.DataFrame(examples)
        print(df_examples.to_string(index=False))
        print("\nThresholds used:")
        print("  - Relative Ratio: 2.0% → HIGH volatility")
        print("  - Z-Score: > 1.5 → HIGH volatility")
        print("  - Otherwise: LOW volatility")

if __name__ == '__main__':
    analyze_daily_volatility_indicators()
    demonstrate_with_sample_data()
