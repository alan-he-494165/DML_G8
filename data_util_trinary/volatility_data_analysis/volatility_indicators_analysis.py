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

    print("\n" + "OPTION 5: COMBINED METRIC WITH EXTREME VALUE OVERRIDES (RECOMMENDED)".center(70, "-"))
    print("""
Best approach: Combine Z-Score and Relative Amplitude Ratio with Extreme Value Overrides

Trinary Classification Model Output: LOW (0), MEDIUM (1), HIGH (2)

Implementation:
  - Calculate daily relative amplitude ratio: (High - Low) / Close
  - Calculate z-score using historical mean/std of amplitude

  Classification Rules (Combined Criteria with Extreme Overrides):
    HIGH (2):   (z_score > 0.5 AND relative_ratio >= 3%)
                OR relative_ratio >= 5% OR z_score > 2.0

    LOW (0):    (z_score <= -0.5 AND relative_ratio < 1.5%)
                OR relative_ratio <= 0.75% OR z_score < -1.0

    MEDIUM (1): Everything else (mixed signals or moderate values)

Advantages:
  ✓ Accounts for stock-specific behavior (z-score)
  ✓ Uses absolute volatility level (relative ratio)
  ✓ Base rule requires agreement between both metrics
  ✓ Extreme value overrides catch true outliers
  ✓ More robust than single-metric approaches

Recommended Trinary Model Output Format:
  {
    'date': pd.Timestamp,
    'symbol': str,
    'high_low_amplitude': float,
    'relative_amplitude_ratio': float,
    'historical_mean_amplitude': float,
    'historical_std_amplitude': float,
    'z_score': float,
    'volatility_trinary': int,  # 0=LOW, 1=MEDIUM, 2=HIGH
    'confidence': float,  # 0.0-1.0
  }
""")

    print("\n" + "="*70)
    print("IMPLEMENTATION RECOMMENDATION")
    print("="*70)
    print("""
For a production trinary volatility classifier, I recommend:

COMBINED METRIC: Z-Score + Relative Amplitude Ratio with Extreme Value Overrides

Implementation:
  - For each trading day:
    1. Calculate daily_amplitude = high - low
    2. Calculate relative_ratio = daily_amplitude / close
    3. Calculate z_score using historical mean/std
    4. Classify using combined criteria:

       HIGH (2):   (z_score > 0.5 AND relative_ratio >= 3%)
                   OR relative_ratio >= 5% OR z_score > 2.0
       MEDIUM (1): Everything else
       LOW (0):    (z_score <= -0.5 AND relative_ratio < 1.5%)
                   OR relative_ratio <= 0.75% OR z_score < -1.0

Note: Thresholds are hardcoded in the VolatilityClassifier class.
      To adjust, modify the threshold constants in __init__().

Expected distribution (from China stock data):
  - ~11% of days classified as LOW volatility (0)
  - ~62% of days classified as MEDIUM volatility (1)
  - ~27% of days classified as HIGH volatility (2)
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

                    # Determine trinary classification using combined criteria with extreme overrides
                    high_zscore = z_score > 0.5
                    high_ratio = relative_ratio >= 3.0
                    low_zscore = z_score <= -0.5
                    low_ratio = relative_ratio < 1.5

                    if (high_zscore and high_ratio) or relative_ratio >= 5.0 or z_score > 2.0:
                        volatility_trinary = 2  # HIGH
                    elif (low_zscore and low_ratio) or relative_ratio <= 0.75 or z_score < -1.0:
                        volatility_trinary = 0  # LOW
                    else:
                        volatility_trinary = 1  # MEDIUM

                    examples.append({
                        'symbol': ticker.symbol,
                        'daily_amplitude': daily_amp,
                        'relative_ratio_%': relative_ratio,
                        'z_score': z_score,
                        'volatility_trinary': volatility_trinary
                    })
        except:
            continue
    
    if examples:
        df_examples = pd.DataFrame(examples)
        print(df_examples.to_string(index=False))
        print("\nThresholds used (Combined Criteria with Extreme Overrides):")
        print("  - HIGH (2):   (z_score > 0.5 AND relative_ratio >= 3%) OR ratio >= 5% OR z_score > 2.0")
        print("  - LOW (0):    (z_score <= -0.5 AND relative_ratio < 1.5%) OR ratio <= 0.75% OR z_score < -1.0")
        print("  - MEDIUM (1): Everything else")

if __name__ == '__main__':
    analyze_daily_volatility_indicators()
    demonstrate_with_sample_data()
