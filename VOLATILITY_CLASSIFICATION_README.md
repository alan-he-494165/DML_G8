# Binary Volatility Classification System

## Overview

This system classifies daily stock volatility as either HIGH or LOW using a combination of:
1. **Relative Amplitude Ratio**: (High - Low) / Close
2. **Z-Score Normalization**: measures deviation from historical mean

## System Components

### 1. Data Processor (`data_processor/volatility_classifier.py`)

**Purpose**: Processes all cached stock data and generates binary volatility labels

**Input**:
- Source: `cache/` directory containing 998 pickle files of Ticker_Day objects

**Output**:
- Destination: `cache_output/` directory
- Format: 998 pickle files (one per stock)
- Each file contains a list of `VolatilityRecord` objects

**Data Structure**:
Each record contains:
```python
@dataclass
class VolatilityRecord:
    date: pd.Timestamp                    # Trading date
    open_price: float                     # Opening price
    symbol: str                           # Stock ticker
    is_high_volatility: bool              # Classification: True=HIGH, False=LOW
    relative_amplitude_ratio: float       # (High-Low)/Close
    z_score: float                        # Standardized deviation
    confidence: float                     # 0.0-1.0 confidence score
```

### 2. Classification Rules

**Primary Metric**: Relative Amplitude Ratio
- **Threshold**: 2.0%
- If `(High - Low) / Close > 0.02` → HIGH volatility

**Secondary Metric**: Z-Score
- **Threshold**: |z-score| > 1.5
- If deviates significantly from historical norm → HIGH volatility

**Final Decision**:
```
is_high_volatility = (relative_ratio > 0.02) OR (|z_score| > 1.5)
```

### 3. Overall Statistics

**Total Processed Data**:
- Total records: 1,561,647
- High volatility records: 1,042,890 (66.78%)
- Low volatility records: 518,757 (33.22%)

**Stock Classification** (by high volatility ratio):

#### Top 5 Most Volatile Stocks
| Symbol | High Vol % | Avg Ratio | Z-Score |
|--------|-----------|-----------|---------|
| SNDK   | 99.55%   | 7.18%    | 0.745   |
| COIN   | 99.49%   | 6.70%    | 0.710   |
| 688506 | 99.45%   | 5.80%    | 0.711   |
| 688017 | 98.84%   | 6.00%    | 0.735   |
| 688521 | 98.84%   | 5.42%    | 0.632   |

#### Top 5 Least Volatile Stocks
| Symbol | High Vol % | Avg Ratio | Z-Score |
|--------|-----------|-----------|---------|
| 601006 | 15.19%   | 1.41%    | 0.699   |
| 601988 | 17.66%   | 1.38%    | 0.712   |
| 601328 | 18.07%   | 1.44%    | 0.708   |
| 601288 | 18.48%   | 1.42%    | 0.692   |
| 601398 | 18.54%   | 1.45%    | 0.710   |

## Usage

### 1. Generate Classifications

```bash
python data_processor/volatility_classifier.py
```

This will:
- Load all 998 stock pickle files from `cache/`
- Calculate daily volatility metrics
- Apply classification rules
- Save results to `cache_output/`
- Generate summary CSV report

### 2. Read Results

```bash
python data_processor/read_volatility_output.py
```

This demonstrates:
- How to load pickled VolatilityRecord objects
- Display sample data
- Access individual fields
- Generate statistics

### 3. Access Output Files

Load classification results programmatically:

```python
import pickle
from pathlib import Path

# Load single stock
with open('cache_output/AAPL.pkl', 'rb') as f:
    records = pickle.load(f)

# Iterate through records
for record in records:
    print(f"Date: {record.date}")
    print(f"Symbol: {record.symbol}")
    print(f"Is High Volatility: {record.is_high_volatility}")
    print(f"Relative Ratio: {record.relative_amplitude_ratio:.4f}")
    print(f"Z-Score: {record.z_score:.4f}")
```

### 4. Load Summary Statistics

```python
import pandas as pd

summary_df = pd.read_csv('cache_output/volatility_classification_summary.csv')
print(summary_df.head())
```

## File Structure

```
DML_G8/
├── cache/
│   ├── 000001.pkl
│   ├── AAPL.pkl
│   └── ... (998 total)
│
├── cache_output/
│   ├── 000001.pkl (VolatilityRecord list)
│   ├── AAPL.pkl (VolatilityRecord list)
│   ├── volatility_classification_summary.csv
│   └── ... (998 total)
│
├── data_processor/
│   ├── volatility_classifier.py (Main processor)
│   └── read_volatility_output.py (Demo reader)
│
└── volatility_analysis.py (Original analysis)
```

## Configuration

You can modify thresholds in `volatility_classifier.py`:

```python
classifier = VolatilityClassifier(
    relative_ratio_threshold=0.02,    # Adjust from 0.015 to 0.025
    z_score_threshold=1.5              # Adjust from 1.0 to 2.0
)
```

### Recommended Settings

- **Conservative** (fewer false positives):
  - relative_ratio_threshold: 0.025 (2.5%)
  - z_score_threshold: 2.0

- **Moderate** (balanced):
  - relative_ratio_threshold: 0.020 (2.0%)  ← Default
  - z_score_threshold: 1.5

- **Aggressive** (fewer false negatives):
  - relative_ratio_threshold: 0.015 (1.5%)
  - z_score_threshold: 1.0

## Example Output

```
Date: 2020-01-06 00:00:00
Open: 70.89
Symbol: AAPL
Relative Amplitude Ratio: 0.0240 (2.40%)
Z-Score: -0.8671
Is High Volatility: True
Confidence: 0.5777
```

## Performance Metrics

- **Processing Time**: ~2-3 minutes for all 998 stocks
- **Total Output Size**: ~150-200 MB (pickled)
- **Records per Stock**: 700-1700 (varies by data availability)
- **Mean Z-Score Magnitude**: 0.697
- **Mean Confidence Score**: 0.595

## Next Steps

The `is_high_volatility` label can be used as:
1. **Target Variable** for machine learning models
2. **Feature Engineering** input for volatility prediction
3. **Risk Assessment** in trading strategies
4. **Anomaly Detection** in market behavior

---

**Created**: February 2026  
**System**: Binary Volatility Classification using Relative Amplitude + Z-Score
