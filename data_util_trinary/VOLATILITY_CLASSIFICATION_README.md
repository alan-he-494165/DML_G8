# Trinary Volatility Classification System

## Overview

This system classifies daily stock volatility into three levels: LOW, MEDIUM, or HIGH using a combination of:
1. **Relative Amplitude Ratio**: (High - Low) / Close
2. **Z-Score Normalization**: measures deviation from historical mean

## System Components

### 1. Data Processor (`data_processor/volatility_classifier.py`)

**Purpose**: Processes all cached stock data and generates trinary volatility labels

**Input**:
- Source: `cache/` directory containing pickle files of Ticker_Day objects

**Output**:
- Destination: `cache_output/` directory
- Format: pickle files (one per stock)
- Each file contains a list of `VolatilityRecord` objects

**Data Structure**:
Each record contains:
```python
@dataclass
class VolatilityRecord:
    date: pd.Timestamp                    # Trading date
    open_price: float                     # Opening price
    symbol: str                           # Stock ticker
    volatility_trinary: int               # Classification: 0=LOW, 1=MEDIUM, 2=HIGH
    relative_amplitude_ratio: float       # (High-Low)/Close
    z_score: float                        # Standardized deviation
    confidence: float                     # 0.0-1.0 confidence score
```

### 2. Classification Rules

**Trinary Classification (Combined Metrics with Extreme Value Overrides)**:

| Label | Value | Condition |
|-------|-------|-----------|
| HIGH | 2 | `(z_score > 0.5 AND relative_ratio >= 3%)` OR `relative_ratio >= 5%` OR `z_score > 2.0` |
| MEDIUM | 1 | Everything else (not clearly HIGH or LOW) |
| LOW | 0 | `(z_score <= -0.5 AND relative_ratio < 1.5%)` OR `relative_ratio <= 0.75%` OR `z_score < -1.0` |

**Metrics Used**:
- **Z-Score**: Measures how many standard deviations the current day's amplitude is from the stock's historical mean
- **Relative Amplitude Ratio**: (High - Low) / Close, expressed as a percentage

**Logic**:
- **HIGH (2)**: Base rule requires both metrics to agree (z-score > 0.5 AND ratio >= 3%), with overrides for extreme values (ratio >= 5% or z-score > 2.0)
- **LOW (0)**: Base rule requires both metrics to indicate calm (z-score <= -0.5 AND ratio < 1.5%), with overrides for extreme low values (ratio <= 0.75% or z-score < -1.0)
- **MEDIUM (1)**: All other cases - mixed signals or moderate values

**Secondary Metric**: Z-Score
- **HIGH threshold**: |z-score| > 1.5
- **MEDIUM threshold**: |z-score| > 0.5

### 3. Overall Statistics

**Total Processed Data**:
- Records are classified into three categories: LOW, MEDIUM, and HIGH volatility

**Stock Classification** (by high volatility ratio):

#### Top 5 Most Volatile Stocks
| Symbol | High Vol % | Medium Vol % | Low Vol % | Avg Ratio |
|--------|-----------|-------------|-----------|-----------|
| ...    | ...       | ...         | ...       | ...       |

#### Top 5 Least Volatile Stocks
| Symbol | High Vol % | Medium Vol % | Low Vol % | Avg Ratio |
|--------|-----------|-------------|-----------|-----------|
| ...    | ...       | ...         | ...       | ...       |

## Usage

### 1. Generate Classifications

```bash
python data_processor/volatility_classifier.py
```

This will:
- Load all stock pickle files from `cache/`
- Calculate daily volatility metrics
- Apply trinary classification rules
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
    print(f"Volatility: {record.volatility_trinary} (0=LOW, 1=MEDIUM, 2=HIGH)")
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
│   └── ...
│
├── cache_output/
│   ├── 000001.pkl (VolatilityRecord list)
│   ├── AAPL.pkl (VolatilityRecord list)
│   ├── volatility_classification_summary.csv
│   └── ...
│
├── data_processor/
│   ├── volatility_classifier.py (Main processor)
│   ├── read_volatility_output.py (Demo reader)
│   └── create_xgboost_dataset.py (XGBoost dataset creator)
│
└── volatility_data_analysis/
    ├── volatility_analysis.py
    └── volatility_indicators_analysis.py
```

## Configuration

The trinary classification thresholds are hardcoded in the `classify_single_day` method:

```python
# Combined criteria with extreme value overrides
high_zscore = z_score > 0.5
high_ratio = relative_ratio >= 0.03
low_zscore = z_score <= -0.5
low_ratio = relative_ratio < 0.015

if (high_zscore and high_ratio) or relative_ratio >= 0.05 or z_score > 2.0:
    volatility_trinary = 2  # HIGH
elif (low_zscore and low_ratio) or relative_ratio <= 0.0075 or z_score < -1.0:
    volatility_trinary = 0  # LOW
else:
    volatility_trinary = 1  # MEDIUM
```

### Threshold Components

**Base Rules** (require both metrics to agree):
- **HIGH**: z_score > 0.5 AND relative_ratio >= 3%
- **LOW**: z_score <= -0.5 AND relative_ratio < 1.5%

**Extreme Value Overrides** (override base rules):
- **HIGH override**: relative_ratio >= 5% OR z_score > 2.0
- **LOW override**: relative_ratio <= 0.75% OR z_score < -1.0

## Example Output

```
Date: 2020-01-06 00:00:00
Open: 70.89
Symbol: AAPL
Relative Amplitude Ratio: 0.0240 (2.40%)
Z-Score: -0.8671
Volatility: 1 (MEDIUM)
Confidence: 0.5777
```

## Performance Metrics

- **Processing Time**: ~2-3 minutes for all stocks
- **Total Output Size**: ~150-200 MB (pickled)
- **Records per Stock**: 700-1700 (varies by data availability)
- **Mean Z-Score Magnitude**: ~0.7
- **Mean Confidence Score**: ~0.16

## Target Distribution (Actual from China Stock Data)

- **Low Volatility (0)**: ~11%
- **Medium Volatility (1)**: ~62%
- **High Volatility (2)**: ~27%

## Next Steps

The `volatility_trinary` label can be used as:
1. **Target Variable** for machine learning models (multi-class classification)
2. **Feature Engineering** input for volatility prediction
3. **Risk Assessment** in trading strategies with granular risk levels
4. **Anomaly Detection** in market behavior

---

**Created**: February 2026
**Updated**: March 2026 (Trinary Classification)
**System**: Trinary Volatility Classification using Relative Amplitude + Z-Score
