# XGBoost Training Dataset (Trinary Classification)

## Overview

A complete training dataset for **trinary** stock volatility prediction using XGBoost. Generated from China stocks with volatility labels: **LOW (0)**, **MEDIUM (1)**, **HIGH (2)**.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | ~1.5+ million |
| Unique Stocks | 998 |
| Date Range | 2019-01-02 to 2025-12-30 |
| Low Volatility (0) | ~11% |
| Medium Volatility (1) | ~62% |
| High Volatility (2) | ~27% |

## Features

### 1. **Intraday Range** (from day t-1)
- Formula: `(High - Low) / Close`
- Represents the percentage price range within a trading day
- Uses yesterday's data to predict today's volatility

### 2. **Volume Change Rate** (from day t-1)
- Formula: `(Volume_today - Volume_yesterday) / Volume_yesterday`
- Measures how volume changed from previous day
- Uses yesterday's volume change

### 3. **Rolling Historical Volatility** (up to day t-1)
- Calculated: Standard deviation of daily returns over 20-day rolling window
- Captures volatility trends from recent trading history
- Uses returns up to yesterday

### 4. **Previous Day News Count** (from day t-1)
- Count of news items for the stock on the previous trading day
- 0 if no news found

### 5. **Previous Day Avg News Sentiment** (from day t-1)
- Average sentiment score of news items for the stock on the previous trading day
- 0 if no news found

## Target Variable

**`target`**: Trinary classification label
- **0**: LOW Volatility (~11%)
- **1**: MEDIUM Volatility (~62%)
- **2**: HIGH Volatility (~27%)

### Classification Rules

Trinary classification using combined metrics with extreme value overrides:

| Label | Value | Condition |
|-------|-------|-----------|
| HIGH | 2 | `(z_score > 0.5 AND relative_ratio >= 3%)` OR `relative_ratio >= 5%` OR `z_score > 2.0` |
| MEDIUM | 1 | Everything else (not clearly HIGH or LOW) |
| LOW | 0 | `(z_score <= -0.5 AND relative_ratio < 1.5%)` OR `relative_ratio <= 0.75%` OR `z_score < -1.0` |

**Metrics Used**:
- **Z-Score**: `(daily_amplitude - historical_mean) / historical_std`
- **Relative Amplitude Ratio**: `(High - Low) / Close`

## Sample Weights

**`confidence`**: Sample weighting factor (0.0 - 1.0)
- Higher confidence = more reliable classification
- Use as `sample_weight` in XGBoost training for weighted learning

### Using Confidence as Weights

```python
import pandas as pd
import xgboost as xgb

data = pd.read_pickle('../xgboost_dataset_china_stock/trinary_xgboost_training_dataset.pkl')

X = data[['intraday_range', 'volume_change_rate', 'rolling_historical_volatility',
          'prev_day_news_count', 'prev_day_avg_news_sentiment']]
y = data['target']
sample_weights = data['confidence']

model = xgb.XGBClassifier(objective='multi:softprob', num_class=3)
model.fit(X, y, sample_weight=sample_weights)
```

## File Formats

### 1. Pickle Format (Primary)
```
../xgboost_dataset_china_stock/trinary_xgboost_training_dataset.pkl
```
**Type**: pandas DataFrame

**Columns**: symbol, date, intraday_range, volume_change_rate, rolling_historical_volatility,
             prev_day_news_count, prev_day_avg_news_sentiment, target, confidence

**Usage**:
```python
import pandas as pd
data = pd.read_pickle('../xgboost_dataset_china_stock/trinary_xgboost_training_dataset.pkl')
```

## Data Preparation for XGBoost

### Load and Split (Multi-class Classification)
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load dataset
data = pd.read_pickle('../xgboost_dataset_china_stock/trinary_xgboost_training_dataset.pkl')

# Prepare features and target
X = data[['intraday_range', 'volume_change_rate', 'rolling_historical_volatility',
          'prev_day_news_count', 'prev_day_avg_news_sentiment']]
y = data['target']
sample_weights = data['confidence']

# Train-test split (80-20)
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42
)

# Train model with sample weights (multi-class)
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,  # LOW, MEDIUM, HIGH
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train, sample_weight=w_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")
```

### Cross-Validation with Weights
```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score

cv_results = cross_validate(
    model, X, y,
    sample_weight=weights,
    cv=5,
    scoring='accuracy'
)
print(f"CV Scores: {cv_results['test_score'].mean():.4f} +/- {cv_results['test_score'].std():.4f}")
```

## Feature Importance

Example to check which features matter most:
```python
import matplotlib.pyplot as plt

feature_names = ['intraday_range', 'volume_change_rate', 'rolling_historical_volatility',
                 'prev_day_news_count', 'prev_day_avg_news_sentiment']

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance Score')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()
```

## Confidence Score Interpretation

The `confidence` score indicates how reliable each sample's label is:

| Confidence Range | Interpretation |
|------------------|----------------|
| 0.90 - 1.00 | Very High - Clear, unambiguous labels |
| 0.70 - 0.90 | High - Reliable labels |
| 0.50 - 0.70 | Medium - Some uncertainty |
| 0.30 - 0.50 | Low - Borderline cases |
| < 0.30 | Very Low - High uncertainty |

**Recommendation**: Consider filtering out samples with `confidence < 0.5` for more conservative model training.

```python
# Use only high-confidence samples
high_conf_mask = data['confidence'] > 0.7
X_clean = X[high_conf_mask]
y_clean = y[high_conf_mask]
weights_clean = weights[high_conf_mask]

model.fit(X_clean, y_clean, sample_weight=weights_clean)
```

## Class Imbalance Handling

The dataset has imbalanced classes (~11% LOW, ~62% MEDIUM, ~27% HIGH). XGBoost handles this with:

1. **Sample Weights**: Already included as `confidence` column
2. **Class Weights**:
   ```python
   from sklearn.utils.class_weight import compute_class_weight

   class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
   # Use class_weights to adjust sample_weight
   ```

## Data Quality Checks

- No missing values in primary features
- All 5 features have complete coverage
- Dates span 7 years (2019-2025)
- 998 unique stocks represented
- Confidence scores well-distributed

## Data Leakage Prevention

**Important**: All features are calculated from day **(t-1)** to predict the label on day **(t)**:

| Feature | Source Day |
|---------|------------|
| Intraday Range | t-1 (yesterday) |
| Volume Change Rate | t-1 vs t-2 |
| Rolling Historical Volatility | Up to t-1 |
| Previous Day News Count | t-1 |
| Previous Day Avg News Sentiment | t-1 |
| **Target (volatility label)** | **t (today)** |

This ensures no data leakage - we never use today's data to predict today's label.

## Integration Example

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_pickle('../xgboost_dataset_china_stock/trinary_xgboost_training_dataset.pkl')

# Prepare features and target
features = ['intraday_range', 'volume_change_rate', 'rolling_historical_volatility',
            'prev_day_news_count', 'prev_day_avg_news_sentiment']
X = data[features]
y = data['target']
sample_weights = data['confidence']

# Split data
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42
)

# Train XGBoost model with sample weights
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)
model.fit(X_train, y_train, sample_weight=w_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['LOW', 'MEDIUM', 'HIGH']))
print(confusion_matrix(y_test, y_pred))
```

---

**Generated**: February 2026
**Updated**: March 2026 (Trinary Classification)
**Dataset Size**: 1.5 Million+ samples
**Features**: 5 input features + confidence weighting
**Target**: Trinary volatility classification (LOW=0, MEDIUM=1, HIGH=2)
