# XGBoost Training Dataset

## Overview

A complete training dataset for binary stock volatility prediction using XGBoost. Generated from 998 stocks with 1.5+ million trading records spanning from 2019 to 2025.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,541,687 |
| Unique Stocks | 998 |
| Date Range | 2019-01-02 to 2025-12-30 |
| High Volatility (1) | 1,033,418 (67.03%) |
| Low Volatility (0) | 508,269 (32.97%) |

## Features

### 1. **Intraday Range** 
- Formula: `(High - Low) / Close`
- Represents the percentage price range within a trading day
- Mean: 0.0315 (3.15%)
- Range: 0.0000 - 1.2127
- Missing: 0.00%

### 2. **Volume Change Rate**
- Formula: `(Volume_today - Volume_yesterday) / Volume_yesterday`
- Measures how volume changed from previous day
- Mean: 0.0886 (8.86%)
- Median: 0.0105 (1.05%)
- Range: -1.0 to 15,420.0
- Missing: 0.00%

### 3. **Rolling Historical Volatility**
- Calculated: Standard deviation of daily returns over 20-day rolling window
- Captures volatility trends from recent trading history
- Mean: 0.0223 (2.23%)
- Range: 0.0001 - 0.4416
- Missing: 0.00%

## Target Variable

**`target`**: Binary classification label
- **1**: High Volatility (67.03% of samples)
- **0**: Low Volatility (32.97% of samples)

Classification Rule:
```
is_high_volatility = (relative_ratio > 0.02) OR (|z_score| > 1.5)
```

## Sample Weights

**`confidence`**: Sample weighting factor (0.0 - 1.0)
- Higher confidence = more reliable classification
- Use as `sample_weight` in XGBoost training for weighted learning
- Mean: 0.5965
- Range: 0.0006 - 1.0000

### Using Confidence as Weights

```python
import pandas as pd
import xgboost as xgb

data = pd.read_csv('data_for_process/xgboost_dataset/xgboost_training_dataset.csv')

X = data[['intraday_range', 'volume_change_rate', 'rolling_historical_volatility']]
y = data['target']
sample_weights = data['confidence']

model = xgb.XGBClassifier()
model.fit(X, y, sample_weight=sample_weights)
```

## File Formats

### 1. CSV Format
```
data_for_process/xgboost_dataset/xgboost_training_dataset.csv
```
**Columns**: date, symbol, open, high, low, close, intraday_range, volume_change_rate, rolling_historical_volatility, target, confidence, z_score, relative_amplitude_ratio

**Usage**:
```python
import pandas as pd
data = pd.read_csv('data_for_process/xgboost_dataset/xgboost_training_dataset.csv')
```

### 2. Pickle Format
```
data_for_process/xgboost_dataset/xgboost_training_dataset.pkl
```
**Type**: pandas DataFrame

**Usage**:
```python
import pandas as pd
data = pd.read_pickle('data_for_process/xgboost_dataset/xgboost_training_dataset.pkl')
```

### 3. NPZ Format (Recommended for XGBoost)
```
data_for_process/xgboost_dataset/xgboost_training_data.npz
```
**Contains**:
- `X`: Feature matrix (1,541,687 × 3)
- `y`: Target labels (1,541,687)
- `weights`: Sample confidence scores (1,541,687)
- `symbols`: Stock symbols (1,541,687)
- `dates`: Trading dates (1,541,687)

**Usage**:
```python
import numpy as np
import xgboost as xgb

data = np.load('data_for_process/xgboost_dataset/xgboost_training_data.npz')
X = data['X']
y = data['y']
weights = data['weights']
symbols = data['symbols']
dates = data['dates']

model = xgb.XGBClassifier()
model.fit(X, y, sample_weight=weights)
```

## Data Preparation for XGBoost

### Load and Split
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load data
data = np.load('data_for_process/xgboost_dataset/xgboost_training_data.npz')
X = data['X']
y = data['y']
weights = data['weights']

# Train-test split (80-20)
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# Train model with sample weights
model = xgb.XGBClassifier(
    objective='binary:logistic',
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def weighted_score(y_true, y_pred, sample_weight):
    return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

cv_results = cross_validate(
    model, X, y, 
    sample_weight=weights,
    cv=5,
    scoring='accuracy'
)
print(f"CV Scores: {cv_results['test_score'].mean():.4f} ± {cv_results['test_score'].std():.4f}")
```

## Feature Importance

Example to check which features matter most:
```python
import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    'feature': ['intraday_range', 'volume_change_rate', 'rolling_historical_volatility'],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance Score')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
```

## Confidence Score Interpretation

The `confidence` score indicates how reliable each sample's label is:

| Confidence Range | Interpretation | Count |
|------------------|---|---|
| 0.90 - 1.00 | Very High | Clear, unambiguous labels |
| 0.70 - 0.90 | High | Reliable labels |
| 0.50 - 0.70 | Medium | Some uncertainty |
| 0.30 - 0.50 | Low | Borderline cases |
| < 0.30 | Very Low | High uncertainty |

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

The dataset is imbalanced (67% positive, 33% negative). XGBoost handles this with:

1. **Sample Weights**: Already included as `weights` column
2. **Scale Pos Weight**:
   ```python
   scale_pos_weight = (y == 0).sum() / (y == 1).sum()  # 0.492
   model = xgb.XGBClassifier(
       scale_pos_weight=scale_pos_weight
   )
   ```

3. **Custom Thresholds**:
   ```python
   # Adjust probability threshold
   if model.predict_proba(X)[:, 1] > 0.4:  # Instead of 0.5
       prediction = 1
   ```

## Data Quality Checks

✅ No missing values in features  
✅ All 3 features have complete coverage  
✅ Dates span 7 years (2019-2025)  
✅ 998 unique stocks represented  
✅ Confidence scores well-distributed  

## Integration with xgb.py

To use this dataset in `xgboost/xgb.py`:

```python
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load dataset
data = np.load('data_for_process/xgboost_dataset/xgboost_training_data.npz')
X = data['X']
y = data['y']
weights = data['weights']

# Split data
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# Train XGBoost model with sample weights
model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100)
model.fit(X_train, y_train, sample_weight=w_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
```

---

**Generated**: February 2026  
**Dataset Size**: 1.5 Million+ samples  
**Features**: 3 technical features + confidence weighting  
**Target**: Binary volatility classification
