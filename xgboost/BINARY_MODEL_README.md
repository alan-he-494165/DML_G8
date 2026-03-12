# Binary Volatility Classification Model

This document describes the binary volatility classification setup that still exists in this repository: the binary label-generation pipeline and the binary XGBoost training dataset.

It also notes an important repository-status detail:

- the current maintained `xgboost/` training pipeline is trinary
- the current `xgboost/volatility_classifier_model.pkl` is also trinary, not binary

So this README documents the binary model design and dataset assets that exist in the repo, but not a currently maintained binary XGBoost training script under `xgboost/`.

## Task

Binary volatility prediction:

- `0`: LOW volatility
- `1`: HIGH volatility

The binary label is stored in the XGBoost dataset as `target`.

## Binary Label Definition

Source:

- `data_util/VOLATILITY_CLASSIFICATION_README.md`
- `data_util/data_processor/volatility_classifier.py`

The binary volatility classifier uses two metrics:

1. relative amplitude ratio
2. z-score of daily amplitude relative to historical amplitude

### Metrics

Relative amplitude ratio:

```text
(High - Low) / Close
```

Z-score:

```text
(daily_amplitude - historical_mean_amplitude) / historical_std_amplitude
```

### Binary Rule

From [volatility_classifier.py](C:\Users\Alan He\Desktop\DML_G8\data_util\data_processor\volatility_classifier.py):

```text
is_high_volatility = (relative_ratio > 0.02) OR (abs(z_score) > 1.5)
```

Default thresholds:

- relative ratio threshold: `0.02`
- z-score threshold: `1.5`

That means:

- HIGH volatility if intraday amplitude exceeds 2% of close, or if amplitude is far from the stock’s historical norm
- LOW volatility otherwise

## Binary Data Pipeline

### 1. Binary label generation

Binary label generator:

- `data_util/data_processor/volatility_classifier.py`

Output:

- `cache_output/` binary volatility-label files

Each label record contains:

- `date`
- `open_price`
- `symbol`
- `is_high_volatility`
- `relative_amplitude_ratio`
- `z_score`
- `confidence`

### 2. Binary XGBoost dataset creation

Dataset creator:

- `data_util/data_processor/create_xgboost_dataset.py`

Primary binary training dataset:

- `xgboost_dataset/xgboost_training_dataset.pkl`

## Binary Dataset

Verified dataset file:

- `xgboost_dataset/xgboost_training_dataset.pkl`

Observed columns:

- `date`
- `symbol`
- `close_today`
- `intraday_range`
- `volume_change_rate`
- `rolling_historical_volatility`
- `target`
- `confidence`
- `relative_amplitude_ratio`
- `z_score`

Observed dataset size:

- `1,540,689` rows

Observed target values:

- binary classes `0` and `1`

Observed positive-class share:

- `target.mean() = 0.6703`

This indicates the current binary dataset is majority-HIGH under the binary labeling rule above.

## Binary Feature Set

From [create_xgboost_dataset.py](C:\Users\Alan He\Desktop\DML_G8\data_util\data_processor\create_xgboost_dataset.py), the intended XGBoost feature set is:

1. `intraday_range`
2. `volume_change_rate`
3. `rolling_historical_volatility`
4. previous-day news count
5. previous-day average news sentiment

The dataset also retains auxiliary columns such as:

- `close_today`
- `relative_amplitude_ratio`
- `z_score`
- `confidence`

For model training, `confidence` is intended to be used as sample weight.

## Data Leakage Design

From [create_xgboost_dataset.py](C:\Users\Alan He\Desktop\DML_G8\data_util\data_processor\create_xgboost_dataset.py):

- features from day `t-1` are used to predict the label on day `t`
- this is explicitly intended to avoid same-day leakage

That design mirrors the trinary pipeline later adopted in `xgboost/`.

## Current Repository Status

Important current-state note:

- `xgboost/xgb.py` is now a trinary training script
- `xgboost/volatility_classifier_model.pkl` is now a trinary model
- the loaded saved model reports:
  - `objective = multi:softprob`
  - `classes_ = [0, 1, 2]`

So there is no current binary XGBoost model artifact under `xgboost/` that should be treated as authoritative.

## What Still Exists for Binary

Binary assets that still exist and are usable:

- binary volatility-labeling logic in `data_util/`
- binary volatility-label readme in `data_util/`
- binary XGBoost training dataset in `xgboost_dataset/xgboost_training_dataset.pkl`

## Recommended Next Step

If you want a maintained binary model again, the cleanest approach is:

1. create a dedicated binary training script under `xgboost/`
2. train on `xgboost_dataset/xgboost_training_dataset.pkl`
3. save a binary model artifact under a binary-specific name
4. generate binary benchmarks and plots separately from the trinary workflow

## Relevant Files

Binary pipeline:

- `data_util/VOLATILITY_CLASSIFICATION_README.md`
- `data_util/data_processor/volatility_classifier.py`
- `data_util/data_processor/create_xgboost_dataset.py`
- `xgboost_dataset/xgboost_training_dataset.pkl`

Current trinary pipeline for comparison:

- `xgboost/xgb.py`
- `xgboost/TRINARY_train_variants.py`
- `xgboost/TRINARY_MODEL_README.md`

## Summary

The repository still contains the binary volatility-labeling system and the binary XGBoost dataset, but the maintained model-training code in `xgboost/` has already moved to trinary classification. This README should therefore be read as documentation of the binary setup and binary data assets, not as documentation of the current saved `xgboost` model artifact.
