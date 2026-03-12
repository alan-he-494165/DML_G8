# Trinary Chronological Experiment

This folder contains a separate trinary XGBoost experiment where:

- all evaluation is chronological
- market and news features are both available
- outputs are isolated from the main `xgboost/` folder

## Dataset

Source dataset:

- `xgboost_dataset_china_stock/trinary_xgboost_training_dataset.pkl`

Target:

- `0`: LOW volatility
- `1`: MEDIUM volatility
- `2`: HIGH volatility

Sample weights:

- `confidence`

## Features

### Base features

1. `intraday_range`
2. `volume_change_rate`
3. `rolling_historical_volatility`
4. `prev_day_news_count`
5. `prev_day_avg_news_sentiment`

### Engineered feature variants

For threshold-feature variants, the following are added:

1. `range_vol_ratio`
2. `volume_change_abs`
3. `log_prev_day_news_count`
4. `news_sentiment_abs`
5. `news_impact`

So the engineered set has 10 total features.

## Split Protocol

All variants use:

- chronological train / validation / test split
- earlier dates for training
- later dates for validation
- latest dates for test

## Training Script

- `train_trinary_chrono_variants.py`

## Variants

### 1. `original_chrono`

- 5 base features
- original baseline hyperparameters
- chronological split
- no threshold tuning

### 2. `class_weight_chrono`

- 5 base features
- class weighting using inverse-frequency weights multiplied by `confidence`
- chronological split

### 3. `threshold_features_chrono`

- 10 engineered features
- chronological split
- validation-tuned LOW/HIGH thresholds

### 4. `class_weight_threshold_chrono`

- 5 base features
- class weighting
- chronological split
- validation-tuned LOW/HIGH thresholds

### 5. `tuned_hparams_chrono`

- 5 base features
- chronological split
- tuned hyperparameters
- no threshold tuning

### 6. `threshold_tuned_hparams_chrono`

- 5 base features
- chronological split
- tuned hyperparameters
- validation-tuned LOW/HIGH thresholds

## Test Results

Source:

- `chrono_variant_benchmarks.json`

Chronological persistence baseline:

- accuracy `0.6450`
- weighted F1 `0.6493`
- macro F1 `0.5867`

| Variant | Accuracy | Weighted F1 | Macro F1 | Weighted OvR AUC | Δ Acc vs Persistence | Δ F1w vs Persistence | Δ F1m vs Persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original_chrono` | 0.6893 | 0.6454 | 0.5150 | 0.7311 | +0.0442 | -0.0039 | -0.0717 |
| `class_weight_chrono` | 0.6452 | 0.6470 | 0.6205 | 0.8105 | +0.0001 | -0.0023 | +0.0338 |
| `threshold_features_chrono` | 0.6552 | 0.6516 | 0.5737 | 0.7309 | +0.0102 | +0.0022 | -0.0131 |
| `class_weight_threshold_chrono` | 0.6460 | 0.6469 | 0.6200 | 0.8105 | +0.0009 | -0.0024 | +0.0333 |
| `tuned_hparams_chrono` | 0.6889 | 0.6448 | 0.5140 | 0.7301 | +0.0439 | -0.0045 | -0.0727 |
| `threshold_tuned_hparams_chrono` | 0.6538 | 0.6502 | 0.5724 | 0.7301 | +0.0087 | +0.0009 | -0.0143 |

## Interpretation

- best headline accuracy: `original_chrono`
- best weighted F1: `threshold_features_chrono`
- best macro F1: `class_weight_chrono`
- best weighted OvR AUC: `class_weight_chrono` and `class_weight_threshold_chrono`

Main pattern:

- chronological evaluation lowers headline accuracy relative to random-split results
- class weighting helps macro F1 much more than accuracy
- threshold tuning improves class balance more than headline accuracy
- engineered features help weighted F1 in chronological testing
- compared with persistence, no variant improves everything at once
- the accuracy-oriented models beat persistence on accuracy but lose on macro F1
- the class-weighted models beat persistence on macro F1 but are roughly flat on accuracy and slightly worse on weighted F1

## Thresholds Learned

- `threshold_features_chrono`: `t0 = 0.25`, `t2 = 0.40`
- `class_weight_threshold_chrono`: `t0 = 0.45`, `t2 = 0.50`
- `threshold_tuned_hparams_chrono`: `t0 = 0.25`, `t2 = 0.40`

## Comparison With Chronological No-News Experiment

Reference folder:

- `xgboost_trinary_chrono_no_news/`

Direct comparison of matched variants:

| With News | Accuracy | Weighted F1 | Macro F1 | No News | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|---|---:|---:|---:|
| `original_chrono` | 0.6893 | 0.6454 | 0.5150 | `original_chrono_no_news` | 0.6858 | 0.6397 | 0.5055 |
| `class_weight_chrono` | 0.6452 | 0.6470 | 0.6205 | `class_weight_chrono_no_news` | 0.6423 | 0.6442 | 0.6168 |
| `threshold_features_chrono` | 0.6552 | 0.6516 | 0.5737 | `threshold_features_chrono_no_news` | 0.6488 | 0.6490 | 0.5774 |
| `class_weight_threshold_chrono` | 0.6460 | 0.6469 | 0.6200 | `class_weight_threshold_chrono_no_news` | 0.6444 | 0.6452 | 0.6172 |
| `tuned_hparams_chrono` | 0.6889 | 0.6448 | 0.5140 | `tuned_hparams_chrono_no_news` | 0.6852 | 0.6390 | 0.5045 |
| `threshold_tuned_hparams_chrono` | 0.6538 | 0.6502 | 0.5724 | `threshold_tuned_hparams_chrono_no_news` | 0.6489 | 0.6493 | 0.5782 |

Comparison summary:

- including news gives slightly better accuracy in every matched pair
- including news also gives slightly better weighted F1 in most matched pairs
- macro F1 differences are small; no-news is slightly better on the threshold-feature variants
- overall, news adds a modest but consistent gain on headline performance

## Artifacts

### Models

Stored in:

- `models/`

### Benchmark files

- `chrono_variant_benchmarks.json`
- `plots/chrono_variant_benchmarks.png`

### Confusion matrices

- `plots/confusion_matrix_original_chrono.png`
- `plots/confusion_matrix_class_weight_chrono.png`
- `plots/confusion_matrix_threshold_features_chrono.png`
- `plots/confusion_matrix_class_weight_threshold_chrono.png`
- `plots/confusion_matrix_tuned_hparams_chrono.png`
- `plots/confusion_matrix_threshold_tuned_hparams_chrono.png`
