# Trinary Chronological No-News Experiment

This folder contains a separate trinary XGBoost experiment where:

- all evaluation is chronological
- all news-based inputs are removed
- existing `xgboost/` and `xgboost_trinary_chrono/` outputs are left unchanged

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

### Base no-news features

Only market-derived features are used:

1. `intraday_range`
2. `volume_change_rate`
3. `rolling_historical_volatility`

### Engineered no-news features

For threshold-feature variants, two additional market-only engineered features are added:

1. `range_vol_ratio`
2. `volume_change_abs`

So the engineered no-news feature set has 5 total features.

## Split Protocol

All variants use:

- chronological train / validation / test split
- earlier dates for training
- later dates for validation
- latest dates for test

This is intended to better reflect real forecasting conditions than random splitting.

## Training Script

- `train_trinary_chrono_no_news.py`

## Variants

### 1. `original_chrono_no_news`

- 3 base no-news features
- original baseline hyperparameters
- chronological split
- no threshold tuning

### 2. `class_weight_chrono_no_news`

- 3 base no-news features
- class weighting using inverse-frequency weights multiplied by `confidence`
- chronological split

### 3. `threshold_features_chrono_no_news`

- 5 engineered no-news features
- chronological split
- validation-tuned LOW/HIGH thresholds

### 4. `class_weight_threshold_chrono_no_news`

- 3 base no-news features
- class weighting
- chronological split
- validation-tuned LOW/HIGH thresholds

### 5. `tuned_hparams_chrono_no_news`

- 3 base no-news features
- chronological split
- tuned hyperparameters
- no threshold tuning

### 6. `threshold_tuned_hparams_chrono_no_news`

- 3 base no-news features
- chronological split
- tuned hyperparameters
- validation-tuned LOW/HIGH thresholds

## Test Results

Source:

- `chrono_no_news_benchmarks.json`

Chronological persistence baseline:

- accuracy `0.6450`
- weighted F1 `0.6493`
- macro F1 `0.5867`

| Variant | Accuracy | Weighted F1 | Macro F1 | Weighted OvR AUC | Δ Acc vs Persistence | Δ F1w vs Persistence | Δ F1m vs Persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original_chrono_no_news` | 0.6858 | 0.6397 | 0.5055 | 0.7292 | +0.0408 | -0.0097 | -0.0812 |
| `class_weight_chrono_no_news` | 0.6423 | 0.6442 | 0.6168 | 0.8090 | -0.0027 | -0.0051 | +0.0301 |
| `threshold_features_chrono_no_news` | 0.6488 | 0.6490 | 0.5774 | 0.7295 | +0.0038 | -0.0003 | -0.0093 |
| `class_weight_threshold_chrono_no_news` | 0.6444 | 0.6452 | 0.6172 | 0.8090 | -0.0006 | -0.0041 | +0.0305 |
| `tuned_hparams_chrono_no_news` | 0.6852 | 0.6390 | 0.5045 | 0.7280 | +0.0402 | -0.0103 | -0.0822 |
| `threshold_tuned_hparams_chrono_no_news` | 0.6489 | 0.6493 | 0.5782 | 0.7280 | +0.0039 | +0.0000 | -0.0085 |

## Interpretation

- best headline accuracy: `original_chrono_no_news`
- best weighted F1: `threshold_tuned_hparams_chrono_no_news`
- best macro F1: `class_weight_threshold_chrono_no_news`
- best weighted OvR AUC: `class_weight_chrono_no_news` and `class_weight_threshold_chrono_no_news`

Main pattern:

- removing news features does not destroy performance
- class weighting helps macro F1 much more than accuracy
- threshold tuning improves class balance more than headline accuracy
- compared with persistence, no-news models show the same tradeoff pattern as the with-news run
- accuracy-oriented models beat persistence on accuracy but lose on macro F1
- class-weighted models beat persistence on macro F1 but do not beat it on weighted F1

## Comparison With Chronological Experiment Using News

Reference folder:

- `xgboost_trinary_chrono/`

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
- macro F1 changes are small; no-news is slightly better for the threshold-feature variants
- overall, market-only features remain strong, but news features still add a small positive edge on headline performance

## Thresholds Learned

- `threshold_features_chrono_no_news`: `t0 = 0.25`, `t2 = 0.35`
- `class_weight_threshold_chrono_no_news`: `t0 = 0.45`, `t2 = 0.50`
- `threshold_tuned_hparams_chrono_no_news`: `t0 = 0.25`, `t2 = 0.35`

## Artifacts

### Models

Stored in:

- `models/`

### Benchmark files

- `chrono_no_news_benchmarks.json`
- `plots/chrono_no_news_benchmarks.png`

### Confusion matrices

- `plots/confusion_matrix_original_chrono_no_news.png`
- `plots/confusion_matrix_class_weight_chrono_no_news.png`
- `plots/confusion_matrix_threshold_features_chrono_no_news.png`
- `plots/confusion_matrix_class_weight_threshold_chrono_no_news.png`
- `plots/confusion_matrix_tuned_hparams_chrono_no_news.png`
- `plots/confusion_matrix_threshold_tuned_hparams_chrono_no_news.png`
