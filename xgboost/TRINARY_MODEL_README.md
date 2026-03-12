# Trinary Volatility Classification Model

This document describes the actual trinary XGBoost model used in this repository for daily stock volatility classification, plus the benchmark variants that were trained and compared.

## Task

Predict next-day volatility as a 3-class classification problem:

- `0`: LOW volatility
- `1`: MEDIUM volatility
- `2`: HIGH volatility

All predictors are computed from day `t-1` and earlier, while the label is defined on day `t`, so the training setup is leakage-aware by construction.

## Dataset

Primary dataset:

- `xgboost_dataset_china_stock/trinary_xgboost_training_dataset.pkl`

Dataset properties:

- ~1.5M+ samples
- 998 Chinese stocks
- date range: 2019-01-02 to 2025-12-30
- class distribution is imbalanced: about 11% LOW, 62% MEDIUM, 27% HIGH

### Input Features

Baseline model uses 5 features:

1. `intraday_range`
2. `volume_change_rate`
3. `rolling_historical_volatility`
4. `prev_day_news_count`
5. `prev_day_avg_news_sentiment`

The dataset also includes:

- `target`: trinary label
- `confidence`: sample weight used during training

## Actual Baseline Model

Baseline training script:

- `xgboost/xgb.py`

Benchmark training script:

- `xgboost/TRINARY_train_variants.py`

Baseline XGBoost configuration:

```python
{
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}
```

Training protocol:

- random stratified split
- 80% train, 10% validation, 10% test
- `confidence` used as `sample_weight`
- prediction rule: `argmax` over the 3-class probability output

Main output artifacts:

- `xgboost/volatility_classifier_model.pkl`
- `xgboost/models/original.pkl`
- `xgboost/models/TRINARY_original.pkl`

## Benchmark Variants

The repository also contains benchmark variants trained by `xgboost/TRINARY_train_variants.py`.

### 1. `original`

Baseline model:

- 5 baseline features
- random stratified split
- confidence-weighted training

### 2. `method1_class_weight`

Class-reweighted variant:

- same 5 baseline features
- random stratified split
- sample weight = `confidence * inverse_frequency_class_weight`
- goal: improve minority-class sensitivity

### 3. `method4_6_threshold_features`

Feature engineering plus threshold tuning:

Additional engineered features:

- `range_vol_ratio`
- `volume_change_abs`
- `log_prev_day_news_count`
- `news_sentiment_abs`
- `news_impact`

Details:

- 10 total features
- random stratified split
- confidence-weighted training
- validation-tuned thresholds for LOW and HIGH classes
- best thresholds found:
  - `t0 = 0.25`
  - `t2 = 0.35`

### 4. `method7_8_hparams_chrono`

Chronological robustness variant:

- 5 baseline features
- chronological split by date instead of random split
- tuned hyperparameters:

```python
{
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 8,
    "min_child_weight": 5,
    "gamma": 0.3,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "random_state": 42,
}
```

### 5. `method1_4_6_class_weight_threshold`

Combined class-weighting plus threshold-tuning variant:

- same 5 baseline features as `original`
- random stratified split
- sample weight = `confidence * inverse_frequency_class_weight`
- validation-tuned thresholds applied at inference time
- best thresholds found:
  - `t0 = 0.45`
  - `t2 = 0.50`

This is the direct combination of the class-weighting idea from `method1_class_weight` and the thresholding idea from `method4_6_threshold_features`.

### 6. `method6_7_8_threshold_hparams_chrono`

Combined threshold plus chronological variant:

- uses the 5 baseline features
- uses the chronological split and tuned hyperparameters from `method7_8_hparams_chrono`
- uses confidence sample weights
- tunes LOW/HIGH class thresholds on the chronological validation set
- best thresholds found:
  - `t0 = 0.25`
  - `t2 = 0.40`

This is the direct combination of:

- threshold tuning
- chronological evaluation
- stronger tuned hyperparameters

### 7. `method6_chrono_threshold_original_hparams`

Chronological threshold-only variant with original hyperparameters:

- uses the 5 baseline features
- uses chronological split
- keeps the original baseline hyperparameters unchanged
- uses confidence sample weights
- tunes LOW/HIGH thresholds on the chronological validation set
- best thresholds found:
  - `t0 = 0.25`
  - `t2 = 0.40`

This isolates the effect of threshold tuning under time-ordered evaluation without adding engineered features or stronger hyperparameters.

## Benchmarks

Source:

- `xgboost/benchmarks/TRINARY_benchmark_variants.json`

### Test Set Summary

| Variant | Split | Features | Accuracy | Weighted F1 | Macro F1 | Weighted OvR AUC |
|---|---|---:|---:|---:|---:|---:|
| `original` | random stratified | 5 | 0.7219 | 0.6783 | 0.4982 | 0.7251 |
| `method1_class_weight` | random stratified | 5 | 0.6169 | 0.6184 | 0.6235 | 0.7897 |
| `method4_6_threshold_features` | random stratified | 10 | 0.6805 | 0.6814 | 0.5722 | 0.7252 |
| `method1_4_6_class_weight_threshold` | random stratified | 5 | 0.6167 | 0.6179 | 0.6232 | 0.7897 |
| `method6_chrono_threshold_original_hparams` | chronological | 5 | 0.6548 | 0.6511 | 0.5733 | 0.7311 |
| `method7_8_hparams_chrono` | chronological | 5 | 0.6889 | 0.6448 | 0.5140 | 0.7301 |
| `method6_7_8_threshold_hparams_chrono` | chronological | 5 | 0.6538 | 0.6502 | 0.5724 | 0.7301 |

### Interpretation

- `original` is the strongest simple baseline on headline accuracy among the random-split models.
- `method4_6_threshold_features` gives the best weighted F1 among the random-split benchmark variants.
- `method1_class_weight` improves macro F1 the most, which indicates better balance across minority classes, but it sacrifices overall accuracy.
- `method1_4_6_class_weight_threshold` performs almost identically to class weighting alone, so thresholding does not materially improve the class-weighted setup here.
- `method6_chrono_threshold_original_hparams` improves chronological macro F1 strongly over `method7_8_hparams_chrono`, and it slightly improves weighted F1 and AUC, but it lowers accuracy.
- `method7_8_hparams_chrono` is harder to compare directly because the split is more realistic and more difficult; it is the best option if temporal robustness matters more than random-split headline metrics.
- `method6_7_8_threshold_hparams_chrono` improves chronological macro F1 strongly over `method7_8_hparams_chrono`, but loses accuracy.

## Recommended Model Choice

Use the model based on your objective:

- best simple baseline: `xgboost/models/TRINARY_original.pkl`
- best random-split weighted F1: `xgboost/models/TRINARY_method4_6_threshold_features.pkl`
- best minority-class balance: `xgboost/models/TRINARY_method1_class_weight.pkl`
- combined class-weight + threshold benchmark: `xgboost/models/TRINARY_method1_4_6_class_weight_threshold.pkl`
- threshold-tuned chronological benchmark with original hyperparameters: `xgboost/models/TRINARY_method6_chrono_threshold_original_hparams.pkl`
- best temporal realism: `xgboost/models/TRINARY_method7_8_hparams_chrono.pkl`
- threshold-tuned chronological benchmark: `xgboost/models/TRINARY_method6_7_8_threshold_hparams_chrono.pkl`

If you want one default recommendation for reporting, use:

- `TRINARY_original.pkl` for a clean baseline
- `TRINARY_method7_8_hparams_chrono.pkl` for a more realistic out-of-time benchmark

## Saved Artifacts

Model files:

- `xgboost/models/original.pkl`
- `xgboost/models/method1_class_weight.pkl`
- `xgboost/models/method4_6_threshold_features.pkl`
- `xgboost/models/method1_4_6_class_weight_threshold.pkl`
- `xgboost/models/method6_chrono_threshold_original_hparams.pkl`
- `xgboost/models/method7_8_hparams_chrono.pkl`
- `xgboost/models/method6_7_8_threshold_hparams_chrono.pkl`
- `xgboost/models/TRINARY_original.pkl`
- `xgboost/models/TRINARY_method1_class_weight.pkl`
- `xgboost/models/TRINARY_method4_6_threshold_features.pkl`
- `xgboost/models/TRINARY_method1_4_6_class_weight_threshold.pkl`
- `xgboost/models/TRINARY_method6_chrono_threshold_original_hparams.pkl`
- `xgboost/models/TRINARY_method7_8_hparams_chrono.pkl`
- `xgboost/models/TRINARY_method6_7_8_threshold_hparams_chrono.pkl`

Plots generated by the baseline training flow:

- `result/confusion_matrix_train.png`
- `result/confusion_matrix_validation.png`
- `result/confusion_matrix_test.png`
- `result/roc_curves.png`
- `result/feature_importance.png`

Benchmark comparison plot:

- `xgboost/benchmarks/trinary_variant_benchmarks.png`

## Reproducing Benchmarks

Run:

```bash
uv run python xgboost/TRINARY_train_variants.py
```

This will:

- train all 7 variants
- save model artifacts under `xgboost/models/`
- write benchmark JSON under `xgboost/benchmarks/`
- write a benchmark comparison PNG under `xgboost/benchmarks/`
- regenerate the variant benchmark README

## Notes

- The repository currently contains both lowercase and `TRINARY_`-prefixed model filenames with matching contents.
- The benchmark script writes the main JSON file to `xgboost/benchmarks/trinary_variant_benchmarks.json`; a duplicate uppercase-named benchmark file is also present in the repo.
- For slide decks or reports, avoid comparing random and chronological split results as if they were the same evaluation setting.
