# Trinary XGBoost Variant Benchmarks

This file compares saved trinary model variants so users can choose by objective.

## Models Saved

- `xgboost/models/original.pkl`
- `xgboost/models/method1_class_weight.pkl`
- `xgboost/models/method4_6_threshold_features.pkl`
- `xgboost/models/method1_4_6_class_weight_threshold.pkl`
- `xgboost/models/method6_chrono_threshold_original_hparams.pkl`
- `xgboost/models/method7_8_hparams_chrono.pkl`
- `xgboost/models/method6_7_8_threshold_hparams_chrono.pkl`

## Benchmark Summary (Test Set)

| Variant | Split | Accuracy | F1 Weighted | F1 Macro | AUC OvR Weighted |
|---|---|---:|---:|---:|---:|
| original | random_stratified | 0.7219 | 0.6783 | 0.4982 | 0.7251 |
| method1_class_weight | random_stratified | 0.6169 | 0.6184 | 0.6235 | 0.7897 |
| method4_6_threshold_features | random_stratified | 0.6805 | 0.6814 | 0.5722 | 0.7252 |
| method1_4_6_class_weight_threshold | random_stratified | 0.6167 | 0.6179 | 0.6232 | 0.7897 |
| method6_chrono_threshold_original_hparams | chronological | 0.6548 | 0.6511 | 0.5733 | 0.7311 |
| method7_8_hparams_chrono | chronological | 0.6889 | 0.6448 | 0.5140 | 0.7301 |
| method6_7_8_threshold_hparams_chrono | chronological | 0.6538 | 0.6502 | 0.5724 | 0.7301 |

## Notes

- `original`: baseline random stratified split with 5 features.
- `method1_class_weight`: baseline + inverse-frequency class weighting multiplied with confidence.
- `method4_6_threshold_features`: engineered features + validation-tuned class thresholds.
- `method1_4_6_class_weight_threshold`: inverse-frequency class weighting plus validation-tuned class thresholds.
- `method6_chrono_threshold_original_hparams`: original hyperparameters + chronological split + validation-tuned class thresholds.
- `method7_8_hparams_chrono`: tuned hyperparameters + chronological split (harder, more realistic).
- `method6_7_8_threshold_hparams_chrono`: validation-tuned thresholds + chronological split + tuned hyperparameters.

## Recommendation

- If optimizing baseline weighted F1 under random split: choose the best `F1 Weighted` among random-split variants.
- If optimizing minority sensitivity: compare `F1 Macro` and class-level confusion matrices externally.
- If optimizing temporal robustness: prefer `method7_8_hparams_chrono` despite lower direct comparability.
