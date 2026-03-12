"""
Train and benchmark trinary XGBoost variants for easy comparison.

Variants:
- original: baseline protocol
- method1_class_weight: confidence * inverse-frequency class weights
- method4_6_threshold_features: feature engineering + threshold tuning
- method7_8_hparams_chrono: tuned hyperparams + chronological split
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "xgboost_dataset_china_stock" / "trinary_xgboost_training_dataset.pkl"
MODELS_DIR = ROOT / "xgboost" / "models"
BENCH_DIR = ROOT / "xgboost" / "benchmarks"
README_PATH = ROOT / "xgboost" / "README_VARIANTS.md"


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    w_train: np.ndarray
    w_val: np.ndarray
    w_test: np.ndarray
    feature_cols: list[str]
    split_type: str


def load_df() -> pd.DataFrame:
    with open(DATASET_PATH, "rb") as f:
        return pickle.load(f)


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "intraday_range",
        "volume_change_rate",
        "rolling_historical_volatility",
        "prev_day_news_count",
        "prev_day_avg_news_sentiment",
    ]
    return df[cols].copy()


def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = base_features(df)
    eps = 1e-8
    out["range_vol_ratio"] = out["intraday_range"] / (out["rolling_historical_volatility"] + eps)
    out["volume_change_abs"] = np.abs(out["volume_change_rate"])
    out["log_prev_day_news_count"] = np.log1p(out["prev_day_news_count"])
    out["news_sentiment_abs"] = np.abs(out["prev_day_avg_news_sentiment"])
    out["news_impact"] = out["prev_day_news_count"] * out["news_sentiment_abs"]
    return out


def random_split(X: np.ndarray, y: np.ndarray, w: np.ndarray, val_size=0.1, test_size=0.1) -> tuple:
    X_train_val, X_test, y_train_val, y_test, w_train_val, w_test = train_test_split(
        X, y, w, test_size=test_size, random_state=42, stratify=y
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_train_val, y_train_val, w_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test


def chronological_split(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, w: np.ndarray, val_size=0.1, test_size=0.1) -> tuple:
    dates = pd.to_datetime(df["date"]).to_numpy()
    uniq = np.array(sorted(np.unique(dates)))
    n_dates = len(uniq)
    train_end = int(round(n_dates * (1.0 - val_size - test_size)))
    val_end = int(round(n_dates * (1.0 - test_size)))
    train_end = max(1, min(train_end, n_dates - 2))
    val_end = max(train_end + 1, min(val_end, n_dates - 1))

    train_dates = set(uniq[:train_end])
    val_dates = set(uniq[train_end:val_end])
    test_dates = set(uniq[val_end:])
    train_mask = np.array([d in train_dates for d in dates])
    val_mask = np.array([d in val_dates for d in dates])
    test_mask = np.array([d in test_dates for d in dates])
    return (
        X[train_mask],
        X[val_mask],
        X[test_mask],
        y[train_mask],
        y[val_mask],
        y[test_mask],
        w[train_mask],
        w[val_mask],
        w[test_mask],
    )


def predict_with_thresholds(y_proba: np.ndarray, t0=0.30, t2=0.30) -> np.ndarray:
    y_pred = np.argmax(y_proba, axis=1)
    p0 = y_proba[:, 0]
    p2 = y_proba[:, 2]
    hit0 = p0 >= t0
    hit2 = p2 >= t2
    both = hit0 & hit2
    only0 = hit0 & ~hit2
    only2 = hit2 & ~hit0
    y_pred[only0] = 0
    y_pred[only2] = 2
    y_pred[both] = np.where(p0[both] >= p2[both], 0, 2)
    return y_pred


def tune_thresholds(y_val: np.ndarray, y_val_proba: np.ndarray, sample_weight: np.ndarray) -> dict:
    grid = np.arange(0.20, 0.61, 0.05)
    best = {"t0": 0.30, "t2": 0.30, "macro_f1": -1.0}
    for t0 in grid:
        for t2 in grid:
            pred = predict_with_thresholds(y_val_proba, t0=t0, t2=t2)
            score = f1_score(y_val, pred, average="macro", zero_division=0, sample_weight=sample_weight)
            if score > best["macro_f1"]:
                best = {"t0": float(t0), "t2": float(t2), "macro_f1": float(score)}
    return best


def eval_split(y_true, y_pred, y_proba, sw) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sw)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sw)),
        "auc_ovr_weighted": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", sample_weight=sw)),
    }


def train_variant(name: str, split: SplitData, params: dict, thresholds: dict | None = None) -> dict:
    model = xgb.XGBClassifier(**params)
    model.fit(split.X_train, split.y_train, sample_weight=split.w_train, verbose=False)

    p_train = model.predict_proba(split.X_train)
    p_val = model.predict_proba(split.X_val)
    p_test = model.predict_proba(split.X_test)

    if thresholds is None:
        y_train_pred = np.argmax(p_train, axis=1)
        y_val_pred = np.argmax(p_val, axis=1)
        y_test_pred = np.argmax(p_test, axis=1)
        th_used = None
    else:
        y_train_pred = predict_with_thresholds(p_train, thresholds["t0"], thresholds["t2"])
        y_val_pred = predict_with_thresholds(p_val, thresholds["t0"], thresholds["t2"])
        y_test_pred = predict_with_thresholds(p_test, thresholds["t0"], thresholds["t2"])
        th_used = thresholds

    metrics = {
        "train": eval_split(split.y_train, y_train_pred, p_train, split.w_train),
        "validation": eval_split(split.y_val, y_val_pred, p_val, split.w_val),
        "test": eval_split(split.y_test, y_test_pred, p_test, split.w_test),
        "thresholds": th_used,
        "split_type": split.split_type,
        "feature_count": len(split.feature_cols),
        "params": params,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    return metrics


def build_readme(results: dict):
    def fmt(v: float) -> str:
        return f"{v:.4f}"

    lines = []
    lines.append("# Trinary XGBoost Variant Benchmarks")
    lines.append("")
    lines.append("This file compares saved trinary model variants so users can choose by objective.")
    lines.append("")
    lines.append("## Models Saved")
    lines.append("")
    lines.append("- `xgboost/models/original.pkl`")
    lines.append("- `xgboost/models/method1_class_weight.pkl`")
    lines.append("- `xgboost/models/method4_6_threshold_features.pkl`")
    lines.append("- `xgboost/models/method7_8_hparams_chrono.pkl`")
    lines.append("")
    lines.append("## Benchmark Summary (Test Set)")
    lines.append("")
    lines.append("| Variant | Split | Accuracy | F1 Weighted | F1 Macro | AUC OvR Weighted |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for k in ["original", "method1_class_weight", "method4_6_threshold_features", "method7_8_hparams_chrono"]:
        t = results[k]["test"]
        lines.append(
            f"| {k} | {results[k]['split_type']} | {fmt(t['accuracy'])} | {fmt(t['f1_weighted'])} | {fmt(t['f1_macro'])} | {fmt(t['auc_ovr_weighted'])} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `original`: baseline random stratified split with 5 features.")
    lines.append("- `method1_class_weight`: baseline + inverse-frequency class weighting multiplied with confidence.")
    lines.append("- `method4_6_threshold_features`: engineered features + validation-tuned class thresholds.")
    lines.append("- `method7_8_hparams_chrono`: tuned hyperparameters + chronological split (harder, more realistic).")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append("- If optimizing baseline weighted F1 under random split: choose the best `F1 Weighted` among random-split variants.")
    lines.append("- If optimizing minority sensitivity: compare `F1 Macro` and class-level confusion matrices externally.")
    lines.append("- If optimizing temporal robustness: prefer `method7_8_hparams_chrono` despite lower direct comparability.")
    lines.append("")
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    df = load_df()
    y = df["target"].to_numpy()
    conf = df["confidence"].to_numpy()

    base_X_df = base_features(df)
    eng_X_df = engineered_features(df)

    base_X = base_X_df.to_numpy()
    eng_X = eng_X_df.to_numpy()

    # original split/weights
    split_orig = SplitData(
        *random_split(base_X, y, conf),
        feature_cols=list(base_X_df.columns),
        split_type="random_stratified",
    )

    # method 1 weights
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    cls_w = {cls: n_samples / (n_classes * cnt) for cls, cnt in zip(classes, counts)}
    w_m1 = np.array([conf[i] * cls_w[y[i]] for i in range(n_samples)])
    split_m1 = SplitData(
        *random_split(base_X, y, w_m1),
        feature_cols=list(base_X_df.columns),
        split_type="random_stratified",
    )

    # method 4+6 split
    split_m46 = SplitData(
        *random_split(eng_X, y, conf),
        feature_cols=list(eng_X_df.columns),
        split_type="random_stratified",
    )

    # method 7+8 split
    split_m78 = SplitData(
        *chronological_split(df, base_X, y, conf),
        feature_cols=list(base_X_df.columns),
        split_type="chronological",
    )

    params_original = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
    }
    params_m1 = dict(params_original)
    params_m46 = dict(params_original)
    params_m78 = {
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
        "verbosity": 0,
    }

    print("Training original...")
    res_original = train_variant("original", split_orig, params_original)

    print("Training method1_class_weight...")
    res_m1 = train_variant("method1_class_weight", split_m1, params_m1)

    print("Training method4_6_threshold_features...")
    # train once to tune thresholds
    tmp_model = xgb.XGBClassifier(**params_m46)
    tmp_model.fit(split_m46.X_train, split_m46.y_train, sample_weight=split_m46.w_train, verbose=False)
    val_proba = tmp_model.predict_proba(split_m46.X_val)
    best_th = tune_thresholds(split_m46.y_val, val_proba, split_m46.w_val)
    # retrain clean and evaluate with tuned thresholds
    res_m46 = train_variant("method4_6_threshold_features", split_m46, params_m46, thresholds=best_th)

    print("Training method7_8_hparams_chrono...")
    res_m78 = train_variant("method7_8_hparams_chrono", split_m78, params_m78)

    results = {
        "original": res_original,
        "method1_class_weight": res_m1,
        "method4_6_threshold_features": res_m46,
        "method7_8_hparams_chrono": res_m78,
    }

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(BENCH_DIR / "trinary_variant_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    build_readme(results)
    print("Saved models, benchmarks JSON, and README_VARIANTS.md")


if __name__ == "__main__":
    main()
