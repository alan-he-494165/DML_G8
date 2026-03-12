"""
Train chronological trinary XGBoost variants without any news input features.

All artifacts are written under xgboost_trinary_chrono_no_news/.
This does not modify existing benchmark folders.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "xgboost_dataset_china_stock" / "trinary_xgboost_training_dataset.pkl"
OUT_DIR = ROOT / "xgboost_trinary_chrono_no_news"
MODELS_DIR = OUT_DIR / "models"
PLOTS_DIR = OUT_DIR / "plots"
BENCH_PATH = OUT_DIR / "chrono_no_news_benchmarks.json"


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


def load_df() -> pd.DataFrame:
    with open(DATASET_PATH, "rb") as f:
        return pickle.load(f)


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "intraday_range",
            "volume_change_rate",
            "rolling_historical_volatility",
        ]
    ].copy()


def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = base_features(df)
    eps = 1e-8
    out["range_vol_ratio"] = out["intraday_range"] / (out["rolling_historical_volatility"] + eps)
    out["volume_change_abs"] = np.abs(out["volume_change_rate"])
    return out


def chronological_split(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, w: np.ndarray, val_size=0.1, test_size=0.1) -> SplitData:
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

    return SplitData(
        X[train_mask],
        X[val_mask],
        X[test_mask],
        y[train_mask],
        y[val_mask],
        y[test_mask],
        w[train_mask],
        w[val_mask],
        w[test_mask],
        [],
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
    else:
        y_train_pred = predict_with_thresholds(p_train, thresholds["t0"], thresholds["t2"])
        y_val_pred = predict_with_thresholds(p_val, thresholds["t0"], thresholds["t2"])
        y_test_pred = predict_with_thresholds(p_test, thresholds["t0"], thresholds["t2"])

    with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    return {
        "train": eval_split(split.y_train, y_train_pred, p_train, split.w_train),
        "validation": eval_split(split.y_val, y_val_pred, p_val, split.w_val),
        "test": eval_split(split.y_test, y_test_pred, p_test, split.w_test),
        "thresholds": thresholds,
        "feature_count": len(split.feature_cols),
        "params": params,
        "confusion_matrix_test": confusion_matrix(split.y_test, y_test_pred, labels=[0, 1, 2]).tolist(),
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path):
    labels = ["LOW", "MEDIUM", "HIGH"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_benchmark_plot(results: dict, order: list[str]):
    labels = [name.replace("_chrono", "").replace("_", "\n") for name in order]
    metric_keys = ["accuracy", "f1_weighted", "f1_macro", "auc_ovr_weighted"]
    metric_labels = ["Accuracy", "Weighted F1", "Macro F1", "Weighted OvR AUC"]
    x = np.arange(len(order))
    width = 0.18

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (metric_key, metric_label) in enumerate(zip(metric_keys, metric_labels)):
        vals = [results[name]["test"][metric_key] for name in order]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=metric_label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_title("Chronological Trinary Variant Comparison Without News Inputs")
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.45, 0.85)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "chrono_no_news_benchmarks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_df()
    y = df["target"].to_numpy()
    conf = df["confidence"].to_numpy()

    base_X_df = base_features(df)
    eng_X_df = engineered_features(df)
    base_X = base_X_df.to_numpy()
    eng_X = eng_X_df.to_numpy()

    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    cls_w = {cls: n_samples / (n_classes * cnt) for cls, cnt in zip(classes, counts)}
    w_class = np.array([conf[i] * cls_w[y[i]] for i in range(n_samples)])

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
    params_tuned = {
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

    variants = [
        ("original_chrono_no_news", base_X_df, base_X, conf, params_original, False),
        ("class_weight_chrono_no_news", base_X_df, base_X, w_class, params_original, False),
        ("threshold_features_chrono_no_news", eng_X_df, eng_X, conf, params_original, True),
        ("class_weight_threshold_chrono_no_news", base_X_df, base_X, w_class, params_original, True),
        ("tuned_hparams_chrono_no_news", base_X_df, base_X, conf, params_tuned, False),
        ("threshold_tuned_hparams_chrono_no_news", base_X_df, base_X, conf, params_tuned, True),
    ]

    results = {}
    order = []

    for name, feature_df, X, weights, params, use_thresholds in variants:
        print(f"Training {name}...")
        split = chronological_split(df, X, y, weights)
        split.feature_cols = list(feature_df.columns)
        thresholds = None
        if use_thresholds:
            tmp_model = xgb.XGBClassifier(**params)
            tmp_model.fit(split.X_train, split.y_train, sample_weight=split.w_train, verbose=False)
            val_proba = tmp_model.predict_proba(split.X_val)
            thresholds = tune_thresholds(split.y_val, val_proba, split.w_val)
        results[name] = train_variant(name, split, params, thresholds=thresholds)
        order.append(name)
        plot_confusion_matrix(
            np.array(results[name]["confusion_matrix_test"]),
            f"{name} (chronological test)",
            PLOTS_DIR / f"confusion_matrix_{name}.png",
        )

    with open(BENCH_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_benchmark_plot(results, order)
    print(f"Saved benchmarks to: {BENCH_PATH}")
    print(f"Saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
