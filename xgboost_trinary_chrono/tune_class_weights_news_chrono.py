"""
Tune class-weight multipliers for the chronological trinary model with news features.

Search space:
- class 1 multiplier fixed at 1.0
- class 0 and class 2 multipliers tuned on a coarse grid

Objective:
- maximize validation macro F1
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "xgboost_dataset_china_stock" / "trinary_xgboost_training_dataset.pkl"
OUT_DIR = ROOT / "xgboost_trinary_chrono" / "class_weight_tuning"
MODEL_PATH = OUT_DIR / "best_class_weight_chrono_model.pkl"
JSON_PATH = OUT_DIR / "class_weight_search_results.json"
PNG_PATH = OUT_DIR / "class_weight_macro_f1_heatmap.png"

FEATURE_COLS = [
    "intraday_range",
    "volume_change_rate",
    "rolling_historical_volatility",
    "prev_day_news_count",
    "prev_day_avg_news_sentiment",
]


def load_df() -> pd.DataFrame:
    with open(DATASET_PATH, "rb") as f:
        return pickle.load(f)


def chronological_split(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, w: np.ndarray, val_size=0.1, test_size=0.1):
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


def eval_split(y_true, y_pred, y_proba, sw):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sw)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sw)),
        "auc_ovr_weighted": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", sample_weight=sw)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()
    X = df[FEATURE_COLS].to_numpy()
    y = df["target"].to_numpy()
    conf = df["confidence"].to_numpy()

    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    base_class_weights = {cls: n_samples / (n_classes * cnt) for cls, cnt in zip(classes, counts)}

    X_train, X_val, X_test, y_train, y_val, y_test, w_train_base, w_val_base, w_test_base = chronological_split(df, X, y, conf)

    params = {
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

    grid = [0.5, 1.0, 1.5, 2.0]
    heatmap = np.zeros((len(grid), len(grid)))
    trials = []
    best = None
    best_model = None

    for i, m0 in enumerate(grid):
        for j, m2 in enumerate(grid):
            class_mult = {0: m0, 1: 1.0, 2: m2}
            w_train = np.array([w_train_base[k] * base_class_weights[y_train[k]] * class_mult[y_train[k]] for k in range(len(y_train))])
            w_val = np.array([w_val_base[k] * base_class_weights[y_val[k]] * class_mult[y_val[k]] for k in range(len(y_val))])
            w_test = np.array([w_test_base[k] * base_class_weights[y_test[k]] * class_mult[y_test[k]] for k in range(len(y_test))])

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=w_train, verbose=False)

            p_val = model.predict_proba(X_val)
            y_val_pred = np.argmax(p_val, axis=1)
            val_metrics = eval_split(y_val, y_val_pred, p_val, w_val)
            heatmap[i, j] = val_metrics["f1_macro"]

            p_test = model.predict_proba(X_test)
            y_test_pred = np.argmax(p_test, axis=1)
            test_metrics = eval_split(y_test, y_test_pred, p_test, w_test)

            trial = {
                "class0_multiplier": m0,
                "class1_multiplier": 1.0,
                "class2_multiplier": m2,
                "validation": val_metrics,
                "test": test_metrics,
            }
            trials.append(trial)

            if best is None or val_metrics["f1_macro"] > best["validation"]["f1_macro"]:
                best = trial
                best_model = model

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    payload = {
        "feature_cols": FEATURE_COLS,
        "split_type": "chronological",
        "base_class_weights": {str(int(k)): float(v) for k, v in base_class_weights.items()},
        "search_grid": {"class0_multiplier": grid, "class1_multiplier": [1.0], "class2_multiplier": grid},
        "best": best,
        "trials": trials,
        "params": params,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(heatmap, cmap="viridis")
    ax.set_title("Validation Macro F1 by Class-Weight Multipliers")
    ax.set_xlabel("Class 2 Multiplier")
    ax.set_ylabel("Class 0 Multiplier")
    ax.set_xticks(range(len(grid)), [str(g) for g in grid])
    ax.set_yticks(range(len(grid)), [str(g) for g in grid])
    for i in range(len(grid)):
        for j in range(len(grid)):
            ax.text(j, i, f"{heatmap[i, j]:.3f}", ha="center", va="center", color="white", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved best model: {MODEL_PATH}")
    print(f"Saved search results: {JSON_PATH}")
    print(f"Saved heatmap: {PNG_PATH}")
    print("Best setting:", best)


if __name__ == "__main__":
    main()
