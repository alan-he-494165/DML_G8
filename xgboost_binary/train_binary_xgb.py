"""
Train a binary XGBoost volatility model in a separate folder and plot:
- model confusion matrix
- persistence baseline confusion matrix
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "xgboost_dataset" / "xgboost_training_dataset.pkl"
OUT_DIR = ROOT / "xgboost_binary"
MODEL_PATH = OUT_DIR / "binary_volatility_classifier_model.pkl"
METRICS_PATH = OUT_DIR / "binary_metrics.json"
MODEL_CM_PATH = OUT_DIR / "binary_confusion_matrix_model.png"
PERSIST_CM_PATH = OUT_DIR / "binary_confusion_matrix_persistence.png"

FEATURE_COLS = [
    "intraday_range",
    "volume_change_rate",
    "rolling_historical_volatility",
]
LABELS = ["LOW (0)", "HIGH (1)"]


def load_df() -> pd.DataFrame:
    with open(DATASET_PATH, "rb") as f:
        return pickle.load(f)


def chronological_split(df: pd.DataFrame, val_size=0.1, test_size=0.1):
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

    train_mask = pd.to_datetime(df["date"]).isin(train_dates).to_numpy()
    val_mask = pd.to_datetime(df["date"]).isin(val_dates).to_numpy()
    test_mask = pd.to_datetime(df["date"]).isin(test_dates).to_numpy()
    return train_mask, val_mask, test_mask


def build_persistence_predictions(df: pd.DataFrame, default_label: int) -> np.ndarray:
    tmp = df[["symbol", "date", "target"]].copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp = tmp.sort_values(["symbol", "date"]).copy()
    tmp["persistence_pred"] = tmp.groupby("symbol")["target"].shift(1)
    tmp["persistence_pred"] = tmp["persistence_pred"].fillna(default_label).astype(int)
    out = pd.Series(tmp["persistence_pred"].to_numpy(), index=tmp.index)
    return out.sort_index().to_numpy()


def eval_binary(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_confusion(cm: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(2), LABELS)
    ax.set_yticks(range(2), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()

    X = df[FEATURE_COLS].to_numpy()
    y = df["target"].to_numpy()
    w = df["confidence"].to_numpy()

    train_mask, val_mask, test_mask = chronological_split(df)

    X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], w[val_mask]
    X_test, y_test, w_test = X[test_mask], y[test_mask], w[test_mask]

    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
        "eval_metric": "logloss",
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=w_train, verbose=False)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    model_metrics = eval_binary(y_test, y_test_pred, y_test_proba)

    default_label = int(pd.Series(y_train).mode().iloc[0])
    persistence_all = build_persistence_predictions(df, default_label=default_label)
    y_persist = persistence_all[test_mask]
    persist_metrics = {
        "accuracy": float(accuracy_score(y_test, y_persist)),
        "precision": float(precision_score(y_test, y_persist, zero_division=0)),
        "recall": float(recall_score(y_test, y_persist, zero_division=0)),
        "f1": float(f1_score(y_test, y_persist, zero_division=0)),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "dataset_path": str(DATASET_PATH),
        "features": FEATURE_COLS,
        "split_type": "chronological",
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "params": params,
        "model_test": model_metrics,
        "persistence_test": persist_metrics,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, indent=2)

    model_cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    persist_cm = confusion_matrix(y_test, y_persist, labels=[0, 1])
    plot_confusion(model_cm, "Binary XGBoost Confusion Matrix (Chronological Test)", MODEL_CM_PATH)
    plot_confusion(persist_cm, "Binary Persistence Confusion Matrix (Chronological Test)", PERSIST_CM_PATH)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved model confusion matrix: {MODEL_CM_PATH}")
    print(f"Saved persistence confusion matrix: {PERSIST_CM_PATH}")
    print("Model test metrics:", model_metrics)
    print("Persistence test metrics:", persist_metrics)


if __name__ == "__main__":
    main()
