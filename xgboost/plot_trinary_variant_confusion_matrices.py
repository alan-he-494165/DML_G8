"""
Plot test-set confusion matrices for all saved trinary model variants.
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from TRINARY_train_variants import (
    BENCH_DIR,
    MODELS_DIR,
    SplitData,
    base_features,
    chronological_split,
    engineered_features,
    load_df,
    predict_with_thresholds,
    random_split,
)


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "result"
BENCH_PATH = BENCH_DIR / "TRINARY_benchmark_variants.json"
LABELS = ["LOW", "MEDIUM", "HIGH"]


def random_split_indices(y: np.ndarray, val_size=0.1, test_size=0.1):
    indices = np.arange(len(y))
    idx_train_val, idx_test, y_train_val, _ = train_test_split(
        indices, y, test_size=test_size, random_state=42, stratify=y
    )
    val_ratio = val_size / (1.0 - test_size)
    idx_train, idx_val, _, _ = train_test_split(
        idx_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )
    return idx_train, idx_val, idx_test


def chronological_split_indices(df: pd.DataFrame, val_size=0.1, test_size=0.1):
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
    indices = np.arange(len(df))
    train_idx = indices[np.array([d in train_dates for d in dates])]
    val_idx = indices[np.array([d in val_dates for d in dates])]
    test_idx = indices[np.array([d in test_dates for d in dates])]
    return train_idx, val_idx, test_idx


def build_splits(df) -> dict[str, SplitData]:
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
    w_m1 = np.array([conf[i] * cls_w[y[i]] for i in range(n_samples)])

    return {
        "original": SplitData(
            *random_split(base_X, y, conf),
            feature_cols=list(base_X_df.columns),
            split_type="random_stratified",
        ),
        "method1_class_weight": SplitData(
            *random_split(base_X, y, w_m1),
            feature_cols=list(base_X_df.columns),
            split_type="random_stratified",
        ),
        "method4_6_threshold_features": SplitData(
            *random_split(eng_X, y, conf),
            feature_cols=list(eng_X_df.columns),
            split_type="random_stratified",
        ),
        "method1_4_6_class_weight_threshold": SplitData(
            *random_split(base_X, y, w_m1),
            feature_cols=list(base_X_df.columns),
            split_type="random_stratified",
        ),
        "method6_chrono_threshold_original_hparams": SplitData(
            *chronological_split(df, base_X, y, conf),
            feature_cols=list(base_X_df.columns),
            split_type="chronological",
        ),
        "method7_8_hparams_chrono": SplitData(
            *chronological_split(df, base_X, y, conf),
            feature_cols=list(base_X_df.columns),
            split_type="chronological",
        ),
        "method6_7_8_threshold_hparams_chrono": SplitData(
            *chronological_split(df, base_X, y, conf),
            feature_cols=list(base_X_df.columns),
            split_type="chronological",
        ),
    }


def build_persistence_predictions(df: pd.DataFrame) -> np.ndarray:
    tmp = df[["symbol", "date", "target"]].copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp = tmp.sort_values(["symbol", "date"]).copy()
    tmp["persistence_pred"] = tmp.groupby("symbol")["target"].shift(1)
    tmp["persistence_pred"] = tmp["persistence_pred"].fillna(1).astype(int)
    out = pd.Series(tmp["persistence_pred"].to_numpy(), index=tmp.index)
    return out.sort_index().to_numpy()


def load_model(name: str):
    path = MODELS_DIR / f"TRINARY_{name}.pkl"
    if not path.exists():
        path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def get_predictions(name: str, model, split: SplitData, benchmarks: dict) -> np.ndarray:
    proba = model.predict_proba(split.X_test)
    thresholds = benchmarks[name].get("thresholds")
    if thresholds:
        return predict_with_thresholds(proba, thresholds["t0"], thresholds["t2"])
    return np.argmax(proba, axis=1)


def plot_cm(ax, cm: np.ndarray, title: str):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
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
                fontsize=10,
            )
    return im


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()
    splits = build_splits(df)
    persistence_pred = build_persistence_predictions(df)
    y_all = df["target"].to_numpy()
    _, _, random_test_idx = random_split_indices(y_all)
    _, _, chrono_test_idx = chronological_split_indices(df)

    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    order = [
        "original",
        "method1_class_weight",
        "method4_6_threshold_features",
        "method1_4_6_class_weight_threshold",
        "method6_chrono_threshold_original_hparams",
        "method7_8_hparams_chrono",
        "method6_7_8_threshold_hparams_chrono",
        "persistence_random_stratified",
        "persistence_chronological",
    ]

    fig, axes = plt.subplots(5, 2, figsize=(14, 27))
    axes = axes.flatten()
    last_im = None

    for ax, name in zip(axes, order):
        if name.startswith("persistence_"):
            if name.endswith("random_stratified"):
                y_true = y_all[random_test_idx]
                y_pred = persistence_pred[random_test_idx]
                split_type = "random_stratified"
            else:
                y_true = y_all[chrono_test_idx]
                y_pred = persistence_pred[chrono_test_idx]
                split_type = "chronological"
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            acc = accuracy_score(y_true, y_pred)
            f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
            title = (
                f"{name}\n"
                f"{split_type} | "
                f"acc={acc:.3f}, "
                f"F1w={f1w:.3f}, "
                f"F1m={f1m:.3f}"
            )
        else:
            model = load_model(name)
            split = splits[name]
            y_pred = get_predictions(name, model, split, benchmarks)
            cm = confusion_matrix(split.y_test, y_pred, labels=[0, 1, 2])
            test_metrics = benchmarks[name]["test"]
            title = (
                f"{name}\n"
                f"{benchmarks[name]['split_type']} | "
                f"acc={test_metrics['accuracy']:.3f}, "
                f"F1w={test_metrics['f1_weighted']:.3f}, "
                f"F1m={test_metrics['f1_macro']:.3f}"
            )
        last_im = plot_cm(ax, cm, title)

        per_variant_path = OUT_DIR / f"confusion_matrix_{name}.png"
        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        plot_cm(ax_single, cm, title)
        fig_single.tight_layout()
        fig_single.savefig(per_variant_path, dpi=300, bbox_inches="tight")
        plt.close(fig_single)

    # Persistence baseline aligned to the 4 main model variants.
    persistence_aliases = {
        "original": ("random_stratified", random_test_idx),
        "method1_class_weight": ("random_stratified", random_test_idx),
        "method4_6_threshold_features": ("random_stratified", random_test_idx),
        "method7_8_hparams_chrono": ("chronological", chrono_test_idx),
    }
    for model_name, (split_type, idx) in persistence_aliases.items():
        y_true = y_all[idx]
        y_pred = persistence_pred[idx]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        title = (
            f"persistence for {model_name}\n"
            f"{split_type} | "
            f"acc={acc:.3f}, "
            f"F1w={f1w:.3f}, "
            f"F1m={f1m:.3f}"
        )
        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        plot_cm(ax_single, cm, title)
        fig_single.tight_layout()
        out = OUT_DIR / f"confusion_matrix_persistence_for_{model_name}.png"
        fig_single.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig_single)

    for ax in axes[len(order):]:
        ax.axis("off")

    fig.colorbar(last_im, ax=axes, shrink=0.75)
    fig.suptitle("Trinary Variant Confusion Matrices (Test Set)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    combined_path = OUT_DIR / "confusion_matrix_trinary_variants.png"
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {combined_path}")
    for name in order:
        print(f"Saved: {OUT_DIR / f'confusion_matrix_{name}.png'}")
    for model_name in persistence_aliases:
        print(f"Saved: {OUT_DIR / f'confusion_matrix_persistence_for_{model_name}.png'}")


if __name__ == "__main__":
    main()
