"""
Trinary Volatility Classification with XGBoost
Trains XGBoost classifier to predict daily stock volatility (LOW=0, MEDIUM=1, HIGH=2)
Uses confidence scores as sample weights for weighted learning
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Dataset path
DATASET_PATH = Path(__file__).parent.parent / "xgboost_dataset_china_stock" / "trinary_xgboost_training_dataset.pkl"


def load_dataset(dataset_path=DATASET_PATH):
    """Load training dataset from pickle file."""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    print(f"Dataset loaded: {len(data)} samples, {len(data.columns)} columns")
    return data


def prepare_data(data, val_size=0.1, test_size=0.1):
    """
    Prepare features and labels, split into train/validation/test sets.
    """
    feature_cols = [
        "intraday_range",
        "volume_change_rate",
        "rolling_historical_volatility",
        "prev_day_news_count",
        "prev_day_avg_news_sentiment",
    ]

    X = data[feature_cols].values
    y = data["target"].values
    weights = data["confidence"].values

    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.0")

    X_train_val, X_test, y_train_val, y_test, weights_train_val, weights_test = train_test_split(
        X,
        y,
        weights,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
        X_train_val,
        y_train_val,
        weights_train_val,
        test_size=val_ratio,
        random_state=42,
        stratify=y_train_val,
    )

    print("\nData split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")

    print("\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(y_train) * 100
        print(f"  Class {label}: {count} ({pct:.2f}%)")

    print("\nFeature statistics:")
    print(f"  intraday_range: mean={X_train[:, 0].mean():.6f}, std={X_train[:, 0].std():.6f}")
    print(f"  volume_change_rate: mean={X_train[:, 1].mean():.6f}, std={X_train[:, 1].std():.6f}")
    print(f"  rolling_historical_volatility: mean={X_train[:, 2].mean():.6f}, std={X_train[:, 2].std():.6f}")
    print(f"  prev_day_news_count: mean={X_train[:, 3].mean():.6f}, std={X_train[:, 3].std():.6f}")
    print(f"  prev_day_avg_news_sentiment: mean={X_train[:, 4].mean():.6f}, std={X_train[:, 4].std():.6f}")

    print("\nSample weights (confidence):")
    print(f"  Training set - mean={weights_train.mean():.4f}, min={weights_train.min():.4f}, max={weights_train.max():.4f}")
    print(f"  Validation set - mean={weights_val.mean():.4f}, min={weights_val.min():.4f}, max={weights_val.max():.4f}")
    print(f"  Test set - mean={weights_test.mean():.4f}, min={weights_test.min():.4f}, max={weights_test.max():.4f}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        weights_train,
        weights_val,
        weights_test,
        feature_cols,
    )


def train_model(X_train, y_train, weights_train, **xgb_params):
    """Train XGBoost classifier with sample weights."""
    default_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 1,
    }
    default_params.update(xgb_params)

    print("\nTraining XGBoost model with parameters:")
    for k, v in default_params.items():
        print(f"  {k}: {v}")

    model = xgb.XGBClassifier(**default_params)
    print(f"\nTraining on {len(X_train)} samples...")
    model.fit(X_train, y_train, sample_weight=weights_train, verbose=True)
    return model


def evaluate_model(
    model,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    weights_train=None,
    weights_val=None,
    weights_test=None,
):
    """Evaluate model on train/val/test."""
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)
    y_test_proba = model.predict_proba(X_test)

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    results = {}

    print("\n--- TRAINING SET ---")
    train_acc = accuracy_score(y_train, y_train_pred, sample_weight=weights_train)
    train_prec = precision_score(y_train, y_train_pred, sample_weight=weights_train, average="weighted", zero_division=0)
    train_rec = recall_score(y_train, y_train_pred, sample_weight=weights_train, average="weighted", zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, sample_weight=weights_train, average="weighted", zero_division=0)
    train_auc = roc_auc_score(y_train, y_train_proba, sample_weight=weights_train, multi_class="ovr", average="weighted")
    print(f"Accuracy:  {train_acc:.4f}")
    print(f"Precision: {train_prec:.4f} (weighted)")
    print(f"Recall:    {train_rec:.4f} (weighted)")
    print(f"F1-Score:  {train_f1:.4f} (weighted)")
    print(f"ROC-AUC:   {train_auc:.4f} (weighted OvR)")
    results["train"] = {"accuracy": train_acc, "precision": train_prec, "recall": train_rec, "f1": train_f1, "auc": train_auc}

    print("\n--- VALIDATION SET ---")
    val_acc = accuracy_score(y_val, y_val_pred, sample_weight=weights_val)
    val_prec = precision_score(y_val, y_val_pred, sample_weight=weights_val, average="weighted", zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, sample_weight=weights_val, average="weighted", zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, sample_weight=weights_val, average="weighted", zero_division=0)
    val_auc = roc_auc_score(y_val, y_val_proba, sample_weight=weights_val, multi_class="ovr", average="weighted")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"Precision: {val_prec:.4f} (weighted)")
    print(f"Recall:    {val_rec:.4f} (weighted)")
    print(f"F1-Score:  {val_f1:.4f} (weighted)")
    print(f"ROC-AUC:   {val_auc:.4f} (weighted OvR)")
    results["validation"] = {"accuracy": val_acc, "precision": val_prec, "recall": val_rec, "f1": val_f1, "auc": val_auc}

    print("\n--- TEST SET ---")
    test_acc = accuracy_score(y_test, y_test_pred, sample_weight=weights_test)
    test_prec = precision_score(y_test, y_test_pred, sample_weight=weights_test, average="weighted", zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, sample_weight=weights_test, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, sample_weight=weights_test, average="weighted", zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_proba, sample_weight=weights_test, multi_class="ovr", average="weighted")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f} (weighted)")
    print(f"Recall:    {test_rec:.4f} (weighted)")
    print(f"F1-Score:  {test_f1:.4f} (weighted)")
    print(f"ROC-AUC:   {test_auc:.4f} (weighted OvR)")
    results["test"] = {"accuracy": test_acc, "precision": test_prec, "recall": test_rec, "f1": test_f1, "auc": test_auc}

    print("\n--- CONFUSION MATRIX ---")
    train_cm = confusion_matrix(y_train, y_train_pred)
    val_cm = confusion_matrix(y_val, y_val_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    labels = ["LOW (0)", "MEDIUM (1)", "HIGH (2)"]

    for split_name, cm in [("Training", train_cm), ("Validation", val_cm), ("Test", test_cm)]:
        print(f"\n{split_name} set:")
        for i, label in enumerate(labels):
            fp = int(sum(cm[:, i]) - cm[i, i])
            fn = int(sum(cm[i, :]) - cm[i, i])
            print(f"  {label}: TP={cm[i,i]}, FP={fp}, FN={fn}")

    results["confusion_matrices"] = {"train": train_cm, "validation": val_cm, "test": test_cm}
    results["predictions"] = {
        "train": y_train_pred,
        "val": y_val_pred,
        "test": y_test_pred,
        "train_proba": y_train_proba,
        "val_proba": y_val_proba,
        "test_proba": y_test_proba,
    }

    print("\n" + "=" * 70)
    return results


def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    importance = model.get_booster().get_score(importance_type="weight")
    if not importance:
        print("No feature importance available")
        return

    features = [feature_names[int(f.replace("f", ""))] if f.startswith("f") else f for f in importance.keys()]
    scores = list(importance.values())
    plt.figure(figsize=(10, 6))
    plt.barh(features, scores)
    plt.xlabel("Importance Score")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    print("Saved: feature_importance.png")
    plt.show()


def plot_roc_curves(y_train, y_val, y_test, y_train_proba, y_val_proba, y_test_proba):
    """Plot one-vs-rest ROC curves for trinary classification."""
    class_names = ["LOW (0)", "MEDIUM (1)", "HIGH (2)"]
    class_ids = [0, 1, 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    split_data = [
        ("Training", y_train, y_train_proba),
        ("Validation", y_val, y_val_proba),
        ("Test", y_test, y_test_proba),
    ]
    for ax, (split_name, y_true, y_proba) in zip(axes, split_data):
        for class_id, class_name in zip(class_ids, class_names):
            y_true_binary = (y_true == class_id).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, class_id])
            class_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={class_auc:.3f})")

        weighted_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_title(f"{split_name} (Weighted OvR AUC={weighted_auc:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    plt.suptitle("ROC Curves - Trinary Volatility Classification")
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=300)
    print("Saved: roc_curves.png")
    plt.show()


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot confusion matrix for trinary classification."""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, cmap=plt.cm.Blues)
    classes = ["LOW (0)", "MEDIUM (1)", "HIGH (2)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    return plt


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("XGBoost Trinary Volatility Classification")
    print("=" * 70)

    data = load_dataset()
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        weights_train,
        weights_val,
        weights_test,
        feature_names,
    ) = prepare_data(data, val_size=0.1, test_size=0.1)

    model = train_model(
        X_train,
        y_train,
        weights_train,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    results = evaluate_model(
        model,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        weights_train,
        weights_val,
        weights_test,
    )

    print("\nFeature Importances:")
    for fname, importance in zip(feature_names, model.feature_importances_):
        print(f"  {fname}: {importance:.4f}")

    print("\nGenerating plots...")
    plot_feature_importance(model, feature_names=feature_names)
    plot_roc_curves(
        y_train,
        y_val,
        y_test,
        results["predictions"]["train_proba"],
        results["predictions"]["val_proba"],
        results["predictions"]["test_proba"],
    )

    fig_train = plot_confusion_matrix(results["confusion_matrices"]["train"], "Confusion Matrix - Training Set")
    fig_train.savefig("confusion_matrix_train.png", dpi=300)
    print("Saved: confusion_matrix_train.png")
    fig_train.show()

    fig_val = plot_confusion_matrix(results["confusion_matrices"]["validation"], "Confusion Matrix - Validation Set")
    fig_val.savefig("confusion_matrix_validation.png", dpi=300)
    print("Saved: confusion_matrix_validation.png")
    fig_val.show()

    fig_test = plot_confusion_matrix(results["confusion_matrices"]["test"], "Confusion Matrix - Test Set")
    fig_test.savefig("confusion_matrix_test.png", dpi=300)
    print("Saved: confusion_matrix_test.png")
    fig_test.show()

    model_path = Path(__file__).parent / "volatility_classifier_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return model, results


if __name__ == "__main__":
    model, results = main()
