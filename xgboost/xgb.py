"""
Multi-class Volatility Classification with XGBoost
Trains XGBoost classifier to predict daily stock volatility as LOW / MEDIUM / HIGH
Uses confidence scores as sample weights for weighted learning
"""

import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from pathlib import Path

CLASS_NAMES = ['Low', 'Medium', 'High']
N_CLASSES = 3

# Dataset path
DATASET_PATH = Path(__file__).parent.parent / 'data_for_process' / 'xgboost_dataset_china_stock' / 'xgboost_training_dataset.pkl'


def load_dataset(dataset_path=DATASET_PATH):
    """Load training dataset from pickle file."""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Dataset loaded: {len(data)} samples, {len(data.columns)} columns")
    return data


def prepare_data(data, val_size=0.1, test_size=0.1):
    """
    Prepare features and labels, split into train/validation sets.
    
    Args:
        data: DataFrame with all features and labels
        val_size: fraction for validation set (default 10%)
        test_size: fraction for test set (default 10%)
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, weights_train, weights_val, weights_test
    """
    # Feature columns (5 inputs)
    feature_cols = [
        'intraday_range',
        'volume_change_rate',
        'rolling_historical_volatility',
        'prev_day_news_count',
        'prev_day_avg_news_sentiment'
    ]
    
    X = data[feature_cols].values
    y = data['target'].values
    weights = data['confidence'].values
    
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.0")

    # Split out test set first
    X_train_val, X_test, y_train_val, y_test, weights_train_val, weights_test = train_test_split(
        X, y, weights,
        test_size=test_size,
        random_state=42,
        stratify=y  # Maintain class balance
    )

    # Split remaining into train/validation
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
        X_train_val, y_train_val, weights_train_val,
        test_size=val_ratio,
        random_state=42,
        stratify=y_train_val
    )
    
    print(f"\nData split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Data distribution
    print(f"\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(y_train) * 100
        print(f"  Class {label}: {count} ({pct:.2f}%)")
    
    print(f"\nFeature statistics:")
    print(f"  intraday_range: mean={X_train[:, 0].mean():.6f}, std={X_train[:, 0].std():.6f}")
    print(f"  volume_change_rate: mean={X_train[:, 1].mean():.6f}, std={X_train[:, 1].std():.6f}")
    print(f"  rolling_historical_volatility: mean={X_train[:, 2].mean():.6f}, std={X_train[:, 2].std():.6f}")
    print(f"  prev_day_news_count: mean={X_train[:, 3].mean():.6f}, std={X_train[:, 3].std():.6f}")
    print(f"  prev_day_avg_news_sentiment: mean={X_train[:, 4].mean():.6f}, std={X_train[:, 4].std():.6f}")
    
    print(f"\nSample weights (confidence):")
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
    )


def compute_combined_weights(y: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """
    Multiply per-class balance weights by confidence scores.

    Class balance weights correct for label imbalance across LOW/MEDIUM/HIGH.
    Confidence scores down-weight borderline samples near threshold boundaries.
    The product is normalised to mean=1 so the XGBoost loss scale stays stable.
    """
    class_w = compute_sample_weight('balanced', y)
    combined = class_w * confidence
    return combined / combined.mean()


def tune_hyperparameters(
    X_train, y_train, weights_train,
    X_val, y_val,
    n_trials: int = 100,
) -> dict:
    """
    Bayesian hyperparameter search with Optuna.
    Optimises macro F1 on the (unweighted) validation set.

    Args:
        X_train, y_train, weights_train: Training data and combined sample weights
        X_val, y_val: Validation data used as the objective signal
        n_trials: Number of Optuna trials (default 100)

    Returns:
        dict of best hyperparameters ready to pass to XGBClassifier
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'objective':        'multi:softprob',
            'num_class':        N_CLASSES,
            'random_state':     42,
            'verbosity':        0,
            'max_depth':        trial.suggest_int('max_depth', 3, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators':     trial.suggest_int('n_estimators', 100, 1000),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma':            trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 3.0),
        }
        model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
        model.fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_val_pred = model.predict(X_val)
        return f1_score(y_val, y_val_pred, average='macro', zero_division=0)

    print(f"\nRunning Optuna hyperparameter search ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest validation macro F1: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params


def train_model(X_train, y_train, weights_train, X_val=None, y_val=None, **xgb_params):
    """
    Train XGBoost classifier with sample weights.

    Args:
        X_train: Training features
        y_train: Training labels
        weights_train: Combined sample weights (class balance × confidence)
        X_val, y_val: Optional validation data for early stopping
        **xgb_params: XGBoost parameters (e.g. from tune_hyperparameters)

    Returns:
        Trained model
    """
    # Default parameters (overridden by xgb_params)
    default_params = {
        'objective': 'multi:softprob',
        'num_class': N_CLASSES,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 1,
    }
    default_params.update(xgb_params)

    print(f"\nTraining XGBoost model with parameters:")
    for k, v in default_params.items():
        print(f"  {k}: {v}")

    if X_val is not None and y_val is not None:
        default_params['early_stopping_rounds'] = 30

    model = xgb.XGBClassifier(**default_params)

    print(f"\nTraining on {len(X_train)} samples...")
    fit_kwargs = {'sample_weight': weights_train, 'verbose': True}
    if X_val is not None and y_val is not None:
        fit_kwargs['eval_set'] = [(X_val, y_val)]

    model.fit(X_train, y_train, **fit_kwargs)

    if X_val is not None:
        print(f"Best iteration (early stopping): {model.best_iteration}")

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
    """
    Evaluate model on training and validation sets.
    
    Args:
        model: Trained XGBoost model
        X_train, X_val, X_test: Training, validation, and test features
        y_train, y_val, y_test: Training, validation, and test labels
        weights_train, weights_val, weights_test: Optional sample weights for weighted metrics
        
    Returns:
        Dictionary with all evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Full probability matrix (n_samples × 3) for multi-class ROC
    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)
    y_test_proba = model.predict_proba(X_test)

    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    results = {}

    def _print_metrics(split_name, y_true, y_pred, y_proba, weights):
        acc  = accuracy_score(y_true, y_pred, sample_weight=weights)
        prec = precision_score(y_true, y_pred, average='macro', sample_weight=weights, zero_division=0)
        rec  = recall_score(y_true, y_pred, average='macro', sample_weight=weights, zero_division=0)
        f1   = f1_score(y_true, y_pred, average='macro', sample_weight=weights, zero_division=0)
        auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro', sample_weight=weights)
        print(f"\n--- {split_name} ---")
        print(f"Accuracy:        {acc:.4f}")
        print(f"Precision (mac): {prec:.4f}")
        print(f"Recall    (mac): {rec:.4f}")
        print(f"F1-Score  (mac): {f1:.4f}")
        print(f"ROC-AUC   (ovr): {auc_score:.4f}")
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_score}

    results['train']      = _print_metrics('TRAINING SET',   y_train, y_train_pred, y_train_proba, weights_train)
    results['validation'] = _print_metrics('VALIDATION SET', y_val,   y_val_pred,   y_val_proba,   weights_val)
    results['test']       = _print_metrics('TEST SET',       y_test,  y_test_pred,  y_test_proba,  weights_test)

    # Confusion matrices
    print(f"\n--- CONFUSION MATRICES (rows=True, cols=Predicted) ---")
    train_cm = confusion_matrix(y_train, y_train_pred)
    val_cm   = confusion_matrix(y_val,   y_val_pred)
    test_cm  = confusion_matrix(y_test,  y_test_pred)

    for split_name, cm in [('Training', train_cm), ('Validation', val_cm), ('Test', test_cm)]:
        print(f"\n{split_name} set ({' / '.join(CLASS_NAMES)}):")
        print(cm)

    results['confusion_matrices'] = {'train': train_cm, 'validation': val_cm, 'test': test_cm}
    results['predictions'] = {
        'train': y_train_pred,
        'val': y_val_pred,
        'test': y_test_pred,
        'train_proba': y_train_proba,
        'val_proba': y_val_proba,
        'test_proba': y_test_proba,
    }

    print("\n" + "="*70)

    return results


def plot_feature_importance(model, feature_names=['intraday_range', 'volume_change_rate', 'rolling_historical_volatility', 'prev_day_news_count', 'prev_day_avg_news_sentiment']):
    """Plot feature importance."""
    importance = model.get_booster().get_score(importance_type='weight')
    
    if not importance:
        print("No feature importance available")
        return
    
    features = [feature_names[int(f.replace('f', ''))] if f.startswith('f') else f for f in importance.keys()]
    scores = list(importance.values())
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, scores)
    plt.xlabel('Importance Score')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("Saved: feature_importance.png")
    plt.show()


def plot_roc_curves(y_train, y_val, y_test, y_train_proba, y_val_proba, y_test_proba):
    """Plot One-vs-Rest ROC curves for all 3 classes across train / val / test splits."""
    colors = ['steelblue', 'darkorange', 'crimson']
    splits = [
        ('Training',   y_train, y_train_proba),
        ('Validation', y_val,   y_val_proba),
        ('Test',       y_test,  y_test_proba),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (split_name, y_true, y_proba) in zip(axes, splits):
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_proba[:, cls_idx])
            auc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{cls_name} (AUC={auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves — {split_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('One-vs-Rest ROC Curves — Multi-class Volatility Classification')
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    print("Saved: roc_curves.png")
    plt.show()


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """Plot 3-class confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap=plt.cm.Blues)

    tick_marks = np.arange(N_CLASSES)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    thresh = cm.max() / 2
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return plt


def main():
    """Main training pipeline."""
    print("="*70)
    print("XGBoost Multi-class Volatility Classification (LOW / MEDIUM / HIGH)")
    print("="*70)
    
    # Load data
    data = load_dataset()
    
    # Prepare data (80% train, 10% validation, 10% test)
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
    ) = prepare_data(data, val_size=0.1, test_size=0.1)

    # Combine class balance weights with confidence scores for training
    # (val/test weights stay as confidence-only for clean evaluation reporting)
    weights_train_combined = compute_combined_weights(y_train, weights_train)
    print(f"\nCombined training weights: mean={weights_train_combined.mean():.4f}, "
          f"min={weights_train_combined.min():.4f}, max={weights_train_combined.max():.4f}")

    # Hyperparameter tuning on validation set
    best_params = tune_hyperparameters(
        X_train, y_train, weights_train_combined,
        X_val, y_val,
        n_trials=100,
    )

    # Train final model with best params and early stopping
    model = train_model(
        X_train, y_train, weights_train_combined,
        X_val=X_val, y_val=y_val,
        **best_params,
    )

    # Evaluate model
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
    
    # Feature importance
    print("\nFeature Importances:")
    feature_names = [
        'intraday_range',
        'volume_change_rate',
        'rolling_historical_volatility',
        'prev_day_news_count',
        'prev_day_avg_news_sentiment'
    ]
    for fname, importance in zip(feature_names, model.feature_importances_):
        print(f"  {fname}: {importance:.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    
    # Feature importance
    plot_feature_importance(model)
    
    # ROC curves (One-vs-Rest, one subplot per split)
    plot_roc_curves(
        y_train,
        y_val,
        y_test,
        results['predictions']['train_proba'],
        results['predictions']['val_proba'],
        results['predictions']['test_proba'],
    )
    
    # Confusion matrices
    fig_train = plot_confusion_matrix(results['confusion_matrices']['train'], 'Confusion Matrix - Training Set')
    fig_train.savefig('confusion_matrix_train.png', dpi=300)
    print("Saved: confusion_matrix_train.png")
    fig_train.show()
    
    fig_val = plot_confusion_matrix(results['confusion_matrices']['validation'], 'Confusion Matrix - Validation Set')
    fig_val.savefig('confusion_matrix_validation.png', dpi=300)
    print("Saved: confusion_matrix_validation.png")
    fig_val.show()

    fig_test = plot_confusion_matrix(results['confusion_matrices']['test'], 'Confusion Matrix - Test Set')
    fig_test.savefig('confusion_matrix_test.png', dpi=300)
    print("Saved: confusion_matrix_test.png")
    fig_test.show()
    
    # Save model
    model_path = Path(__file__).parent / 'volatility_classifier_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return model, results


if __name__ == '__main__':
    model, results = main()
