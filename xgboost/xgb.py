"""
Binary Volatility Classification with XGBoost
Trains XGBoost classifier to predict daily stock volatility
Uses confidence scores as sample weights for weighted learning
"""

import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
from pathlib import Path

# Dataset path
DATASET_PATH = Path(__file__).parent.parent / 'data_for_process' / 'xgboost_dataset' / 'xgboost_training_dataset.pkl'


def load_dataset(dataset_path=DATASET_PATH):
    """Load training dataset from pickle file."""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Dataset loaded: {len(data)} samples, {len(data.columns)} columns")
    return data


def prepare_data(data, test_size=0.1):
    """
    Prepare features and labels, split into train/validation sets.
    
    Args:
        data: DataFrame with all features and labels
        test_size: fraction for validation set (default 10%)
        
    Returns:
        X_train, X_val, y_train, y_val, weights_train, weights_val
    """
    # Feature columns
    feature_cols = ['intraday_range', 'volume_change_rate', 'rolling_historical_volatility']
    
    X = data[feature_cols].values
    y = data['target'].values
    weights = data['confidence'].values
    
    # Split 90% train, 10% validation
    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
        X, y, weights,
        test_size=test_size,
        random_state=42,
        stratify=y  # Maintain class balance
    )
    
    print(f"\nData split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    
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
    
    print(f"\nSample weights (confidence):")
    print(f"  Training set - mean={weights_train.mean():.4f}, min={weights_train.min():.4f}, max={weights_train.max():.4f}")
    print(f"  Validation set - mean={weights_val.mean():.4f}, min={weights_val.min():.4f}, max={weights_val.max():.4f}")
    
    return X_train, X_val, y_train, y_val, weights_train, weights_val


def train_model(X_train, y_train, weights_train, **xgb_params):
    """
    Train XGBoost classifier with sample weights.
    
    Args:
        X_train: Training features
        y_train: Training labels
        weights_train: Sample weights (confidence scores)
        **xgb_params: Additional XGBoost parameters
        
    Returns:
        Trained model
    """
    # Default parameters
    default_params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 1,
    }
    
    # Update with custom parameters
    default_params.update(xgb_params)
    
    print(f"\nTraining XGBoost model with parameters:")
    for k, v in default_params.items():
        print(f"  {k}: {v}")
    
    model = xgb.XGBClassifier(**default_params)
    
    # Train with sample weights
    print(f"\nTraining on {len(X_train)} samples...")
    model.fit(X_train, y_train, sample_weight=weights_train, verbose=True)
    
    return model


def evaluate_model(model, X_train, X_val, y_train, y_val, weights_train=None, weights_val=None):
    """
    Evaluate model on training and validation sets.
    
    Args:
        model: Trained XGBoost model
        X_train, X_val: Training and validation features
        y_train, y_val: Training and validation labels
        weights_train, weights_val: Optional sample weights for weighted metrics
        
    Returns:
        Dictionary with all evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Probabilities for ROC curve
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    results = {}
    
    # Training metrics
    print(f"\n--- TRAINING SET ---")
    train_acc = accuracy_score(y_train, y_train_pred, sample_weight=weights_train)
    train_prec = precision_score(y_train, y_train_pred, sample_weight=weights_train)
    train_rec = recall_score(y_train, y_train_pred, sample_weight=weights_train)
    train_f1 = f1_score(y_train, y_train_pred, sample_weight=weights_train)
    train_auc = roc_auc_score(y_train, y_train_proba, sample_weight=weights_train)
    
    print(f"Accuracy:  {train_acc:.4f}")
    print(f"Precision: {train_prec:.4f}")
    print(f"Recall:    {train_rec:.4f}")
    print(f"F1-Score:  {train_f1:.4f}")
    print(f"ROC-AUC:   {train_auc:.4f}")
    
    results['train'] = {
        'accuracy': train_acc,
        'precision': train_prec,
        'recall': train_rec,
        'f1': train_f1,
        'auc': train_auc,
    }
    
    # Validation metrics
    print(f"\n--- VALIDATION SET ---")
    val_acc = accuracy_score(y_val, y_val_pred, sample_weight=weights_val)
    val_prec = precision_score(y_val, y_val_pred, sample_weight=weights_val)
    val_rec = recall_score(y_val, y_val_pred, sample_weight=weights_val)
    val_f1 = f1_score(y_val, y_val_pred, sample_weight=weights_val)
    val_auc = roc_auc_score(y_val, y_val_proba, sample_weight=weights_val)
    
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"Precision: {val_prec:.4f}")
    print(f"Recall:    {val_rec:.4f}")
    print(f"F1-Score:  {val_f1:.4f}")
    print(f"ROC-AUC:   {val_auc:.4f}")
    
    results['validation'] = {
        'accuracy': val_acc,
        'precision': val_prec,
        'recall': val_rec,
        'f1': val_f1,
        'auc': val_auc,
    }
    
    # Confusion matrices
    print(f"\n--- CONFUSION MATRIX ---")
    train_cm = confusion_matrix(y_train, y_train_pred)
    val_cm = confusion_matrix(y_val, y_val_pred)
    
    print(f"\nTraining set:")
    print(f"  TN={train_cm[0,0]}, FP={train_cm[0,1]}")
    print(f"  FN={train_cm[1,0]}, TP={train_cm[1,1]}")
    
    print(f"\nValidation set:")
    print(f"  TN={val_cm[0,0]}, FP={val_cm[0,1]}")
    print(f"  FN={val_cm[1,0]}, TP={val_cm[1,1]}")
    
    results['confusion_matrices'] = {'train': train_cm, 'validation': val_cm}
    results['predictions'] = {
        'train': y_train_pred,
        'val': y_val_pred,
        'train_proba': y_train_proba,
        'val_proba': y_val_proba,
    }
    
    print("\n" + "="*70)
    
    return results


def plot_feature_importance(model, feature_names=['intraday_range', 'volume_change_rate', 'rolling_historical_volatility']):
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


def plot_roc_curves(y_train, y_val, y_train_proba, y_val_proba):
    """Plot ROC curves for training and validation sets."""
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
    
    auc_train = auc(fpr_train, tpr_train)
    auc_val = auc(fpr_val, tpr_val)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_train, tpr_train, label=f'Training (AUC = {auc_train:.4f})', linewidth=2)
    plt.plot(fpr_val, tpr_val, label=f'Validation (AUC = {auc_val:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Binary Volatility Classification')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    print("Saved: roc_curves.png")
    plt.show()


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, cmap=plt.cm.Blues)
    
    # Labels
    classes = ['Low Volatility', 'High Volatility']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    return plt


def main():
    """Main training pipeline."""
    print("="*70)
    print("XGBoost Binary Volatility Classification")
    print("="*70)
    
    # Load data
    data = load_dataset()
    
    # Prepare data (90% train, 10% validation)
    X_train, X_val, y_train, y_val, weights_train, weights_val = prepare_data(data, test_size=0.1)
    
    # Train model
    model = train_model(
        X_train, y_train, weights_train,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    
    # Evaluate model
    results = evaluate_model(model, X_train, X_val, y_train, y_val, weights_train, weights_val)
    
    # Feature importance
    print("\nFeature Importances:")
    for fname, importance in zip(['intraday_range', 'volume_change_rate', 'rolling_historical_volatility'], 
                                  model.feature_importances_):
        print(f"  {fname}: {importance:.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    
    # Feature importance
    plot_feature_importance(model)
    
    # ROC curves
    plot_roc_curves(y_train, y_val, 
                   results['predictions']['train_proba'],
                   results['predictions']['val_proba'])
    
    # Confusion matrices
    fig_train = plot_confusion_matrix(results['confusion_matrices']['train'], 'Confusion Matrix - Training Set')
    fig_train.savefig('confusion_matrix_train.png', dpi=300)
    print("Saved: confusion_matrix_train.png")
    fig_train.show()
    
    fig_val = plot_confusion_matrix(results['confusion_matrices']['validation'], 'Confusion Matrix - Validation Set')
    fig_val.savefig('confusion_matrix_validation.png', dpi=300)
    print("Saved: confusion_matrix_validation.png")
    fig_val.show()
    
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
