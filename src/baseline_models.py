import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from data_loader import CHBMITDataset


SplitData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def prepare_dataset(data_dir: Path, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> SplitData:
    """Load windows and create flattened train/val/test splits."""
    dataset = CHBMITDataset(data_dir)

    X = dataset.windows.astype(np.float32)
    y = dataset.labels.astype(np.int64)

    n_samples = len(y)
    X = X.reshape(n_samples, -1)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
) -> Dict[str, float | str]:
    """Compute evaluation metrics for binary classification."""
    metrics: Dict[str, float | str] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": float("nan"),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
    }

    if y_scores is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics["auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) else 0.0

    return metrics


def get_prediction_scores(model, X: np.ndarray) -> np.ndarray | None:
    """Return probability or decision scores when available."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def train_and_evaluate(
    name: str,
    model,
    use_scaler: bool,
    splits: SplitData,
) -> Dict[str, float | str]:
    """Fit the model and evaluate on validation and test splits."""
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    scaler = StandardScaler() if use_scaler else None

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_scores = get_prediction_scores(model, X_val)
    test_scores = get_prediction_scores(model, X_test)

    val_metrics = compute_metrics(y_val, y_val_pred, val_scores)
    test_metrics = compute_metrics(y_test, y_test_pred, test_scores)

    return {
        "model": name,
        "val_accuracy": val_metrics["accuracy"],
        "val_auc": val_metrics["auc"],
        "val_sensitivity": val_metrics["sensitivity"],
        "val_specificity": val_metrics["specificity"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_sensitivity": test_metrics["sensitivity"],
        "test_specificity": test_metrics["specificity"],
    }


def build_models() -> Dict[str, Tuple[object, bool]]:
    """Create baseline models and indicate whether scaling is required."""
    models: Dict[str, Tuple[object, bool]] = {}

    # Logistic Regression
    models["logistic_regression"] = (
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        True,
    )

    # SVM
    models["svm"] = (
        SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            gamma="scale",
        ),
        True,
    )

    # Random Forest
    models["random_forest"] = (
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        False,
    )

    return models


def main():
    parser = argparse.ArgumentParser(description="Train baseline ML models on EEG dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to root directory containing dataset",
    )
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    # Model selection
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "svm", "random_forest", "all"],
        default="all",
        help="Which model to train (default: all baselines).",
    )
    args = parser.parse_args()

    splits = prepare_dataset(args.data, args.train_split, args.val_split, args.seed)
    models = build_models()

    selected_models = (
        models.items()
        if args.model == "all"
        else [(args.model, models[args.model])]
    )

    for name, (model, use_scaler) in selected_models:
        print(f"Training {name} baseline...")
        model_results = train_and_evaluate(name, model, use_scaler, splits)

        print(
            f"{name} | "
            f"Val Acc: {model_results['val_accuracy']:.3f}, "
            f"Val AUC: {model_results['val_auc']:.3f}, "
            f"Val Sensitivity: {model_results['val_sensitivity']:.3f}, "
            f"Val Specificity: {model_results['val_specificity']:.3f} | "
            f"Test Acc: {model_results['test_accuracy']:.3f}, "
            f"Test AUC: {model_results['test_auc']:.3f}, "
            f"Test Sensitivity: {model_results['test_sensitivity']:.3f}, "
            f"Test Specificity: {model_results['test_specificity']:.3f}"
        )


if __name__ == "__main__":
    main()

