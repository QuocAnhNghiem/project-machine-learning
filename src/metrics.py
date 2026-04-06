import numpy as np


def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary log loss: J = -(1/m) * sum(y*log(p) + (1-y)*log(1-p))."""
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return TP, TN, FP, FN counts from y_true and y_pred."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute standard binary metrics.

    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1        = 2 * precision * recall / (precision + recall)
    """
    cm = confusion_matrix(y_true, y_pred)
    total = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]

    accuracy = (cm["tp"] + cm["tn"]) / total if total else 0.0
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0.0
    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "log_loss": log_loss(y_true, y_proba),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm,
    }
