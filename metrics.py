import numpy as np

def get_tp(truth: list[str], preds: list[str], c: str) -> int:
    return np.sum((preds == truth)[truth == c])

def get_tn(truth: list[str], preds: list[str], c: str) -> int:
    return np.sum((preds == truth)[truth != c])

def get_fp(truth: list[str], preds: list[str], c: str) -> int:
    return np.sum((preds == c)[truth != c])

def get_fn(truth: list[str], preds: list[str], c: str) -> int:
    return np.sum((preds != c)[truth == c])

def accuracy(truth: list[str], preds: list[str]) -> float:
    correct = np.sum(preds == truth)
    return correct / preds.shape[0]

def precision(truth: list[str], preds: list[str], c: str) -> float:
    tp = get_tp(truth, preds, c)
    fp = get_fp(truth, preds, c)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def avg_precision(truth: list[str], preds: list[str]) -> float:
    """Returns the average precision across all classes."""
    precs = []
    for cls in np.unique(truth):
        precs.append(precision(truth, preds, cls))
    return float(np.mean(precs))




