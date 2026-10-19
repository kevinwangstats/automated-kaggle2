"""
Universal Metric Registry for Kaggle Competitions.

Provides deterministic, leak-free metric implementations supporting:
- Regression: rmse, rmsle, mae, r2
- Classification: roc_auc, logloss, multi_logloss, f1, macro_f1, mcc
- Ordinal: qwk (Quadratic Weighted Kappa)
"""

from typing import Callable, Tuple
import numpy as np
from sklearn import metrics as sk_metrics


MetricFunction = Callable[[np.ndarray, np.ndarray], float]


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true length {len(y_t)} != y_pred length {len(y_p)}")

    if np.isnan(y_p).any():
        raise ValueError("y_pred contains NaN values!")
    if np.isinf(y_p).any():
        raise ValueError("y_pred contains Infinite values!")

    return y_t, y_p


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = _validate_inputs(y_true, y_pred)
    # Clip negative values to zero to avoid log of negative numbers
    y_t_clipped = np.clip(y_t, 0.0, None)
    y_p_clipped = np.clip(y_p, 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(y_t_clipped) - np.log1p(y_p_clipped)) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return float(sk_metrics.r2_score(y_t, y_p))


def compute_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = _validate_inputs(y_true, y_pred)
    # Check if multi-class (2D prediction array or >2 unique classes)
    if y_p.ndim > 1 and y_p.shape[1] > 2:
        return float(sk_metrics.roc_auc_score(y_t, y_p, multi_class="ovr", average="macro"))
    elif y_p.ndim > 1 and y_p.shape[1] == 2:
        # Take probability of positive class
        y_p = y_p[:, 1]
    return float(sk_metrics.roc_auc_score(y_t, y_p))


def compute_logloss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_t, y_p = _validate_inputs(y_true, y_pred)
    if y_p.ndim > 1 and y_p.shape[1] == 2:
        y_p = y_p[:, 1]
    y_p_clipped = np.clip(y_p, eps, 1.0 - eps)
    return float(sk_metrics.log_loss(y_t, y_p_clipped))


def compute_multi_logloss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_t, y_p = _validate_inputs(y_true, y_pred)
    y_p_clipped = np.clip(y_p, eps, 1.0 - eps)
    # Renormalize rows so they sum to 1
    if y_p_clipped.ndim > 1:
        y_p_clipped = y_p_clipped / y_p_clipped.sum(axis=1, keepdims=True)
    return float(sk_metrics.log_loss(y_t, y_p_clipped))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_t, y_p = _validate_inputs(y_true, y_pred)
    if y_p.ndim == 1:
        binary_preds = (y_p >= threshold).astype(int)
    else:
        binary_preds = np.argmax(y_p, axis=1)
    return float(sk_metrics.f1_score(y_t, binary_preds, average="binary" if len(np.unique(y_t)) <= 2 else "macro"))


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = _validate_inputs(y_true, y_pred)
    preds = np.argmax(y_p, axis=1) if y_p.ndim > 1 else np.round(y_p).astype(int)
    return float(sk_metrics.f1_score(y_t, preds, average="macro"))


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_t, y_p = _validate_inputs(y_true, y_pred)
    preds = (y_p >= threshold).astype(int) if y_p.ndim == 1 else np.argmax(y_p, axis=1)
    return float(sk_metrics.matthews_corrcoef(y_t, preds))


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = _validate_inputs(y_true, y_pred)
    # If continuous predictions, round to nearest integer
    preds = np.round(y_p).astype(int) if pd_is_float(y_p) else y_p.astype(int)
    return float(sk_metrics.cohen_kappa_score(y_t.astype(int), preds, weights="quadratic"))


def pd_is_float(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.floating)


class MetricRegistry:
    """Registry mapping metric names to callables and optimization direction."""

    _METRICS: dict[str, Tuple[MetricFunction, bool]] = {
        "rmse": (compute_rmse, False),
        "rmsle": (compute_rmsle, False),
        "mae": (compute_mae, False),
        "r2": (compute_r2, True),
        "roc_auc": (compute_roc_auc, True),
        "auc": (compute_roc_auc, True),
        "logloss": (compute_logloss, False),
        "multi_logloss": (compute_multi_logloss, False),
        "f1": (compute_f1, True),
        "macro_f1": (compute_macro_f1, True),
        "mcc": (compute_mcc, True),
        "qwk": (compute_qwk, True),
    }

    @classmethod
    def get_metric(cls, metric_name: str) -> Tuple[MetricFunction, bool]:
        """
        Returns:
            (metric_fn, greater_is_better)
        """
        key = metric_name.lower().strip()
        if key not in cls._METRICS:
            raise ValueError(f"Unsupported metric '{metric_name}'. Registered: {list(cls._METRICS.keys())}")
        return cls._METRICS[key]

    @classmethod
    def evaluate(cls, metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        fn, _ = cls.get_metric(metric_name)
        return fn(y_true, y_pred)

    @classmethod
    def is_greater_better(cls, metric_name: str) -> bool:
        _, greater_is_better = cls.get_metric(metric_name)
        return greater_is_better

    @classmethod
    def is_improvement(cls, metric_name: str, new_score: float, baseline_score: float, epsilon: float = 0.0) -> bool:
        """Determines if new_score beats baseline_score by at least epsilon."""
        greater_is_better = cls.is_greater_better(metric_name)
        if greater_is_better:
            return (new_score - baseline_score) > epsilon
        else:
            return (baseline_score - new_score) > epsilon
