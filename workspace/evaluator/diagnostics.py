"""
Automated Diagnostic Auditor and Report Compiler.

Generates a compact (< 1,500 tokens) diagnostics_report.json from:
- train_folds.parquet (with ground truth and fold column)
- artifacts/oof_preds.npy (out-of-fold predictions)
- test.csv (for adversarial validation)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor, _tree

from workspace.evaluator.metrics import MetricRegistry
from workspace.evaluator.adversarial import run_adversarial_validation


def compute_calibration_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across n_bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return round(float(ece), 4)


def extract_error_slices(
    X: pd.DataFrame, sample_losses: np.ndarray, max_slices: int = 3
) -> List[Dict[str, Any]]:
    """
    Fits a surrogate decision tree to sample-level losses and extracts the
    decision rules associated with the highest-error leaf clusters.
    """
    # Preprocess features for surrogate tree
    X_num = pd.DataFrame(index=X.index)
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            X_num[col] = s.fillna(s.median())
        else:
            freq = s.value_counts(normalize=True).to_dict()
            X_num[col] = s.map(freq).fillna(0.0)

    min_leaf = max(25, int(len(X) * 0.04))
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=min_leaf, random_state=42)
    tree.fit(X_num, sample_losses)

    tree_ = tree.tree_
    feature_names = list(X_num.columns)
    global_mean_loss = float(np.mean(sample_losses))

    leaf_nodes = []

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [f"({name} <= {threshold:.2f})"])
            recurse(tree_.children_right[node], conditions + [f"({name} > {threshold:.2f})"])
        else:
            # Leaf node
            avg_loss = float(tree_.value[node][0][0])
            count = int(tree_.n_node_samples[node])
            if len(conditions) > 0:
                leaf_nodes.append({
                    "condition": " & ".join(conditions),
                    "slice_loss": round(avg_loss, 4),
                    "global_loss": round(global_mean_loss, 4),
                    "sample_count": count,
                })

    recurse(0, [])
    # Sort by highest slice loss
    leaf_nodes.sort(key=lambda x: x["slice_loss"], reverse=True)
    
    # Filter to slices with error above average if possible, otherwise return highest error leaves
    elevated = [leaf for leaf in leaf_nodes if leaf["slice_loss"] >= global_mean_loss]
    selected = elevated if elevated else leaf_nodes
    return selected[:max_slices]


def generate_diagnostics_report(
    folds_path: Path,
    oof_path: Path,
    test_path: Optional[Path] = None,
    metric_name: Optional[str] = None,
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
    problem_type: Optional[str] = None,
    iteration: int = 1,
) -> Dict[str, Any]:
    folds_path = Path(folds_path).resolve()
    oof_path = Path(oof_path).resolve()

    if not folds_path.exists():
        raise FileNotFoundError(f"train_folds.parquet not found: {folds_path}")
    if not oof_path.exists():
        raise FileNotFoundError(f"oof_preds.npy not found: {oof_path}")

    # Load competition metadata if available
    meta_path = folds_path.parent / "competition_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            metric_name = metric_name or meta.get("evaluation_metric")
            target_col = target_col or meta.get("target_column")
            id_col = id_col or meta.get("id_column")
            problem_type = problem_type or meta.get("problem_type")

    metric_name = (metric_name or "roc_auc").lower().strip()
    df = pd.read_parquet(folds_path)
    oof_preds = np.load(oof_path)

    if target_col is None or target_col not in df.columns:
        # Fallback to non-fold, non-id column
        potential = [c for c in df.columns if c not in ["fold", id_col, "id", "PassengerId"]]
        target_col = potential[-1] if potential else df.columns[0]

    y_true = df[target_col].to_numpy()

    if problem_type is None:
        if metric_name in ("rmse", "rmsle", "mae", "r2") or (pd.api.types.is_float_dtype(y_true) and len(np.unique(y_true)) > 50):
            problem_type = "regression"
        elif len(np.unique(y_true)) == 2:
            problem_type = "binary_classification"
        else:
            problem_type = "multiclass_classification"

    if len(df) != len(oof_preds):
        raise ValueError(f"Length mismatch: folds rows {len(df)} != oof_preds {len(oof_preds)}")

    # 1. Fold-by-fold and global metric evaluation
    fold_scores = []
    unique_folds = sorted(df["fold"].unique())
    for f in unique_folds:
        mask = df["fold"] == f
        f_true = y_true[mask]
        f_pred = oof_preds[mask]
        score = MetricRegistry.evaluate(metric_name, f_true, f_pred)
        fold_scores.append(round(score, 5))

    global_score = round(MetricRegistry.evaluate(metric_name, y_true, oof_preds), 5)
    fold_std = round(float(np.std(fold_scores, ddof=1)), 5) if len(fold_scores) > 1 else 0.0

    if fold_std < 0.015:
        stability = "HIGH"
    elif fold_std < 0.035:
        stability = "MEDIUM"
    else:
        stability = "LOW"

    # 2. Residual & Calibration Mechanics
    residual_analysis: Dict[str, Any] = {}
    if problem_type in ("binary_classification", "multiclass_classification"):
        # Probabilities
        brier = float(np.mean((oof_preds - y_true) ** 2))
        binary_pred_50 = (oof_preds >= 0.5).astype(int)
        fp = int(np.sum((binary_pred_50 == 1) & (y_true == 0)))
        fn = int(np.sum((binary_pred_50 == 0) & (y_true == 1)))
        fp_fn_ratio = round(float(fp / max(fn, 1)), 2)
        ece = compute_calibration_ece(y_true, oof_preds, n_bins=10)

        residual_analysis = {
            "brier_score": round(brier, 4),
            "fp_count": fp,
            "fn_count": fn,
            "fp_fn_ratio": fp_fn_ratio,
            "calibration_ece": ece,
        }
        sample_losses = np.abs(oof_preds - y_true)
    else:
        # Regression
        residuals = y_true - oof_preds
        abs_residuals = np.abs(residuals)
        residual_skewness = round(float(stats.skew(residuals)), 4)
        corr_res = round(float(np.corrcoef(abs_residuals, oof_preds)[0, 1]), 4)
        if math.isnan(corr_res):
            corr_res = 0.0

        residual_analysis = {
            "residual_skewness": residual_skewness,
            "heteroskedasticity_corr": corr_res,
            "abs_error_p95": round(float(np.percentile(abs_residuals, 95)), 4),
            "abs_error_p99": round(float(np.percentile(abs_residuals, 99)), 4),
        }
        sample_losses = abs_residuals

    # 3. Error Slice Profiling via Surrogate Tree
    feature_cols = [c for c in df.columns if c not in [target_col, id_col, "fold"]]
    X_features = df[feature_cols]
    error_slices = extract_error_slices(X_features, sample_losses, max_slices=3)

    # 4. Feature Importance & Pruning Candidates (using surrogate tree or lightgbm)
    feature_importances = []
    zero_imp_features = []
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        X_num = pd.DataFrame(index=X_features.index)
        for col in X_features.columns:
            s = X_features[col]
            if pd.api.types.is_numeric_dtype(s):
                X_num[col] = s.fillna(s.median())
            else:
                X_num[col] = s.map(s.value_counts(normalize=True).to_dict()).fillna(0.0)

        forest = ExtraTreesRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
        forest.fit(X_num, y_true)
        importances = forest.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        for idx in sorted_indices[:15]:
            imp = float(importances[idx])
            if imp > 0.005:
                feature_importances.append({"name": feature_cols[idx], "importance": round(imp, 4)})
        for idx in sorted_indices:
            if importances[idx] < 0.001:
                zero_imp_features.append(feature_cols[idx])
    except Exception as e:
        print(f"Warning computing feature importances: {e}")

    # 5. Adversarial Validation
    adv_results = {"adversarial_auc": 0.5, "covariate_shift_detected": False, "drifting_features": []}
    if test_path and Path(test_path).exists():
        test_df = pd.read_csv(test_path)
        adv_results = run_adversarial_validation(
            train_df=df, test_df=test_df, target_col=target_col, id_col=id_col
        )

    report = {
        "iteration": iteration,
        "metric_name": metric_name,
        "global_cv": global_score,
        "fold_scores": fold_scores,
        "fold_std": fold_std,
        "stability_assessment": stability,
        "residual_analysis": residual_analysis,
        "error_slices": error_slices,
        "top_features": feature_importances,
        "zero_importance_features": zero_imp_features[:5],
        "adversarial_validation": adv_results,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostics_report.json.")
    parser.add_argument("--folds-path", type=str, default="workspace/data/train_folds.parquet")
    parser.add_argument("--oof-path", type=str, default="workspace/artifacts/oof_preds.npy")
    parser.add_argument("--test-path", type=str, default="workspace/data/raw/test.csv")
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--output", type=str, default="workspace/memory/diagnostics_report.json")
    args = parser.parse_args()

    report = generate_diagnostics_report(
        folds_path=Path(args.folds_path),
        oof_path=Path(args.oof_path),
        test_path=Path(args.test_path),
        metric_name=args.metric,
        iteration=args.iteration,
    )

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(report, f, indent=2)

    json_str = json.dumps(report)
    approx_tokens = len(json_str) / 4
    print(f"Diagnostics report saved to: {out_p}")
    print(f"  CV: {report['global_cv']} (std: {report['fold_std']})")
    print(f"  Estimated token footprint: {approx_tokens:.0f} tokens (Limit: < 1500)")


if __name__ == "__main__":
    main()
