"""
Automated Exploratory Data Analysis (EDA) Scanner.

Extracts statistical primitives, skewness, missingness, cardinality, and target distributions
into workspace/data/eda_summary.json without heavy visualization dependencies.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats


def calculate_skewness(series: pd.Series) -> Optional[float]:
    """Calculates Fisher-Pearson standardized moment coefficient skewness."""
    clean = series.dropna().to_numpy(dtype=float)
    if len(clean) < 3:
        return None
    s = float(stats.skew(clean, bias=False))
    return None if (math.isnan(s) or math.isinf(s)) else round(s, 4)


def scan_dataset(
    train_path: Path,
    test_path: Path,
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> Dict[str, Any]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    # Load competition metadata if available
    meta_path = train_path.parent.parent / "competition_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            target_col = target_col or meta.get("target_column")
            id_col = id_col or meta.get("id_column")

    features_profile: Dict[str, Any] = {}
    feature_cols = [c for c in train_df.columns if c != target_col and c != id_col]

    for col in feature_cols:
        col_series = train_df[col]
        test_series = test_df[col] if (test_df is not None and col in test_df.columns) else None

        train_missing = float(col_series.isna().mean())
        test_missing = float(test_series.isna().mean()) if test_series is not None else 0.0

        is_numeric = pd.api.types.is_numeric_dtype(col_series)
        unique_count = int(col_series.nunique(dropna=True))

        col_stats: Dict[str, Any] = {
            "dtype": str(col_series.dtype),
            "is_numeric": is_numeric,
            "train_missing_ratio": round(train_missing, 4),
            "test_missing_ratio": round(test_missing, 4),
            "unique_count": unique_count,
        }

        if is_numeric and unique_count > 10:
            clean_train = col_series.dropna()
            q25 = float(clean_train.quantile(0.25))
            q75 = float(clean_train.quantile(0.75))
            col_stats.update({
                "min": round(float(clean_train.min()), 4),
                "max": round(float(clean_train.max()), 4),
                "mean": round(float(clean_train.mean()), 4),
                "median": round(float(clean_train.median()), 4),
                "std": round(float(clean_train.std()), 4),
                "iqr": round(q75 - q25, 4),
                "skewness": calculate_skewness(clean_train),
            })
        else:
            # Categorical or low-cardinality discrete
            val_counts = col_series.value_counts(normalize=True, dropna=True)
            top_cats = {str(k): round(float(v), 4) for k, v in val_counts.head(5).items()}
            has_rare = any(v < 0.01 for v in val_counts.values)

            unseen_in_test = 0
            if test_series is not None:
                train_cats = set(col_series.dropna().astype(str).unique())
                test_cats = set(test_series.dropna().astype(str).unique())
                unseen_in_test = len(test_cats - train_cats)

            col_stats.update({
                "top_categories": top_cats,
                "has_rare_categories": has_rare,
                "test_unseen_categories_count": unseen_in_test,
            })

        features_profile[col] = col_stats

    # Target distribution analysis
    target_profile: Dict[str, Any] = {}
    if target_col and target_col in train_df.columns:
        target_s = train_df[target_col].dropna()
        target_unique = target_s.nunique()
        if target_unique <= 50 and not pd.api.types.is_float_dtype(target_s):
            val_counts = target_s.value_counts(normalize=True).to_dict()
            target_profile = {
                "type": "discrete",
                "classes": [str(k) for k in val_counts.keys()],
                "class_distribution": {str(k): round(float(v), 4) for k, v in val_counts.items()},
                "imbalance_ratio": round(float(max(val_counts.values()) / min(val_counts.values())), 2),
            }
        else:
            clean_tgt = target_s.astype(float)
            quantiles = {
                f"q_{int(p*100):02d}": round(float(clean_tgt.quantile(p)), 4)
                for p in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
            }
            target_profile = {
                "type": "continuous",
                "min": round(float(clean_tgt.min()), 4),
                "max": round(float(clean_tgt.max()), 4),
                "mean": round(float(clean_tgt.mean()), 4),
                "std": round(float(clean_tgt.std()), 4),
                "skewness": calculate_skewness(clean_tgt),
                "quantiles": quantiles,
            }

    summary = {
        "train_shape": {"rows": len(train_df), "columns": len(train_df.columns)},
        "test_shape": {"rows": len(test_df), "columns": len(test_df.columns)} if test_df is not None else None,
        "id_column": id_col,
        "target_column": target_col,
        "target_profile": target_profile,
        "features": features_profile,
    }
    return summary


def run_scanner(
    workspace_dir: Path,
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    workspace_dir = Path(workspace_dir).resolve()
    train_path = train_path or (workspace_dir / "data" / "raw" / "train.csv")
    test_path = test_path or (workspace_dir / "data" / "raw" / "test.csv")
    output_path = output_path or (workspace_dir / "data" / "eda_summary.json")

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at: {train_path}")

    summary = scan_dataset(train_path=train_path, test_path=test_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"EDA Summary successfully written to: {output_path}")
    print(f"  Features profiled: {len(summary['features'])}")
    print(f"  Target type: {summary.get('target_profile', {}).get('type')}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run automated EDA scanner.")
    parser.add_argument("--workspace-dir", type=str, default="workspace", help="Workspace root dir")
    parser.add_argument("--data-dir", type=str, default=None, help="Raw data directory")
    args = parser.parse_args()

    ws = Path(args.workspace_dir)
    train_p = Path(args.data_dir) / "train.csv" if args.data_dir else None
    test_p = Path(args.data_dir) / "test.csv" if args.data_dir else None

    run_scanner(workspace_dir=ws, train_path=train_p, test_path=test_p)


if __name__ == "__main__":
    main()
