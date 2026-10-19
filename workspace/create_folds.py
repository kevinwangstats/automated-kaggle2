"""
Immutable Cross-Validation (CV) Generator.

Generates workspace/data/train_folds.parquet with a deterministic 'fold' column
using StratifiedKFold, Quantile-Binned StratifiedKFold, GroupKFold, or TimeSeriesSplit.
Once written, enforces read-only immutability (chmod 444) to prevent data leakage.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit

from workspace.permissions import lock_folds_file, unlock_folds_file


def create_folds(
    train_path: Path,
    output_path: Path,
    n_splits: int = 5,
    target_col: Optional[str] = None,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    problem_type: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Splits train.csv into k folds and saves as train_folds.parquet with read-only lock.
    """
    train_path = Path(train_path).resolve()
    output_path = Path(output_path).resolve()

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    # Load competition metadata if available
    meta_path = train_path.parent.parent / "competition_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            target_col = target_col or meta.get("target_column")
            problem_type = problem_type or meta.get("problem_type")

    df = pd.read_csv(train_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {train_path}")

    df["fold"] = -1

    # Select split strategy
    if group_col and group_col in df.columns:
        print(f"Applying GroupKFold(n_splits={n_splits}) on group column '{group_col}'...")
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(df, df[target_col], groups=df[group_col])
    elif time_col and time_col in df.columns:
        print(f"Applying TimeSeriesSplit(n_splits={n_splits}) sorted on time column '{time_col}'...")
        df = df.sort_values(by=time_col).reset_index(drop=True)
        splitter = TimeSeriesSplit(n_splits=n_splits)
        splits = splitter.split(df)
    elif problem_type == "regression" or (
        pd.api.types.is_float_dtype(df[target_col]) and df[target_col].nunique() > 50
    ):
        print(f"Applying Quantile-Binned StratifiedKFold(n_splits={n_splits}) for continuous regression...")
        # Bin continuous target into 10 quantiles for balanced stratification
        clean_target = df[target_col].fillna(df[target_col].median())
        try:
            target_bins = pd.qcut(clean_target, q=min(10, df[target_col].nunique()), labels=False, duplicates="drop")
        except Exception:
            target_bins = pd.cut(clean_target, bins=10, labels=False)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(df, target_bins)
    else:
        # Binary or multiclass classification
        print(f"Applying StratifiedKFold(n_splits={n_splits}, shuffle=True, seed={random_state})...")
        clean_target = df[target_col].fillna(df[target_col].mode()[0])
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(df, clean_target)

    for fold_idx, (_, val_idx) in enumerate(splits):
        df.loc[val_idx, "fold"] = fold_idx

    assert (df["fold"] >= 0).all(), "Unassigned folds detected!"
    fold_counts = df["fold"].value_counts().to_dict()
    print(f"Folds generated successfully. Distribution: {fold_counts}")

    # Verify target distribution across folds
    if problem_type in ("binary_classification", "multiclass_classification"):
        for f in range(n_splits):
            fold_dist = df[df["fold"] == f][target_col].value_counts(normalize=True).to_dict()
            print(f"  Fold {f} target distribution: { {k: round(v, 4) for k, v in fold_dist.items()} }")

    # If output file already exists, unlock it temporarily to overwrite
    if output_path.exists():
        unlock_folds_file(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Saved folds to {output_path}")

    # Enforce read-only permission lock
    lock_folds_file(output_path)
    print(f"Enforced read-only lock (chmod 444) on {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Create immutable train_folds.parquet.")
    parser.add_argument("--workspace-dir", type=str, default="workspace", help="Workspace root")
    parser.add_argument("--data-dir", type=str, default=None, help="Raw data directory")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--group", type=str, default=None, help="Group column name")
    parser.add_argument("--time", type=str, default=None, help="Time column name")
    args = parser.parse_args()

    ws = Path(args.workspace_dir)
    train_p = Path(args.data_dir) / "train.csv" if args.data_dir else ws / "data" / "raw" / "train.csv"
    out_p = ws / "data" / "train_folds.parquet"

    create_folds(
        train_path=train_p,
        output_path=out_p,
        n_splits=args.folds,
        target_col=args.target,
        group_col=args.group,
        time_col=args.time,
    )


if __name__ == "__main__":
    main()
