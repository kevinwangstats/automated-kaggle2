"""
Kaggle Ingestion and Metadata Parsing Pipeline.

Downloads competition data (if not already present locally) using the Kaggle API/CLI,
unpacks archives, and automatically determines:
- ID column
- Target column
- Problem type (binary_classification, multiclass_classification, regression)
- Suggested default evaluation metric

Outputs:
- workspace/data/competition_meta.json
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, Optional
import pandas as pd
import zipfile


def download_competition(slug: str, raw_dir: Path, kaggle_bin: Optional[str] = None) -> None:
    """Invokes kaggle CLI to download competition files into raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if train.csv already exists
    if (raw_dir / "train.csv").exists():
        print(f"Data files already exist in {raw_dir}, skipping download.")
        return

    kaggle_cmd = kaggle_bin or shutil.which("kaggle") or "kaggle"
    print(f"Downloading competition '{slug}' to {raw_dir} using {kaggle_cmd}...")

    cmd = [kaggle_cmd, "competitions", "download", "-c", slug, "-p", str(raw_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Kaggle download failed:\n{res.stderr}\n{res.stdout}")

    # Unpack any downloaded zip archives
    for zip_path in raw_dir.glob("*.zip"):
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)


def detect_id_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Optional[str]:
    """Auto-detects the identifier column based on regex and uniqueness."""
    id_pattern = re.compile(r"^(id|row_id|uid|passengerid|index)$", re.IGNORECASE)
    common_cols = [c for c in train_df.columns if c in test_df.columns]

    # First priority: matching standard ID regex
    for col in common_cols:
        if id_pattern.match(col):
            return col

    # Second priority: string/integer column that is strictly unique in both train and test
    for col in common_cols:
        if train_df[col].nunique() == len(train_df) and test_df[col].nunique() == len(test_df):
            if not pd.api.types.is_float_dtype(train_df[col]):
                return col

    return None


def detect_target_and_problem(
    train_df: pd.DataFrame, test_df: pd.DataFrame, explicit_target: Optional[str] = None
) -> tuple[str, str, str]:
    """
    Detects:
    - target_column: present in train_df but missing in test_df
    - problem_type: binary_classification | multiclass_classification | regression
    - default_metric: roc_auc | logloss | rmse
    """
    if explicit_target:
        target_col = explicit_target
    else:
        diff_cols = [c for c in train_df.columns if c not in test_df.columns]
        if not diff_cols:
            raise ValueError("Could not auto-detect target column: train and test have identical columns.")
        # If multiple diff columns, prefer standard target names
        pref_pattern = re.compile(r"(target|label|survived|outcome|class|status|price|cost)", re.IGNORECASE)
        matched = [c for c in diff_cols if pref_pattern.search(c)]
        target_col = matched[0] if matched else diff_cols[0]

    target_series = train_df[target_col].dropna()
    num_unique = target_series.nunique()
    is_float = pd.api.types.is_float_dtype(target_series)

    if num_unique == 2:
        problem_type = "binary_classification"
        default_metric = "roc_auc"
    elif 3 <= num_unique <= 50 and not is_float:
        problem_type = "multiclass_classification"
        default_metric = "multi_logloss"
    else:
        problem_type = "regression"
        default_metric = "rmse"

    return target_col, problem_type, default_metric


def run_ingestion(
    competition_slug: str,
    workspace_dir: Path,
    kaggle_bin: Optional[str] = None,
    explicit_target: Optional[str] = None,
) -> Dict[str, Any]:
    workspace_dir = Path(workspace_dir).resolve()
    raw_dir = workspace_dir / "data" / "raw"
    data_dir = workspace_dir / "data"

    download_competition(competition_slug, raw_dir, kaggle_bin)

    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected train.csv and test.csv in {raw_dir}")

    # Read minimal head for fast schema inference
    train_head = pd.read_csv(train_path, nrows=5000)
    test_head = pd.read_csv(test_path, nrows=1000)

    id_col = detect_id_column(train_head, test_head)
    target_col, problem_type, default_metric = detect_target_and_problem(
        train_head, test_head, explicit_target
    )

    # Read true row counts
    train_rows = sum(1 for _ in open(train_path)) - 1
    test_rows = sum(1 for _ in open(test_path)) - 1

    metadata = {
        "competition_slug": competition_slug,
        "id_column": id_col,
        "target_column": target_col,
        "problem_type": problem_type,
        "evaluation_metric": default_metric,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_features_count": len(train_head.columns) - 1 - (1 if id_col else 0),
        "raw_train_columns": list(train_head.columns),
        "raw_test_columns": list(test_head.columns),
    }

    meta_file = data_dir / "competition_meta.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata generated successfully at: {meta_file}")
    print(f"  Target: {target_col} | Problem: {problem_type} | Metric: {default_metric} | ID: {id_col}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Ingest Kaggle competition dataset.")
    parser.add_argument("--competition", type=str, required=True, help="Kaggle competition slug")
    parser.add_argument("--workspace-dir", type=str, default="workspace", help="Workspace root dir")
    parser.add_argument("--kaggle-bin", type=str, default=None, help="Path to kaggle executable")
    parser.add_argument("--target", type=str, default=None, help="Explicit target column name")
    args = parser.parse_args()

    run_ingestion(
        competition_slug=args.competition,
        workspace_dir=Path(args.workspace_dir),
        kaggle_bin=args.kaggle_bin,
        explicit_target=args.target,
    )


if __name__ == "__main__":
    main()
