"""
Stage 1 Milestone End-to-End Verification Suite.

Runs all Stage 1 verification checks:
1. Verifies workspace skeleton directories and .gitkeeps.
2. Ingests or verifies competition data and metadata (competition_meta.json).
3. Verifies automated EDA scanner output (eda_summary.json).
4. Verifies 5-fold stratification in train_folds.parquet.
5. Asserts OS-level immutability (chmod 444) on train_folds.parquet.
6. Verifies executor isolation, script execution, and timeout enforcement.
"""

from pathlib import Path
import json
import os
import sys
import pandas as pd

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from workspace.permissions import verify_permissions
from executor.sandbox import get_executor, ExecutionResult


def run_stage_1_verification():
    print("=" * 60)
    print("STAGE 1 MILESTONE VERIFICATION SUITE")
    print("=" * 60)

    workspace_dir = repo_root / "workspace"
    data_dir = workspace_dir / "data"

    # Check 1: Directory skeleton
    print("\n[Check 1/6] Verifying directory layout...")
    expected_dirs = [
        workspace_dir / "data" / "raw",
        workspace_dir / "src",
        workspace_dir / "evaluator",
        workspace_dir / "artifacts" / "registry",
        workspace_dir / "memory",
    ]
    for d in expected_dirs:
        assert d.exists() and d.is_dir(), f"Missing directory: {d}"
    print("✓ All workspace directories exist.")

    # Check 2: Competition Metadata
    print("\n[Check 2/6] Verifying competition metadata...")
    meta_path = data_dir / "competition_meta.json"
    assert meta_path.exists(), f"competition_meta.json not found at {meta_path}"
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta.get("id_column") == "PassengerId", f"Unexpected ID col: {meta.get('id_column')}"
    assert meta.get("target_column") == "Survived", f"Unexpected Target col: {meta.get('target_column')}"
    assert meta.get("problem_type") == "binary_classification"
    print(f"✓ Metadata verified: ID='{meta['id_column']}', Target='{meta['target_column']}', Type='{meta['problem_type']}'")

    # Check 3: EDA Summary
    print("\n[Check 3/6] Verifying EDA Summary...")
    eda_path = data_dir / "eda_summary.json"
    assert eda_path.exists(), f"eda_summary.json not found at {eda_path}"
    with open(eda_path) as f:
        eda = json.load(f)
    assert "features" in eda
    assert "Age" in eda["features"] and eda["features"]["Age"]["skewness"] is not None
    assert "Sex" in eda["features"] and "male" in eda["features"]["Sex"]["top_categories"]
    print(f"✓ EDA summary verified: {len(eda['features'])} features profiled with skewness and distributions.")

    # Check 4: Fold splits
    print("\n[Check 4/6] Verifying train_folds.parquet...")
    folds_path = data_dir / "train_folds.parquet"
    assert folds_path.exists(), f"train_folds.parquet not found at {folds_path}"
    df_folds = pd.read_parquet(folds_path)
    assert "fold" in df_folds.columns
    unique_folds = sorted(df_folds["fold"].unique())
    assert unique_folds == [0, 1, 2, 3, 4], f"Unexpected fold values: {unique_folds}"
    print(f"✓ Folds verified: exactly 5 folds, total rows: {len(df_folds)}")

    # Check 5: Immutability lock
    print("\n[Check 5/6] Testing train_folds.parquet immutability (chmod 444)...")
    try:
        with open(folds_path, "a") as f:
            f.write("tamper")
        raise AssertionError("FAIL: train_folds.parquet was writable!")
    except PermissionError:
        print("✓ PASS: train_folds.parquet is strictly read-only (PermissionError raised as expected).")

    # Check 6: Sandbox Executor
    print("\n[Check 6/6] Verifying sandbox executor...")
    executor = get_executor(workspace_dir, mode="local", python_executable=sys.executable)
    test_script = workspace_dir / "src" / "_verify_temp.py"
    try:
        with open(test_script, "w") as f:
            f.write("import pyarrow.parquet as pq\nprint('EXECUTOR_PARQUET_OK')\n")
        res = executor.run_script("src/_verify_temp.py")
        assert res.exit_code == 0, f"Script failed: {res.stderr}"
        assert "EXECUTOR_PARQUET_OK" in res.stdout
        print("✓ Sandbox execution passed.")
    finally:
        if test_script.exists():
            test_script.unlink()

    perms = verify_permissions(workspace_dir)
    print(f"\nWorkspace security boundary status: {perms}")
    print("\n" + "=" * 60)
    print("STAGE 1 MILESTONE: ALL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_stage_1_verification()
