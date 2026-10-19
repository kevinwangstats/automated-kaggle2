"""
Stage 2 Milestone End-to-End Verification Suite.

Validates:
1. All metric registry calculations pass without error.
2. Diagnostic generator executes on real data in < 30 seconds.
3. diagnostics_report.json conforms to strict compact schema (< 1,500 tokens).
4. Error slice profiling extracts human-readable rules.
5. Adversarial validation runs train-vs-test discrimination.
"""

from pathlib import Path
import json
import os
import sys
import time
import numpy as np
import pyarrow.parquet as pq

repo_root = Path(__file__).parent.parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from workspace.evaluator.metrics import MetricRegistry
from workspace.evaluator.diagnostics import generate_diagnostics_report


def run_stage_2_verification():
    print("=" * 60)
    print("STAGE 2 MILESTONE VERIFICATION SUITE")
    print("=" * 60)

    workspace_dir = repo_root / "workspace"
    folds_path = workspace_dir / "data" / "train_folds.parquet"
    test_path = workspace_dir / "data" / "raw" / "test.csv"
    oof_path = workspace_dir / "artifacts" / "oof_preds.npy"
    report_path = workspace_dir / "memory" / "diagnostics_report.json"

    # Check 1: Metric Registry Calculations
    print("\n[Check 1/5] Testing metric calculations...")
    y_true_cls = np.array([0, 1, 0, 1, 1, 0])
    y_pred_cls = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    auc = MetricRegistry.evaluate("roc_auc", y_true_cls, y_pred_cls)
    logloss = MetricRegistry.evaluate("logloss", y_true_cls, y_pred_cls)
    f1 = MetricRegistry.evaluate("f1", y_true_cls, y_pred_cls)
    mcc = MetricRegistry.evaluate("mcc", y_true_cls, y_pred_cls)
    assert auc == 1.0
    assert logloss < 0.4
    assert f1 == 1.0
    assert mcc == 1.0

    y_true_reg = np.array([10.0, 20.0, 30.0])
    y_pred_reg = np.array([11.0, 19.0, 31.0])
    rmse = MetricRegistry.evaluate("rmse", y_true_reg, y_pred_reg)
    rmsle = MetricRegistry.evaluate("rmsle", y_true_reg, y_pred_reg)
    mae = MetricRegistry.evaluate("mae", y_true_reg, y_pred_reg)
    assert rmse == 1.0
    assert mae == 1.0
    assert rmsle > 0.0
    print("✓ Metric calculations verified across classification and regression.")

    # Check 2: Setup OOF predictions on Titanic
    print("\n[Check 2/5] Setting up out-of-fold predictions on real folds...")
    table = pq.read_table(folds_path)
    n_rows = table.num_rows
    y_ground_truth = table.column("Survived").to_numpy()

    # Create realistic probability predictions with noise
    np.random.seed(42)
    sample_oof = np.clip(y_ground_truth * 0.6 + np.random.uniform(0.1, 0.4, size=n_rows), 0.01, 0.99)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(oof_path, sample_oof)
    print(f"✓ Saved {n_rows} OOF predictions to {oof_path}")

    # Check 3: Diagnostic Report Execution & Timing
    print("\n[Check 3/5] Executing diagnostics generator (< 30s ceiling)...")
    start_time = time.monotonic()
    report = generate_diagnostics_report(
        folds_path=folds_path,
        oof_path=oof_path,
        test_path=test_path,
        metric_name="roc_auc",
        iteration=1,
    )
    elapsed = time.monotonic() - start_time
    print(f"✓ Diagnostics generated in {elapsed:.2f} seconds (Limit: < 30s).")
    assert elapsed < 30.0, f"Diagnostics took too long: {elapsed:.2f}s"

    # Save to memory/diagnostics_report.json
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Check 4: Token Budget Verification (< 1,500 tokens)
    print("\n[Check 4/5] Measuring token footprint...")
    raw_json_str = json.dumps(report)
    char_count = len(raw_json_str)
    estimated_tokens = char_count / 4  # Standard ~4 chars per token rule of thumb
    print(f"  Character count: {char_count}")
    print(f"  Estimated tokens: {estimated_tokens:.0f} (Budget ceiling: < 1,500 tokens)")
    assert estimated_tokens < 1500, f"Token budget exceeded! {estimated_tokens:.0f} >= 1500"
    print("✓ Report is strictly within the < 1,500 token limit.")

    # Check 5: Report Content & Statistical Primitive Sanity
    print("\n[Check 5/5] Asserting structural and statistical integrity...")
    assert "global_cv" in report and report["global_cv"] > 0.5
    assert len(report["fold_scores"]) == 5
    assert "fold_std" in report
    assert "stability_assessment" in report
    assert "residual_analysis" in report
    assert "brier_score" in report["residual_analysis"]
    assert "calibration_ece" in report["residual_analysis"]
    assert "error_slices" in report and len(report["error_slices"]) > 0
    assert "top_features" in report and len(report["top_features"]) > 0
    assert "adversarial_validation" in report
    assert "adversarial_auc" in report["adversarial_validation"]
    print(f"  CV Score: {report['global_cv']:.4f} (std: {report['fold_std']:.4f})")
    print(f"  Error Slice Rule 1: {report['error_slices'][0]['condition']}")
    print(f"  Adversarial AUC: {report['adversarial_validation']['adversarial_auc']:.4f}")
    print("✓ All structural and statistical primitives verified.")

    print("\n" + "=" * 60)
    print("STAGE 2 MILESTONE: ALL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_stage_2_verification()
