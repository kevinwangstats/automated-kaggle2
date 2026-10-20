"""
Stage 3 Milestone End-to-End Verification Suite.

Runs the Actor-Critic loop against the real Titanic dataset:
1. Tests the inner self-healing loop on synthetic runtime error.
2. Executes a complete 5-iteration autonomous modeling loop.
3. Validates Methodologist hypothesis generation, Coder code application,
   Auditor diagnostic reporting, and DevOps deterministic gatekeeping.
"""

from pathlib import Path
import json
import os
import sys

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from orchestrator.graph import ActorCriticLoop
from orchestrator.agents.coder import EmpiricalCoder


def run_stage_3_verification():
    print("=" * 60)
    print("STAGE 3 MILESTONE VERIFICATION SUITE (TITANIC DATASET)")
    print("=" * 60)

    workspace_dir = repo_root / "workspace"

    # Part 1: Test the inner self-healing loop
    print("\n[Check 1/2] Testing inner self-healing loop on injected KeyError...")
    coder = EmpiricalCoder(workspace_dir=workspace_dir)
    coder.write_initial_baseline()

    # Inject an intentional KeyError access into features.py
    features_path = workspace_dir / "src" / "features.py"
    with open(features_path) as f:
        content = f.read()
    broken_content = content.replace("out['family_size'] = ", "out['family_size'] = df['non_existent_column'] + ")
    with open(features_path, "w") as f:
        f.write(broken_content)

    # Test self-healing repair
    traceback_mock = "KeyError: 'non_existent_column'\n  File 'src/features.py', line 36"
    healed = coder.heal_traceback(traceback_mock)
    assert healed, "Self-healing failed to handle KeyError."

    with open(features_path) as f:
        repaired_content = f.read()
    assert "df.get('non_existent_column'" in repaired_content or "non_existent_column" not in repaired_content
    print("✓ Inner self-healing successfully repaired the injected KeyError.")

    # Restore clean baseline
    coder.write_initial_baseline()

    # Part 2: Run the full autonomous Actor-Critic loop on Titanic
    print("\n[Check 2/2] Running 5-iteration autonomous loop on Titanic...")
    loop = ActorCriticLoop(
        workspace_dir=workspace_dir,
        competition_slug="titanic",
        model_family="lightgbm",
        max_iterations=5,
        epsilon=0.0005,
        executor_mode="local",
        python_executable=sys.executable,
    )

    final_state = loop.run_all()

    # Assertions
    history = final_state["history"]
    print(f"\nCompleted {len(history)} total evaluations.")
    for entry in history:
        hyp = entry.get("hypothesis")
        h_id = hyp.get("hypothesis_id") if isinstance(hyp, dict) else str(hyp)
        print(f"  Iter {entry['iteration']}: {entry['status']} | CV: {entry['cv_score']} | {h_id}")

    assert len(history) >= 5, f"Expected at least 5 history records, got {len(history)}"
    assert final_state["best_cv_score"] > 0.80, f"CV score {final_state['best_cv_score']} is below expected threshold!"
    assert final_state["status"] == "COMPLETED"

    # Verify artifacts exist
    assert (workspace_dir / "artifacts" / "oof_preds.npy").exists()
    assert (workspace_dir / "artifacts" / "test_preds.npy").exists()
    assert (workspace_dir / "memory" / "diagnostics_report.json").exists()

    print(f"\nFinal Best CV Score: {final_state['best_cv_score']:.5f}")
    print("\n" + "=" * 60)
    print("STAGE 3 MILESTONE: ALL CHECKS PASSED SUCCESSFULLY ON TITANIC!")
    print("=" * 60)


if __name__ == "__main__":
    run_stage_3_verification()
