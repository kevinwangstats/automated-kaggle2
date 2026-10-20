"""
DevOps and State Gatekeeper Agent.

Enforces deterministic gatekeeping:
- Backs up the best working code state.
- Reverts degrading modifications when ΔCV <= ε.
- Archives execution history and metrics.
"""

from pathlib import Path
import shutil
from typing import Any, Dict

from orchestrator.state import AgentState
from workspace.evaluator.metrics import MetricRegistry


class DevOpsGatekeeper:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.src_dir = self.workspace_dir / "src"
        self.backup_dir = self.workspace_dir / "src" / ".backup_best"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_current_best(self) -> None:
        """Saves current features.py and train.py as the golden state."""
        for filename in ["features.py", "train.py"]:
            src_file = self.src_dir / filename
            if src_file.exists():
                shutil.copy2(src_file, self.backup_dir / filename)

    def revert_to_last_best(self) -> None:
        """Restores features.py and train.py from the golden state."""
        for filename in ["features.py", "train.py"]:
            backup_file = self.backup_dir / filename
            if backup_file.exists():
                shutil.copy2(backup_file, self.src_dir / filename)
        print("DevOps Gate: Reverted workspace/src to previous best state.")

    def evaluate_gate(self, state: AgentState, epsilon: float = 0.0005) -> AgentState:
        """
        Determines whether the iteration is ACCEPTED or REJECTED based on out-of-fold metric improvement.
        """
        metric_name = state["metric_name"]
        curr_score = state.get("current_cv_score")
        best_score = state["best_cv_score"]

        history_entry: Dict[str, Any] = {
            "iteration": state["iteration"],
            "hypothesis": state["current_hypothesis"].model_dump() if state["current_hypothesis"] else None,
            "cv_score": curr_score,
            "best_cv_score_before": best_score,
            "status": "FAILED",
        }

        if curr_score is None:
            # Execution failed completely
            self.revert_to_last_best()
            state["status"] = "FAILED"
            history_entry["status"] = "FAILED"
        else:
            is_better = MetricRegistry.is_improvement(
                metric_name=metric_name,
                new_score=curr_score,
                baseline_score=best_score,
                epsilon=epsilon,
            )

            if is_better:
                print(f"DevOps Gate: [ACCEPTED] CV improved from {best_score:.5f} to {curr_score:.5f} (+{abs(curr_score - best_score):.5f})")
                state["best_cv_score"] = curr_score
                state["status"] = "IMPROVED"
                history_entry["status"] = "ACCEPTED"
                self.backup_current_best()
            else:
                print(f"DevOps Gate: [REJECTED] CV did not beat baseline: {curr_score:.5f} vs {best_score:.5f}")
                self.revert_to_last_best()
                state["status"] = "REJECTED"
                history_entry["status"] = "REJECTED"

        state["history"].append(history_entry)
        return state
