"""
Actor-Critic State Machine & Graph Orchestrator.

Implements the cyclic execution graph:
Methodologist -> Coder -> Sandbox Execution (with inner self-healing) -> Auditor -> DevOps Gate -> Budget Check
"""

from pathlib import Path
import json
import os
import sys
from typing import Any, Dict, Optional

from orchestrator.state import AgentState, HypothesisSpec
from orchestrator.agents.coder import EmpiricalCoder
from orchestrator.agents.methodologist import Methodologist
from orchestrator.agents.devops import DevOpsGatekeeper
from executor.sandbox import get_executor, ExecutionResult
from workspace.evaluator.diagnostics import generate_diagnostics_report
from workspace.evaluator.metrics import MetricRegistry


class ActorCriticLoop:
    def __init__(
        self,
        workspace_dir: Path,
        competition_slug: str = "titanic",
        model_family: str = "lightgbm",
        max_iterations: int = 5,
        epsilon: float = 0.0005,
        executor_mode: str = "auto",
        python_executable: Optional[str] = None,
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.competition_slug = competition_slug
        self.model_family = model_family
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.executor_mode = executor_mode
        self.python_executable = python_executable or sys.executable

        self.coder = EmpiricalCoder(workspace_dir=self.workspace_dir)
        self.methodologist = Methodologist(workspace_dir=self.workspace_dir)
        self.devops = DevOpsGatekeeper(workspace_dir=self.workspace_dir)
        self.executor = get_executor(
            workspace_dir=self.workspace_dir,
            mode=self.executor_mode,
            python_executable=self.python_executable,
        )

        # Load competition metadata
        meta_path = self.workspace_dir / "data" / "competition_meta.json"
        self.metric_name = "roc_auc"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.metric_name = meta.get("evaluation_metric", "roc_auc")

        self.greater_is_better = MetricRegistry.is_greater_better(self.metric_name)

    def initialize_state(self) -> AgentState:
        """Sets up the initial state and runs Iteration 0 (Baseline)."""
        print(f"\n[Actor-Critic] Initializing Iteration 0 (Baseline: {self.model_family})...")
        self.coder.write_initial_baseline()

        # Run baseline model
        result = self.executor.run_script("src/train.py")
        if result.exit_code != 0:
            raise RuntimeError(f"Baseline model failed to execute:\n{result.stderr}")

        # Compute baseline diagnostics
        report = generate_diagnostics_report(
            folds_path=self.workspace_dir / "data" / "train_folds.parquet",
            oof_path=self.workspace_dir / "artifacts" / "oof_preds.npy",
            test_path=self.workspace_dir / "data" / "raw" / "test.csv",
            metric_name=self.metric_name,
            iteration=0,
        )
        report_path = self.workspace_dir / "memory" / "diagnostics_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        baseline_score = float(report["global_cv"])
        print(f"[Actor-Critic] Baseline CV ({self.metric_name}): {baseline_score:.5f} (std: {report['fold_std']:.5f})")

        # Golden backup of baseline
        self.devops.backup_current_best()

        initial_state: AgentState = {
            "competition_slug": self.competition_slug,
            "iteration": 1,
            "max_iterations": self.max_iterations,
            "active_branch": f"branch/{self.model_family}",
            "model_family": self.model_family,
            "best_cv_score": baseline_score,
            "current_cv_score": baseline_score,
            "metric_name": self.metric_name,
            "greater_is_better": self.greater_is_better,
            "current_hypothesis": None,
            "last_diagnostics": report,
            "execution_error_log": None,
            "retry_count": 0,
            "max_retries": 3,
            "status": "RUNNING",
            "history": [
                {
                    "iteration": 0,
                    "hypothesis": "Baseline Model Template",
                    "cv_score": baseline_score,
                    "status": "ACCEPTED",
                }
            ],
        }
        return initial_state

    def step(self, state: AgentState) -> AgentState:
        """Executes a single iteration of the Actor-Critic loop."""
        iter_num = state["iteration"]
        print(f"\n{'='*55}\n[Actor-Critic] Starting Iteration {iter_num}/{self.max_iterations} (Best CV: {state['best_cv_score']:.5f})\n{'='*55}")

        # Node 1: The Methodologist (Theoretician Critic)
        hypothesis = self.methodologist.formulate_hypothesis(state)
        state["current_hypothesis"] = hypothesis
        print(f"[Methodologist] Hypothesis ({hypothesis.hypothesis_id}):")
        print(f"  Diagnosis: {hypothesis.statistical_diagnosis}")
        print(f"  Intervention: {hypothesis.proposed_intervention}")

        # Node 2: The Empirical Coder (Actor)
        self.coder.apply_hypothesis(hypothesis)

        # Node 3: Execution Sandbox with Inner Self-Healing Loop
        retry_count = 0
        execution_success = False

        while retry_count <= state["max_retries"] and not execution_success:
            print(f"[Sandbox] Executing src/train.py (Attempt {retry_count + 1}/{state['max_retries'] + 1})...")
            result: ExecutionResult = self.executor.run_script("src/train.py")

            if result.exit_code == 0:
                oof_file = self.workspace_dir / "artifacts" / "oof_preds.npy"
                if oof_file.exists():
                    execution_success = True
                    state["execution_error_log"] = None
                    print("[Sandbox] Execution passed cleanly.")
                    break

            # Inner self-healing
            retry_count += 1
            state["retry_count"] = retry_count
            state["execution_error_log"] = result.stderr
            print(f"[Self-Healing] Execution failed with exit code {result.exit_code}. Attempting automatic repair...")
            healed = self.coder.heal_traceback(result.stderr)
            if not healed:
                print("[Self-Healing] Could not auto-heal traceback.")
                break

        if not execution_success:
            print(f"[Actor-Critic] Iteration {iter_num} crashed after {retry_count} retries.")
            state["current_cv_score"] = None
            state = self.devops.evaluate_gate(state, epsilon=self.epsilon)
            state["iteration"] += 1
            return state

        # Node 4: The Diagnostic Auditor (Gatekeeper)
        report = generate_diagnostics_report(
            folds_path=self.workspace_dir / "data" / "train_folds.parquet",
            oof_path=self.workspace_dir / "artifacts" / "oof_preds.npy",
            test_path=self.workspace_dir / "data" / "raw" / "test.csv",
            metric_name=self.metric_name,
            iteration=iter_num,
        )
        report_path = self.workspace_dir / "memory" / "diagnostics_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        new_cv = float(report["global_cv"])
        state["current_cv_score"] = new_cv
        state["last_diagnostics"] = report
        print(f"[Auditor] Evaluated OOF Score: {new_cv:.5f} (Fold std: {report['fold_std']:.5f})")

        # Node 5: DevOps Gate (Deterministic Acceptance or Revert)
        state = self.devops.evaluate_gate(state, epsilon=self.epsilon)

        state["iteration"] += 1
        return state

    def run_all(self) -> AgentState:
        """Runs the complete loop until max_iterations is reached."""
        state = self.initialize_state()
        while state["iteration"] <= state["max_iterations"]:
            state = self.step(state)

        state["status"] = "COMPLETED"
        print(f"\n{'='*55}\n[Actor-Critic] Completed {self.max_iterations} Iterations. Final Best CV: {state['best_cv_score']:.5f}\n{'='*55}")
        return state
