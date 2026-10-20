"""
CLI Runner for the Actor-Critic Autonomous Loop.
"""

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from orchestrator.graph import ActorCriticLoop


def main():
    parser = argparse.ArgumentParser(description="Run the Actor-Critic Kaggle modeling loop.")
    parser.add_argument("--competition", type=str, default="titanic", help="Competition slug")
    parser.add_argument("--workspace-dir", type=str, default="workspace", help="Workspace root directory")
    parser.add_argument("--model-family", type=str, default="lightgbm", help="Model family to run (default: lightgbm)")
    parser.add_argument("--max-iterations", type=int, default=5, help="Number of iterations to execute")
    parser.add_argument("--epsilon", type=float, default=0.0005, help="Minimum CV improvement threshold")
    parser.add_argument("--mode", type=str, default="local", help="Executor mode: 'local' or 'docker'")
    parser.add_argument("--model", type=str, default=None, help="LLM model (defaults to deepseek/deepseek-chat via DEEPSEEK_API_KEY)")
    args = parser.parse_args()

    loop = ActorCriticLoop(
        workspace_dir=Path(args.workspace_dir),
        competition_slug=args.competition,
        model_family=args.model_family,
        max_iterations=args.max_iterations,
        epsilon=args.epsilon,
        executor_mode=args.mode,
        python_executable=sys.executable,
        model_name=args.model,
    )

    final_state = loop.run_all()
    print(f"\nFinal State Summary:")
    print(f"  Competition: {final_state['competition_slug']}")
    print(f"  Metric: {final_state['metric_name']}")
    print(f"  Best CV Score: {final_state['best_cv_score']:.5f}")
    print(f"  Total Iterations: {len(final_state['history'])}")


if __name__ == "__main__":
    main()
