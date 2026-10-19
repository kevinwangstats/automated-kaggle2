"""
Initializes and standardizes the workspace directory structure for a competition.
"""

from pathlib import Path
import argparse
import sys


def setup_workspace(workspace_dir: Path) -> None:
    """Creates the standard directory tree required by the agent architecture."""
    workspace_dir = workspace_dir.resolve()
    dirs = [
        workspace_dir / "data" / "raw",
        workspace_dir / "src",
        workspace_dir / "evaluator",
        workspace_dir / "artifacts" / "registry",
        workspace_dir / "memory",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists() and not list(d.glob("*")):
            gitkeep.touch()

    # Create empty __init__.py files
    for init_path in [
        workspace_dir / "__init__.py",
        workspace_dir / "src" / "__init__.py",
        workspace_dir / "evaluator" / "__init__.py",
    ]:
        if not init_path.exists():
            init_path.touch()

    print(f"Workspace skeleton successfully initialized at: {workspace_dir}")


def main():
    parser = argparse.ArgumentParser(description="Initialize workspace directory skeleton.")
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="workspace",
        help="Path to the workspace root directory (default: 'workspace')",
    )
    args = parser.parse_args()
    setup_workspace(Path(args.workspace_dir))


if __name__ == "__main__":
    main()
