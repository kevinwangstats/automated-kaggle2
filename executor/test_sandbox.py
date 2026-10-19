"""
Tests for executor isolation, execution result handling, and timeout mechanics.
"""

from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from executor.sandbox import get_executor, ExecutionResult


def test_executor_basic():
    repo_root = Path(__file__).parent.parent.resolve()
    workspace_dir = repo_root / "workspace"
    test_script = workspace_dir / "src" / "_sandbox_test_temp.py"

    try:
        # Write temporary script
        with open(test_script, "w") as f:
            f.write("import sys\nprint('HELLO_FROM_SANDBOX')\nsys.exit(0)\n")

        executor = get_executor(workspace_dir, mode="local", python_executable=sys.executable)
        result: ExecutionResult = executor.run_script("src/_sandbox_test_temp.py")

        assert result.exit_code == 0, f"Expected 0, got {result.exit_code}. Stderr: {result.stderr}"
        assert "HELLO_FROM_SANDBOX" in result.stdout
        assert not result.timed_out
        print("✓ Basic script execution test passed.")
    finally:
        if test_script.exists():
            test_script.unlink()


def test_executor_timeout():
    repo_root = Path(__file__).parent.parent.resolve()
    workspace_dir = repo_root / "workspace"
    test_script = workspace_dir / "src" / "_sandbox_timeout_temp.py"

    try:
        with open(test_script, "w") as f:
            f.write("import time\ntime.sleep(5)\n")

        # Set timeout to 1 second
        executor = get_executor(
            workspace_dir, mode="local", timeout_seconds=1, python_executable=sys.executable
        )
        result: ExecutionResult = executor.run_script("src/_sandbox_timeout_temp.py")

        assert result.timed_out, "Expected execution to time out."
        assert result.exit_code == -1
        assert "[TIMEOUT]" in result.stderr
        print("✓ Timeout enforcement test passed.")
    finally:
        if test_script.exists():
            test_script.unlink()


if __name__ == "__main__":
    test_executor_basic()
    test_executor_timeout()
    print("All executor unit tests passed successfully.")
