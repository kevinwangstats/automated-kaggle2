"""
Execution Sandbox Runners for Model Code.

Provides:
- DockerExecutor: Containerized, air-gapped isolation with read-only volume mounts.
- LocalSubprocessExecutor: Fallback local runner using the current Python environment
  while relying on file permission lockdowns (chmod 444 / 555) for data safety.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional


@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    oom_killed: bool
    timed_out: bool


class DockerExecutor:
    def __init__(
        self,
        workspace_dir: Path,
        image_name: str = "automated-kaggle-sandbox:latest",
        timeout_seconds: int = 900,  # 15 minutes max
        memory_limit: str = "16g",
        cpu_quota: float = 4.0,
        use_gpu: bool = False,
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.image_name = image_name
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.use_gpu = use_gpu

    @staticmethod
    def is_docker_daemon_running() -> bool:
        """Checks if the docker binary exists and the daemon responds."""
        if shutil.which("docker") is None:
            return False
        try:
            res = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=3,
            )
            return res.returncode == 0
        except Exception:
            return False

    def run_script(
        self, script_relative_path: str, args: Optional[List[str]] = None
    ) -> ExecutionResult:
        """
        Executes a script inside the isolated Docker container.
        Mounting contract:
          workspace/src       -> /app/src:rw
          workspace/data      -> /app/data:ro
          workspace/evaluator -> /app/evaluator:ro
          workspace/artifacts -> /app/artifacts:rw
        """
        args = args or []
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_quota}",
            "-v",
            f"{self.workspace_dir}/src:/app/src:rw",
            "-v",
            f"{self.workspace_dir}/data:/app/data:ro",
            "-v",
            f"{self.workspace_dir}/evaluator:/app/evaluator:ro",
            "-v",
            f"{self.workspace_dir}/artifacts:/app/artifacts:rw",
            "-w",
            "/app",
            self.image_name,
            "python",
            script_relative_path,
            *args,
        ]
        if self.use_gpu:
            cmd.insert(3, "--gpus=all")

        start_time = time.monotonic()
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            duration = time.monotonic() - start_time
            return ExecutionResult(
                exit_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                duration_seconds=duration,
                oom_killed=(process.returncode == 137),
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start_time
            stdout_str = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode() if exc.stdout else "")
            stderr_str = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else "")
            return ExecutionResult(
                exit_code=-1,
                stdout=stdout_str,
                stderr=stderr_str + f"\n[TIMEOUT] Execution exceeded {self.timeout_seconds}s limit.",
                duration_seconds=duration,
                oom_killed=False,
                timed_out=True,
            )


class LocalSubprocessExecutor:
    """
    Fallback executor executing directly within the active Python environment.
    Data integrity is enforced by OS-level file permissions (0o444 / 0o555).
    """

    def __init__(
        self,
        workspace_dir: Path,
        python_executable: Optional[str] = None,
        timeout_seconds: int = 900,
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = timeout_seconds

    def run_script(
        self, script_relative_path: str, args: Optional[List[str]] = None
    ) -> ExecutionResult:
        args = args or []
        script_full_path = self.workspace_dir / script_relative_path
        if not script_full_path.exists():
            # If path is relative to workspace parent or src
            alt_path = self.workspace_dir / "src" / script_relative_path
            if alt_path.exists():
                script_full_path = alt_path

        env = os.environ.copy()
        # Ensure PYTHONPATH includes workspace root so imports work
        env["PYTHONPATH"] = f"{self.workspace_dir}:{env.get('PYTHONPATH', '')}"

        cmd = [self.python_executable, str(script_full_path), *args]
        start_time = time.monotonic()
        try:
            process = subprocess.run(
                cmd,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                env=env,
                timeout=self.timeout_seconds,
            )
            duration = time.monotonic() - start_time
            return ExecutionResult(
                exit_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                duration_seconds=duration,
                oom_killed=False,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start_time
            stdout_str = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode() if exc.stdout else "")
            stderr_str = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else "")
            return ExecutionResult(
                exit_code=-1,
                stdout=stdout_str,
                stderr=stderr_str + f"\n[TIMEOUT] Execution exceeded {self.timeout_seconds}s limit.",
                duration_seconds=duration,
                oom_killed=False,
                timed_out=True,
            )


def get_executor(
    workspace_dir: Path,
    mode: str = "auto",
    timeout_seconds: int = 900,
    python_executable: Optional[str] = None,
) -> DockerExecutor | LocalSubprocessExecutor:
    """
    Factory function for obtaining an execution sandbox.
    Modes:
      - 'auto': Uses Docker if daemon is running, otherwise falls back to LocalSubprocessExecutor.
      - 'docker': Forces DockerExecutor (fails if daemon offline).
      - 'local': Forces LocalSubprocessExecutor.
    """
    workspace_dir = Path(workspace_dir).resolve()
    if mode == "docker":
        return DockerExecutor(workspace_dir=workspace_dir, timeout_seconds=timeout_seconds)
    elif mode == "local":
        return LocalSubprocessExecutor(
            workspace_dir=workspace_dir,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )

    # auto mode
    if DockerExecutor.is_docker_daemon_running():
        return DockerExecutor(workspace_dir=workspace_dir, timeout_seconds=timeout_seconds)
    else:
        return LocalSubprocessExecutor(
            workspace_dir=workspace_dir,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
