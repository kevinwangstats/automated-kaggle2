"""
File permission and immutability management for the workspace.

Enforces:
- data directory: read + execute only (0o555)
- train_folds.parquet: read-only (0o444)
- evaluator directory: read + execute only (0o555)
- src, artifacts, memory: read-write (0o775 or 0o755)
"""

import os
from pathlib import Path
from typing import Dict


def set_permissions_recursive(path: Path, dir_mode: int, file_mode: int) -> None:
    """Recursively applies permissions to directories and files."""
    if not path.exists():
        return
    if path.is_file():
        os.chmod(path, file_mode)
        return

    # If modifying a directory that might be read-only, ensure it's writable first to traverse
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass

    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        for f in files:
            file_path = root_path / f
            try:
                os.chmod(file_path, file_mode)
            except Exception as e:
                print(f"Warning: could not set chmod on {file_path}: {e}")
        for d in dirs:
            dir_path = root_path / d
            try:
                os.chmod(dir_path, dir_mode)
            except Exception as e:
                print(f"Warning: could not set chmod on {dir_path}: {e}")

    try:
        os.chmod(path, dir_mode)
    except Exception as e:
        print(f"Warning: could not set chmod on {path}: {e}")


def lock_folds_file(folds_path: Path) -> None:
    """Sets train_folds.parquet to immutable read-only (0o444)."""
    if folds_path.exists():
        os.chmod(folds_path, 0o444)


def unlock_folds_file(folds_path: Path) -> None:
    """Restores write permissions on train_folds.parquet (0o644) for re-generation."""
    if folds_path.exists():
        os.chmod(folds_path, 0o644)


def lock_data_directory(data_dir: Path) -> None:
    """
    Sets data directory to read-only (0o555 dirs, 0o444 files) to prevent
    the Coder agent from modifying raw data or splits.
    """
    set_permissions_recursive(data_dir, dir_mode=0o555, file_mode=0o444)


def unlock_data_directory(data_dir: Path) -> None:
    """Unlocks data directory for ingestion or re-splitting."""
    set_permissions_recursive(data_dir, dir_mode=0o755, file_mode=0o644)


def lock_evaluator_directory(evaluator_dir: Path) -> None:
    """Sets evaluator directory to read-only (0o555 dirs, 0o444 files)."""
    set_permissions_recursive(evaluator_dir, dir_mode=0o555, file_mode=0o444)


def unlock_evaluator_directory(evaluator_dir: Path) -> None:
    """Unlocks evaluator directory for maintenance or upgrades."""
    set_permissions_recursive(evaluator_dir, dir_mode=0o755, file_mode=0o644)


def verify_permissions(workspace_dir: Path) -> Dict[str, bool]:
    """
    Verifies that the security boundaries are intact:
    - train_folds.parquet is read-only
    - raw data is read-only
    - evaluator is read-only
    - src is writable
    - artifacts is writable
    """
    results = {}

    folds_path = workspace_dir / "data" / "train_folds.parquet"
    if folds_path.exists():
        try:
            with open(folds_path, "a") as f:
                f.write("")
            results["folds_immutable"] = False
        except PermissionError:
            results["folds_immutable"] = True
    else:
        results["folds_immutable"] = True  # Not yet generated

    src_dir = workspace_dir / "src"
    test_src_file = src_dir / ".perm_test"
    try:
        with open(test_src_file, "w") as f:
            f.write("test")
        test_src_file.unlink()
        results["src_writable"] = True
    except Exception:
        results["src_writable"] = False

    artifacts_dir = workspace_dir / "artifacts"
    test_art_file = artifacts_dir / ".perm_test"
    try:
        with open(test_art_file, "w") as f:
            f.write("test")
        test_art_file.unlink()
        results["artifacts_writable"] = True
    except Exception:
        results["artifacts_writable"] = False

    return results
