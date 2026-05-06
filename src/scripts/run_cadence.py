from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.runtime.cadence_support import execute_cadence


def _default_python_bin(repo_root: Path) -> str:
    configured = os.environ.get("PYTHON_BIN")
    if configured:
        return configured
    candidate = repo_root / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return "python3"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local cadence workflows through the Python orchestration entrypoint.")
    parser.add_argument("cadence", choices=("daily", "weekly", "monthly", "shadow"))
    parser.add_argument("--python-bin", default=None, help="Python executable used for delegated commands.")
    parser.add_argument("--repo-root", default=".", help="Repository root for cadence execution.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    python_bin = str(args.python_bin or _default_python_bin(repo_root))
    execute_cadence(args.cadence, python_bin=python_bin, repo_root=repo_root)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())