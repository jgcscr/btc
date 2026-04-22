from __future__ import annotations

import argparse

from src.runtime.reliability_registry import ReliabilityRunRegistry


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve the latest trustworthy reliability run id.")
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Artifacts root containing reliability outputs and registry metadata.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_id = ReliabilityRunRegistry(args.artifacts_root).resolve_latest_trustworthy_run_id()
    if not run_id:
        return 1
    print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())