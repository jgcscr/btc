from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.service import job_runner


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute a registered service job in a worker subprocess.")
    parser.add_argument("--job", required=True, help="Registered service job name.")
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Path where the worker writes run metadata as JSON.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the job.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parsed = build_arg_parser().parse_args(argv)
    forwarded_args = list(parsed.args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    result = job_runner.execute_job_in_process(parsed.job, forwarded_args)
    metadata_path = Path(parsed.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "job_name": result.job_name,
                "run_id": result.run_id,
                "returncode": result.returncode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=__import__("sys").stderr)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())