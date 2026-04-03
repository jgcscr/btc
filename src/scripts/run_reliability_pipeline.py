from __future__ import annotations

from typing import Sequence

from src.runtime.reliability_pipeline import execute_reliability_pipeline
from src.scripts import run_reliability_workflow as legacy


def main(argv: Sequence[str] | None = None):
    args = legacy.parse_args(argv)
    return execute_reliability_pipeline(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
