from __future__ import annotations

from typing import Sequence

from src.runtime.models import RuntimeMode
from src.runtime.refresh_pipeline import execute_refresh_pipeline
from src.scripts import run_refresh_and_predict as legacy


def main(argv: Sequence[str] | None = None):
    args = legacy.parse_args(argv)
    if getattr(args, "config", None):
        print(f"Loaded CLI defaults from config: {args.config}")
    return execute_refresh_pipeline(args, mode=RuntimeMode.RESEARCH)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
