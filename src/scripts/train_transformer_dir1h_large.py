from __future__ import annotations

import argparse
from typing import Sequence

from src.scripts import train_transformer_dir1h as base_trainer

DEFAULT_OUTPUT_DIR = "artifacts/models/transformer_dir1h_large"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return base_trainer.parse_args(
        argv,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        preset_default="large",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    base_trainer.train_transformer(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
