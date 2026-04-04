from __future__ import annotations

import argparse
from pathlib import Path

from src.data.derivatives_loader import (
    DEFAULT_DERIVATIVES_METADATA_PATH,
    DEFAULT_DERIVATIVES_OUTPUT_PATH,
    build_derivatives_feature_frame,
    build_derivatives_source_manifest,
    load_derivatives_features,
    resolve_incremental_start_timestamp,
    write_derivatives_source_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh local Binance Futures-derived hourly features.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_DERIVATIVES_OUTPUT_PATH))
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_DERIVATIVES_METADATA_PATH))
    parser.add_argument("--start-ts", type=str, default=None, help="Optional ISO UTC start timestamp.")
    parser.add_argument("--end-ts", type=str, default=None, help="Optional ISO UTC end timestamp.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output_path)
    metadata_path = Path(args.metadata_path)

    existing = load_derivatives_features(output_path) if output_path.exists() else None
    start_ts = args.start_ts or resolve_incremental_start_timestamp(existing)
    frame = build_derivatives_feature_frame(
        start_ts=start_ts,
        end_ts=args.end_ts,
        existing=existing,
    )
    if frame.empty:
        raise SystemExit("No derivatives rows were produced.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    manifest = build_derivatives_source_manifest(frame, start_ts=start_ts)
    manifest["refresh"]["output_path"] = str(output_path)
    write_derivatives_source_manifest(metadata_path, manifest)

    print(f"Wrote derivatives features: {output_path}")
    print(f"Wrote derivatives manifest: {metadata_path}")


if __name__ == "__main__":
    main()