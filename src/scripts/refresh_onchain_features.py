from __future__ import annotations

import argparse
from pathlib import Path

from src.data.onchain_loader import (
    DEFAULT_ONCHAIN_METADATA_PATH,
    DEFAULT_ONCHAIN_OUTPUT_PATH,
    DEFAULT_ONCHAIN_START_DATE,
    build_onchain_feature_frame,
    build_onchain_source_manifest,
    load_onchain_cached,
    load_onchain_features,
    resolve_incremental_start_timestamp,
    write_onchain_source_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh processed on-chain features for BTC research and runtime use.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_ONCHAIN_OUTPUT_PATH))
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_ONCHAIN_METADATA_PATH))
    parser.add_argument("--start-ts", type=str, default=None)
    parser.add_argument("--end-ts", type=str, default=None)
    parser.add_argument("--cache-path", type=str, default=None, help="Optional cached CSV with raw on-chain metrics.")
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--overlap-hours", type=int, default=72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    metadata_path = Path(args.metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    existing = None
    if not args.full_refresh and output_path.exists():
        existing = load_onchain_features(output_path)

    start_ts = args.start_ts
    if start_ts is None:
        start_ts = resolve_incremental_start_timestamp(
            existing,
            default_start=DEFAULT_ONCHAIN_START_DATE,
            overlap_hours=args.overlap_hours,
        )

    raw_frame = load_onchain_cached(args.cache_path) if args.cache_path else None
    frame = build_onchain_feature_frame(
        start_ts=start_ts,
        end_ts=args.end_ts,
        existing=None if args.full_refresh else existing,
        raw_frame=raw_frame,
    )
    frame.to_parquet(output_path, index=False)

    manifest = build_onchain_source_manifest()
    manifest["row_count"] = int(len(frame))
    manifest["ts_start"] = frame["ts"].min().isoformat() if not frame.empty else None
    manifest["ts_end"] = frame["ts"].max().isoformat() if not frame.empty else None
    manifest["refresh"] = {
        "full_refresh": bool(args.full_refresh),
        "requested_start_ts": start_ts,
        "requested_end_ts": args.end_ts,
        "output_path": str(output_path),
        "cache_path": args.cache_path,
    }
    write_onchain_source_manifest(metadata_path, manifest)

    print(f"Saved on-chain features to {output_path}")
    print(f"Wrote on-chain source manifest to {metadata_path}")


if __name__ == "__main__":
    main()