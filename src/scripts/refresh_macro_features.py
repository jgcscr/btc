from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.macro_loader import (
    DEFAULT_MACRO_METADATA_PATH,
    DEFAULT_MACRO_OUTPUT_PATH,
    DEFAULT_MACRO_START_DATE,
    DEFAULT_REFRESH_OVERLAP_DAYS,
    build_macro_feature_frame,
    build_source_manifest,
    load_macro_features,
    resolve_incremental_start_date,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh free macro context features for BTC research and runtime use.")
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_MACRO_OUTPUT_PATH),
        help="Parquet path for the normalized macro feature frame.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=str(DEFAULT_MACRO_METADATA_PATH),
        help="JSON path for the source manifest and timing notes.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help=f"Inclusive start date for refreshes in YYYY-MM-DD format (default: incremental, else {DEFAULT_MACRO_START_DATE}).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Inclusive end date for refreshes in YYYY-MM-DD format (default: today UTC).",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore any existing macro parquet and fetch the full history window.",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=DEFAULT_REFRESH_OVERLAP_DAYS,
        help="Refresh overlap window used when appending onto an existing file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    metadata_path = Path(args.metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    existing = None
    if not args.full_refresh and output_path.exists():
        existing = load_macro_features(output_path)

    start_date = args.start_date
    if start_date is None:
        start_date = resolve_incremental_start_date(
            existing,
            default_start_date=DEFAULT_MACRO_START_DATE,
            overlap_days=args.overlap_days,
        )

    frame = build_macro_feature_frame(
        start_date=start_date,
        end_date=args.end_date,
        existing=None if args.full_refresh else existing,
    )
    frame.to_parquet(output_path, index=False)

    manifest = build_source_manifest()
    manifest["row_count"] = int(len(frame))
    manifest["ts_start"] = frame["ts"].min().isoformat() if not frame.empty else None
    manifest["ts_end"] = frame["ts"].max().isoformat() if not frame.empty else None
    manifest["refresh"] = {
        "full_refresh": bool(args.full_refresh),
        "requested_start_date": start_date,
        "requested_end_date": args.end_date,
        "output_path": str(output_path),
    }
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved macro features to {output_path}")
    print(f"Wrote macro source manifest to {metadata_path}")


if __name__ == "__main__":
    main()
