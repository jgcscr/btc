from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from data.ingestors.binance_us_spot import ingest_binance_us_spot


def _parse_ts(value: str | None) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _chunk_delta(interval: str, chunk_size: int) -> timedelta:
    suffix = interval[-1]
    amount = int(interval[:-1])
    if suffix == "m":
        return timedelta(minutes=amount * chunk_size)
    if suffix == "h":
        return timedelta(hours=amount * chunk_size)
    if suffix == "d":
        return timedelta(days=amount * chunk_size)
    raise ValueError(f"Unsupported interval: {interval}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Binance US spot history in chunked tidy parquet snapshots.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h", choices=["15m", "1h", "4h", "1d"])
    parser.add_argument("--days", type=int, default=365, help="Historical lookback window in days.")
    parser.add_argument("--end", type=str, default=None, help="Optional ISO UTC end timestamp.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Candles fetched per API chunk/write.")
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/market/binanceus"))
    parser.add_argument("--symbol-id", type=str, default="BINANCEUS_SPOT_BTC_USDT")
    args = parser.parse_args()

    end_ts = _parse_ts(args.end) or datetime.now(tz=UTC)
    start_ts = end_ts - timedelta(days=max(args.days, 1))
    delta = _chunk_delta(args.interval, max(args.chunk_size, 1))

    cursor = start_ts
    chunks = 0
    while cursor < end_ts:
        chunk_end = min(cursor + delta, end_ts)
        path = ingest_binance_us_spot(
            symbol=args.symbol,
            interval=args.interval,
            start=cursor.isoformat(),
            end=chunk_end.isoformat(),
            limit=max(args.chunk_size, 1),
            output_root=args.output_root,
            symbol_id=args.symbol_id,
        )
        chunks += 1
        print(f"[{chunks}] Saved {args.interval} chunk {cursor.isoformat()} -> {chunk_end.isoformat()} to {path}")
        cursor = chunk_end

    print(
        f"Completed Binance US backfill for {args.symbol} {args.interval}: "
        f"start={start_ts.isoformat()} end={end_ts.isoformat()} chunks={chunks}"
    )


if __name__ == "__main__":
    main()