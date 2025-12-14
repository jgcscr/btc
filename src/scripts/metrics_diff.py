#!/usr/bin/env python3
"""Compare regression metrics against stored baselines with tolerances."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

DEFAULT_INDEX_FIELDS: Sequence[str] = ("window", "label", "name", "id", "strategy")

TOLERANCES: Mapping[str, float] = {
    "hit_rate": 0.02,
    "cum_ret": 0.05,
    "cum_ret_net": 0.05,
    "max_drawdown": 0.01,
    "sharpe_like": 0.05,
}

N_TRADES_PCT_TOLERANCE = 0.05
N_TRADES_ABS_TOLERANCE = 2
N_TRADES_SMALL_THRESHOLD = 50

METRIC_GROUPS: Sequence[Tuple[str, ...]] = (
    ("hit_rate",),
    ("cum_ret_net", "cum_ret"),
    ("max_drawdown",),
    ("n_trades",),
    ("sharpe_like",),
)


@dataclass
class Violation:
    key: str
    metric: str
    baseline: float
    new: float
    tolerance: float
    details: str


class MetricsFormatError(RuntimeError):
    """Raised when provided metric files cannot be parsed."""


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not math.isnan(value):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"nan", "none", "null"}:
            return None
        try:
            return float(text)
        except ValueError as exc:  # pragma: no cover - handled by caller
            raise MetricsFormatError(f"Unable to parse float from '{value}'") from exc
    return None


def _load_json(path: Path) -> List[MutableMapping[str, object]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return list(data)
    raise MetricsFormatError(f"Unsupported JSON structure in {path}")


def _load_csv(path: Path) -> List[MutableMapping[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_entries(path: Path) -> List[MutableMapping[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise MetricsFormatError(f"Unsupported file extension for {path}")


def detect_index_field(entries: Sequence[Mapping[str, object]], explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    if not entries:
        return None
    for candidate in DEFAULT_INDEX_FIELDS:
        if all(candidate in entry and entry[candidate] not in (None, "") for entry in entries):
            return candidate
    first = entries[0]
    for key, value in first.items():
        if isinstance(value, str) and all(key in entry for entry in entries):
            return key
    return None


def build_mapping(entries: Sequence[MutableMapping[str, object]], index_field: Optional[str]) -> Dict[str, MutableMapping[str, object]]:
    mapping: Dict[str, MutableMapping[str, object]] = {}
    for idx, entry in enumerate(entries):
        if index_field:
            key = entry.get(index_field)
            if key in (None, ""):
                key = f"row_{idx}"
        else:
            key = f"row_{idx}"
        mapping[str(key)] = entry
    return mapping


def _metric_name(group: Sequence[str], base_entry: Mapping[str, object], new_entry: Mapping[str, object]) -> Optional[str]:
    for name in group:
        if name in base_entry and name in new_entry:
            return name
    return None


def _within_n_trades(baseline: float, new: float) -> bool:
    diff = abs(new - baseline)
    if baseline < N_TRADES_SMALL_THRESHOLD:
        return diff <= N_TRADES_ABS_TOLERANCE
    return diff <= baseline * N_TRADES_PCT_TOLERANCE


def compare_metrics(
    baseline_map: Mapping[str, Mapping[str, object]],
    new_map: Mapping[str, Mapping[str, object]],
) -> Tuple[List[Violation], List[str], List[str]]:
    violations: List[Violation] = []
    missing = sorted(set(baseline_map) - set(new_map))
    extra = sorted(set(new_map) - set(baseline_map))

    for key in sorted(set(baseline_map) & set(new_map)):
        base_entry = baseline_map[key]
        new_entry = new_map[key]

        for group in METRIC_GROUPS:
            metric = _metric_name(group, base_entry, new_entry)
            if metric is None:
                continue

            base_value = _to_float(base_entry.get(metric))
            new_value = _to_float(new_entry.get(metric))
            if base_value is None or new_value is None:
                continue

            if metric == "n_trades":
                if not _within_n_trades(base_value, new_value):
                    violations.append(
                        Violation(
                            key=key,
                            metric=metric,
                            baseline=base_value,
                            new=new_value,
                            tolerance=max(base_value * N_TRADES_PCT_TOLERANCE, N_TRADES_ABS_TOLERANCE),
                            details="absolute change exceeds allowed band",
                        )
                    )
                continue

            tol_key = metric if metric in TOLERANCES else group[0]
            tolerance = TOLERANCES[tol_key]

            if metric == "max_drawdown":
                diff = new_value - base_value
                if diff > tolerance:
                    violations.append(
                        Violation(
                            key=key,
                            metric=metric,
                            baseline=base_value,
                            new=new_value,
                            tolerance=tolerance,
                            details="drawdown worsened beyond tolerance",
                        )
                    )
                continue

            diff = abs(new_value - base_value)
            if diff > tolerance:
                violations.append(
                    Violation(
                        key=key,
                        metric=metric,
                        baseline=base_value,
                        new=new_value,
                        tolerance=tolerance,
                        details="absolute delta exceeds tolerance",
                    )
                )

    return violations, missing, extra


def write_baseline(path: Path, entries: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(entries), handle, indent=2)
        handle.write("\n")


def _format_violations(violations: Sequence[Violation]) -> str:
    lines = ["Metric deviations beyond tolerance:"]
    header = f"{'key':<20} {'metric':<15} {'baseline':>14} {'new':>14} {'tolerance':>12}  details"
    lines.append(header)
    lines.append("-" * len(header))
    for violation in violations:
        lines.append(
            f"{violation.key:<20} {violation.metric:<15} {violation.baseline:>14.6f} {violation.new:>14.6f} {violation.tolerance:>12.6f}  {violation.details}"
        )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare regression metrics against baselines")
    parser.add_argument("--baseline", required=True, type=Path, help="Path to baseline JSON/CSV file")
    parser.add_argument("--new", required=True, type=Path, help="Path to freshly computed metrics (JSON/CSV)")
    parser.add_argument("--index-field", type=str, default=None, help="Optional key field shared across rows")
    parser.add_argument("--update", action="store_true", help="Overwrite the baseline with the new metrics and exit")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    baseline_entries = load_entries(args.baseline)
    new_entries = load_entries(args.new)

    index_field = detect_index_field(baseline_entries or new_entries, args.index_field)

    if args.update:
        write_baseline(args.baseline, new_entries)
        print(f"Baseline updated at {args.baseline}")
        return 0

    baseline_map = build_mapping(baseline_entries, index_field)
    new_map = build_mapping(new_entries, index_field)

    violations, missing, extra = compare_metrics(baseline_map, new_map)

    if missing:
        print(f"Missing entries in new metrics: {', '.join(missing)}")
    if extra:
        print(f"Unexpected entries in new metrics: {', '.join(extra)}")

    if violations:
        print(_format_violations(violations))
        return 1

    if missing or extra:
        return 1

    print("Metrics are within configured tolerances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
