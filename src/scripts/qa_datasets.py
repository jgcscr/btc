from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

CRITICAL_EPS = 1e-12


@dataclass
class SplitMetrics:
    split: str
    rows: int
    features: int
    nan_pct: float
    inf_pct: float
    constant_columns: List[str]
    min_ts: str | None = None
    max_ts: str | None = None


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QA checks on NPZ dataset splits (row counts, NaNs, constants, label consistency).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("artifacts/datasets"),
        help="Directory containing NPZ split artifacts (default: artifacts/datasets).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Subset of dataset files to analyze. Accepts base names or paths; '.npz' is appended when missing.",
    )
    return parser.parse_args(argv)


def _format_ts(value: np.ndarray | None) -> Tuple[str | None, str | None]:
    if value is None or value.size == 0:
        return None, None
    try:
        ts_min = np.nanmin(value)
        ts_max = np.nanmax(value)
    except (TypeError, ValueError):
        return None, None
    if isinstance(ts_min, np.datetime64):
        ts_min_str = str(ts_min.astype("datetime64[s]"))
        ts_max_str = str(ts_max.astype("datetime64[s]"))
    else:
        ts_min_str = str(ts_min)
        ts_max_str = str(ts_max)
    return ts_min_str, ts_max_str


def _constant_column_names(mask: np.ndarray, feature_names: List[str]) -> List[str]:
    indices = np.where(mask)[0].tolist()
    if not indices:
        return []
    names: List[str] = []
    for idx in indices:
        if 0 <= idx < len(feature_names):
            names.append(str(feature_names[idx]))
        else:
            names.append(f"col_{idx}")
    return names


def analyze_dataset(path: Path) -> Tuple[List[Tuple[str, SplitMetrics]], List[str]]:
    issues: List[str] = []
    splits: Dict[str, SplitMetrics] = {}
    feature_names: List[str] = []

    if not path.exists():
        return [], [f"Dataset not found: {path}"]

    with np.load(path, allow_pickle=True) as data:
        if "feature_names" in data:
            raw_feature_names = data["feature_names"]
            if isinstance(raw_feature_names, np.ndarray):
                feature_names = [str(name) for name in raw_feature_names.tolist()]

        row_counts: Dict[str, int] = {}
        feature_counts: Dict[str, int] = {}
        nan_pct: Dict[str, float] = {}
        inf_pct: Dict[str, float] = {}
        const_cols: Dict[str, List[str]] = {}
        ts_arrays: Dict[str, np.ndarray] = {}

        def register_split(split: str, rows: int, cols: int) -> None:
            row_counts[split] = rows
            feature_counts[split] = cols

        # Track mismatches for non-feature arrays
        label_arrays: Dict[Tuple[str, str], np.ndarray] = {}

        for key in data.files:
            arr = data[key]
            if not isinstance(arr, np.ndarray):
                continue

            split = None
            base = None
            for suffix in ("_train", "_val", "_test"):
                if key.endswith(suffix):
                    base = key[: -len(suffix)]
                    split = suffix[1:]
                    break

            if split is None:
                if key.startswith("ts_"):
                    ts_arrays[key[3:]] = arr
                continue

            if base == "X" and arr.ndim == 2:
                rows, cols = arr.shape
                register_split(split, rows, cols)
                try:
                    arr_float = arr.astype(np.float64)
                except (TypeError, ValueError):
                    issues.append(f"{path.name}: split {split} contains non-numeric features in X_{split}.")
                    arr_float = np.array(arr, dtype=np.float64, copy=False)

                total = arr_float.size
                if total == 0:
                    nan_pct[split] = 0.0
                    inf_pct[split] = 0.0
                    const_cols[split] = []
                else:
                    nan_count = np.count_nonzero(np.isnan(arr_float))
                    inf_count = np.count_nonzero(np.isinf(arr_float))
                    nan_pct[split] = (nan_count / total) * 100.0
                    inf_pct[split] = (inf_count / total) * 100.0
                    std = np.nanstd(arr_float, axis=0)
                    valid_counts = np.sum(~np.isnan(arr_float), axis=0)
                    const_mask = (valid_counts > 1) & np.less_equal(std, CRITICAL_EPS)
                    const_names = _constant_column_names(const_mask, feature_names)
                    const_cols[split] = const_names
                    if const_names:
                        issues.append(
                            f"{path.name}: split {split} has zero-variance features {const_names}.",
                        )
                    if nan_count > 0:
                        issues.append(f"{path.name}: split {split} contains {nan_count} NaN feature values.")
                    if inf_count > 0:
                        issues.append(f"{path.name}: split {split} contains {inf_count} Inf feature values.")

            elif base == "ts":
                ts_arrays[split] = arr
            else:
                label_arrays[(base or key, split)] = arr

        # Validate feature counts against feature_names
        if feature_names:
            for split, cols in feature_counts.items():
                if cols != len(feature_names):
                    issues.append(
                        f"{path.name}: split {split} has {cols} features but feature_names lists {len(feature_names)}.",
                    )

        # Record split metrics
        for split, rows in row_counts.items():
            cols = feature_counts.get(split, 0)
            metrics = SplitMetrics(
                split=split,
                rows=rows,
                features=cols,
                nan_pct=nan_pct.get(split, 0.0),
                inf_pct=inf_pct.get(split, 0.0),
                constant_columns=const_cols.get(split, []),
            )
            ts_min, ts_max = _format_ts(ts_arrays.get(split))
            metrics.min_ts = ts_min
            metrics.max_ts = ts_max
            splits[split] = metrics

        # Validate label arrays align with feature row counts
        for (base, split), arr in label_arrays.items():
            expected = row_counts.get(split)
            actual = arr.shape[0]
            if expected is not None and expected != actual:
                issues.append(
                    f"{path.name}: array {base}_{split} has {actual} rows, expected {expected}.",
                )
            if np.isnan(arr).any():
                issues.append(f"{path.name}: array {base}_{split} contains NaN values.")
            if np.isinf(arr).any():
                issues.append(f"{path.name}: array {base}_{split} contains Inf values.")

        # Check that all splits share the same feature dimensionality
        if len(set(feature_counts.values())) > 1:
            issues.append(f"{path.name}: mismatched feature dimensions across splits {feature_counts}.")

    ordered: List[Tuple[str, SplitMetrics]] = []
    for split in ("train", "val", "test"):
        if split in splits:
            ordered.append((split, splits[split]))
    for split in sorted(set(splits) - {"train", "val", "test"}):
        ordered.append((split, splits[split]))
    return ordered, issues


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)

    dataset_paths: List[Path] = []
    if args.datasets:
        seen: set[str] = set()
        for item in args.datasets:
            candidate = Path(item)
            if candidate.suffix != ".npz":
                candidate = candidate.with_suffix(".npz")

            options: List[Path]
            if candidate.is_absolute():
                options = [candidate]
            else:
                options = [candidate, args.datasets_dir / candidate]

            selected = None
            for option in options:
                if option.exists():
                    selected = option
                    break
            if selected is None:
                selected = options[-1]

            key = str(selected.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            dataset_paths.append(selected)
    else:
        dataset_paths = sorted(args.datasets_dir.glob("*.npz"))

    if not dataset_paths:
        print(f"No NPZ files found under {args.datasets_dir}.")
        return 0

    summary_rows: List[Tuple[str, str, SplitMetrics]] = []
    all_issues: List[str] = []

    for path in dataset_paths:
        split_metrics, issues = analyze_dataset(path)
        for split, metrics in split_metrics:
            summary_rows.append((path.name, split, metrics))
        all_issues.extend(issues)

    print("# Dataset QA Summary\n")
    print("| Dataset | Split | Rows | Features | NaN% | Inf% | ConstCols | Min TS | Max TS |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")

    for dataset, split, metrics in summary_rows:
        nan_display = f"{metrics.nan_pct:.2f}" if not math.isnan(metrics.nan_pct) else "nan"
        inf_display = f"{metrics.inf_pct:.2f}" if not math.isnan(metrics.inf_pct) else "nan"
        const_display = ", ".join(metrics.constant_columns) if metrics.constant_columns else "-"
        print(
            f"| {dataset} | {split} | "
            f"{metrics.rows} | {metrics.features} | {nan_display} | {inf_display} | {const_display} | "
            f"{metrics.min_ts or '-'} | {metrics.max_ts or '-'} |",
        )

    if all_issues:
        print("\n## Issues")
        for issue in all_issues:
            print(f"- {issue}")

    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main())
