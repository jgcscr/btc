from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class LabelVariant:
    name: str
    labeling_scheme: str
    no_trade_abs_ret: float
    no_trade_vol_mult: float
    tb_horizon_steps: int
    tb_vol_window: int
    tb_upper_mult: float
    tb_lower_mult: float


def _run(cmd: List[str], *, dry_run: bool) -> None:
    rendered = " ".join(cmd)
    print(f"Running: {rendered}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _dataset_rows(npz_path: Path) -> int:
    data = np.load(npz_path, allow_pickle=True)
    return int(len(data["y_train"]) + len(data["y_val"]) + len(data["y_test"]))


def _compute_feasible_walkforward(
    n_samples: int,
    *,
    folds: int,
    train_size: int,
    val_size: int,
    test_size: int,
    gap: int,
    purge: int,
    embargo: int,
) -> tuple[int, int, int, int]:
    overhead = 2 * int(gap) + 2 * int(purge) + int(embargo)
    available = n_samples - overhead
    if available < 60:
        # Minimal fallback for tiny samples.
        test = max(10, available // 5)
        val = max(10, available // 4)
        train = max(20, available - val - test)
    else:
        test = max(20, min(int(test_size), max(20, available // 5)))
        val = max(20, min(int(val_size), max(20, available // 4)))
        train = max(30, min(int(train_size), available - val - test))

    # Ensure base requirement fits.
    while train + val + test + overhead > n_samples and train > 20:
        train -= 5
    while train + val + test + overhead > n_samples and val > 10:
        val -= 5
    while train + val + test + overhead > n_samples and test > 10:
        test -= 5

    base = train + val + test + overhead
    if base > n_samples:
        raise ValueError("Insufficient samples for even minimal walkforward setup")

    max_splits = 1 + max(0, (n_samples - base) // max(test, 1))
    resolved_folds = max(1, min(int(folds), int(max_splits)))
    return resolved_folds, train, val, test


def _run_walkforward_with_retry(
    *,
    dataset_path: Path,
    walkforward_output: Path,
    args: argparse.Namespace,
) -> None:
    def _build_cmd(
        folds: int,
        train_size: int,
        val_size: int,
        test_size: int,
    ) -> List[str]:
        return [
            sys.executable,
            "-m",
            "src.scripts.run_walkforward_validation",
            "--dataset-path",
            str(dataset_path),
            "--y-key",
            "y",
            "--folds",
            str(int(folds)),
            "--train-size",
            str(int(train_size)),
            "--val-size",
            str(int(val_size)),
            "--test-size",
            str(int(test_size)),
            "--gap",
            str(int(args.walkforward_gap)),
            "--purge-size",
            str(int(args.walkforward_purge)),
            "--embargo-size",
            str(int(args.walkforward_embargo)),
            "--mode",
            str(args.walkforward_mode),
            "--model-kind",
            str(args.model_kind),
            "--signal-threshold",
            str(float(args.signal_threshold)),
            "--fee-bps",
            str(float(args.fee_bps)),
            "--slippage-bps",
            str(float(args.slippage_bps)),
            "--output",
            str(walkforward_output),
        ]

    primary_cmd = _build_cmd(
        int(args.walkforward_folds),
        int(args.walkforward_train_size),
        int(args.walkforward_val_size),
        int(args.walkforward_test_size),
    )
    try:
        _run(primary_cmd, dry_run=args.dry_run)
        return
    except subprocess.CalledProcessError:
        if args.dry_run:
            raise

    n_rows = _dataset_rows(dataset_path)
    alt_folds, alt_train, alt_val, alt_test = _compute_feasible_walkforward(
        n_rows,
        folds=int(args.walkforward_folds),
        train_size=int(args.walkforward_train_size),
        val_size=int(args.walkforward_val_size),
        test_size=int(args.walkforward_test_size),
        gap=int(args.walkforward_gap),
        purge=int(args.walkforward_purge),
        embargo=int(args.walkforward_embargo),
    )
    print(
        "Retrying walkforward with feasible windows "
        f"(rows={n_rows}, folds={alt_folds}, train={alt_train}, val={alt_val}, test={alt_test}).",
    )
    retry_cmd = _build_cmd(alt_folds, alt_train, alt_val, alt_test)
    _run(retry_cmd, dry_run=False)


def _read_json(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run label ablation on 1h direction dataset and compare OOS walk-forward metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reliability/label_ablation"),
        help="Directory where variant datasets and summary JSON files are written.",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--walkforward-folds", type=int, default=4)
    parser.add_argument("--walkforward-train-size", type=int, default=1500)
    parser.add_argument("--walkforward-val-size", type=int, default=300)
    parser.add_argument("--walkforward-test-size", type=int, default=300)
    parser.add_argument("--walkforward-gap", type=int, default=24)
    parser.add_argument("--walkforward-purge", type=int, default=0)
    parser.add_argument("--walkforward-embargo", type=int, default=0)
    parser.add_argument("--walkforward-mode", choices=("expanding", "rolling"), default="expanding")
    parser.add_argument("--model-kind", choices=("xgb", "meta_stack"), default="meta_stack")
    parser.add_argument("--signal-threshold", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--feature-reliability-json", type=str, default=None)
    parser.add_argument("--feature-reliability-min-score", type=float, default=0.55)
    parser.add_argument(
        "--economics-min-trades",
        type=float,
        default=10.0,
        help="Minimum trade count required for deployable economics-first recommendation.",
    )
    parser.add_argument(
        "--economics-min-cum-ret",
        type=float,
        default=0.0,
        help="Minimum cumulative net return required for deployable economics-first recommendation.",
    )
    parser.add_argument(
        "--economics-turnover-penalty",
        type=float,
        default=0.002,
        help="Penalty weight on excess trade count over economics-min-trades in economics score.",
    )
    parser.add_argument(
        "--economics-downside-penalty",
        type=float,
        default=2.0,
        help="Penalty multiplier on negative cumulative returns in economics score.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _economics_score(
    *,
    cum_ret_net_total: float,
    trade_count_total: float,
    min_trades: float,
    turnover_penalty: float,
    downside_penalty: float,
) -> float:
    excess_turnover = max(0.0, float(trade_count_total) - float(min_trades)) / max(float(min_trades), 1.0)
    downside = max(0.0, -float(cum_ret_net_total))
    return float(cum_ret_net_total) - float(turnover_penalty) * excess_turnover - float(downside_penalty) * downside


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        LabelVariant(
            name="binary",
            labeling_scheme="binary",
            no_trade_abs_ret=0.0,
            no_trade_vol_mult=0.0,
            tb_horizon_steps=1,
            tb_vol_window=24,
            tb_upper_mult=1.0,
            tb_lower_mult=1.0,
        ),
        LabelVariant(
            name="binary_no_trade",
            labeling_scheme="binary_no_trade",
            no_trade_abs_ret=0.0004,
            no_trade_vol_mult=0.5,
            tb_horizon_steps=1,
            tb_vol_window=24,
            tb_upper_mult=1.0,
            tb_lower_mult=1.0,
        ),
        LabelVariant(
            name="triple_barrier_wide",
            labeling_scheme="triple_barrier",
            no_trade_abs_ret=0.0,
            no_trade_vol_mult=0.0,
            tb_horizon_steps=3,
            tb_vol_window=24,
            tb_upper_mult=1.5,
            tb_lower_mult=1.5,
        ),
    ]

    rows: List[Dict[str, float | str]] = []
    for variant in variants:
        variant_dir = output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        dataset_out_dir = variant_dir / "datasets"
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        build_cmd = [
            sys.executable,
            "-m",
            "src.scripts.build_training_dataset_direction",
            "--output-dir",
            str(dataset_out_dir),
            "--threshold",
            str(float(args.threshold)),
            "--labeling-scheme",
            variant.labeling_scheme,
            "--tb-horizon-steps",
            str(int(variant.tb_horizon_steps)),
            "--tb-vol-window",
            str(int(variant.tb_vol_window)),
            "--tb-upper-mult",
            str(float(variant.tb_upper_mult)),
            "--tb-lower-mult",
            str(float(variant.tb_lower_mult)),
            "--no-trade-abs-ret",
            str(float(variant.no_trade_abs_ret)),
            "--no-trade-vol-mult",
            str(float(variant.no_trade_vol_mult)),
        ]
        if args.feature_reliability_json:
            build_cmd.extend(
                [
                    "--feature-reliability-json",
                    str(args.feature_reliability_json),
                    "--feature-reliability-min-score",
                    str(float(args.feature_reliability_min_score)),
                ]
            )

        _run(build_cmd, dry_run=args.dry_run)

        dataset_path = dataset_out_dir / "btc_features_1h_direction_splits.npz"
        walkforward_output = variant_dir / "walkforward.json"
        _run_walkforward_with_retry(
            dataset_path=dataset_path,
            walkforward_output=walkforward_output,
            args=args,
        )

        if args.dry_run:
            continue

        wf = _read_json(walkforward_output)
        row: Dict[str, float | str] = {
            "variant": variant.name,
            "labeling_scheme": variant.labeling_scheme,
            "auc_mean": float(wf.get("auc_mean", float("nan"))),
            "brier_mean": float(wf.get("brier_mean", float("nan"))),
            "ece_10_mean": float(wf.get("ece_10_mean", float("nan"))),
            "cum_ret_net_mean": float(wf.get("cum_ret_net_mean", float("nan"))),
            "cum_ret_net_total": float(wf.get("cum_ret_net_total", float("nan"))),
            "trade_count_total": float(wf.get("trade_count_total", float("nan"))),
        }
        row["economics_score"] = _economics_score(
            cum_ret_net_total=float(row["cum_ret_net_total"]),
            trade_count_total=float(row["trade_count_total"]),
            min_trades=float(args.economics_min_trades),
            turnover_penalty=float(args.economics_turnover_penalty),
            downside_penalty=float(args.economics_downside_penalty),
        )
        rows.append(row)

    summary = {
        "rows": rows,
    }
    if rows:
        best_auc = max(rows, key=lambda item: float(item.get("auc_mean", float("-inf"))))
        best_net = max(rows, key=lambda item: float(item.get("cum_ret_net_total", float("-inf"))))
        deployable_rows = [
            row
            for row in rows
            if float(row.get("trade_count_total", 0.0)) >= float(args.economics_min_trades)
            and float(row.get("cum_ret_net_total", float("-inf"))) >= float(args.economics_min_cum_ret)
        ]
        best_deployable = max(deployable_rows, key=lambda item: float(item.get("economics_score", float("-inf")))) if deployable_rows else None
        best_economics_overall = max(rows, key=lambda item: float(item.get("economics_score", float("-inf"))))

        summary["best_by_auc"] = best_auc.get("variant")
        summary["best_by_cum_ret_net_total"] = best_net.get("variant")
        summary["economics_policy"] = {
            "min_trades": float(args.economics_min_trades),
            "min_cum_ret_net_total": float(args.economics_min_cum_ret),
            "turnover_penalty": float(args.economics_turnover_penalty),
            "downside_penalty": float(args.economics_downside_penalty),
            "deployable_variant_count": int(len(deployable_rows)),
        }
        summary["recommended_primary_label"] = (
            best_deployable.get("variant") if best_deployable is not None else best_economics_overall.get("variant")
        )
        summary["recommended_primary_reason"] = (
            "best_deployable_economics_score" if best_deployable is not None else "fallback_best_economics_score"
        )
        summary["best_by_economics_score"] = best_economics_overall.get("variant")
        summary["ranking_by_economics_score"] = [
            {
                "variant": str(item.get("variant")),
                "economics_score": float(item.get("economics_score", float("nan"))),
            }
            for item in sorted(rows, key=lambda item: float(item.get("economics_score", float("-inf"))), reverse=True)
        ]
        if best_auc.get("variant") != best_net.get("variant"):
            summary["recommended_secondary_filter_label"] = best_auc.get("variant")
            summary["recommended_secondary_usage"] = "abstention_or_regime_filter"

    summary_path = output_dir / "label_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved label ablation summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
