from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def _run(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _dataset_rows(path: Path) -> int:
    with np.load(path, allow_pickle=True) as data:
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
    min_train_size: int,
    min_val_size: int,
    min_test_size: int,
) -> tuple[int, int, int, int]:
    overhead = 2 * int(gap) + 2 * int(purge) + int(embargo)
    available = n_samples - overhead
    min_train = max(10, int(min_train_size))
    min_val = max(5, int(min_val_size))
    min_test = max(5, int(min_test_size))

    if available < (min_train + min_val + min_test):
        test = max(min_test, available // 5)
        val = max(min_val, available // 4)
        train = max(min_train, available - val - test)
    else:
        test = max(min_test, min(int(test_size), max(min_test, available // 5)))
        val = max(min_val, min(int(val_size), max(min_val, available // 4)))
        train = max(min_train, min(int(train_size), available - val - test))

    while train + val + test + overhead > n_samples and train > min_train:
        train -= 5
    while train + val + test + overhead > n_samples and val > min_val:
        val -= 5
    while train + val + test + overhead > n_samples and test > min_test:
        test -= 5

    base = train + val + test + overhead
    if base > n_samples:
        raise ValueError("Insufficient samples for even minimal walkforward setup")

    max_splits = 1 + max(0, (n_samples - base) // max(test, 1))
    resolved_folds = max(1, min(int(folds), int(max_splits)))
    return resolved_folds, train, val, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare walk-forward model kinds and pick best by net return.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--y-key", type=str, default="y")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument("--purge-size", type=int, default=0)
    parser.add_argument("--embargo-size", type=int, default=0)
    parser.add_argument("--mode", choices=("expanding", "rolling"), default="expanding")
    parser.add_argument("--min-train-size", type=int, default=30)
    parser.add_argument("--min-val-size", type=int, default=20)
    parser.add_argument("--min-test-size", type=int, default=20)
    parser.add_argument("--signal-threshold", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument(
        "--rolling-guard",
        action="store_true",
        help="Require meta_stack to also beat simple alternatives on rolling windows.",
    )
    parser.add_argument(
        "--meta-margin",
        type=float,
        default=0.0,
        help="Minimum net-return margin meta_stack must keep over the best simple model under rolling guard.",
    )
    parser.add_argument(
        "--meta-min-rolling-trades",
        type=int,
        default=0,
        help="Minimum rolling-window trade count required to retain meta_stack.",
    )
    parser.add_argument(
        "--selection-policy",
        choices=("incumbent_guarded", "best_cum_ret"),
        default="incumbent_guarded",
        help=(
            "Model selection behavior: incumbent_guarded keeps XGB as incumbent unless meta_stack clears guards; "
            "best_cum_ret selects model with highest cumulative net return."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _best(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return max(rows, key=lambda r: float(r.get("cum_ret_net_total", float("-inf"))))


def _best_simple(rows: List[Dict[str, object]]) -> Dict[str, object]:
    simple = [r for r in rows if str(r.get("model_kind")) in {"xgb", "selector_simple"}]
    return _best(simple) if simple else _best(rows)


def _find_row(rows: List[Dict[str, object]], model_kind: str) -> Dict[str, object] | None:
    for row in rows:
        if str(row.get("model_kind")) == model_kind:
            return row
    return None


def _model_output_path(base_output: Path, model_kind: str, *, rolling: bool) -> Path:
    prefix = f"{base_output.stem}_rolling" if rolling else str(base_output.stem)
    return base_output.parent / f"{prefix}_{model_kind}.json"


def main() -> None:
    args = parse_args()
    n_rows = _dataset_rows(args.dataset_path)
    folds_eff, train_eff, val_eff, test_eff = _compute_feasible_walkforward(
        n_rows,
        folds=int(args.folds),
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
        gap=int(args.gap),
        purge=int(args.purge_size),
        embargo=int(args.embargo_size),
        min_train_size=int(args.min_train_size),
        min_val_size=int(args.min_val_size),
        min_test_size=int(args.min_test_size),
    )
    if (
        folds_eff != int(args.folds)
        or train_eff != int(args.train_size)
        or val_eff != int(args.val_size)
        or test_eff != int(args.test_size)
    ):
        print(
            "Adjusted walkforward windows for dataset size "
            f"(rows={n_rows}, folds={folds_eff}, train={train_eff}, val={val_eff}, test={test_eff}).",
        )
    models = ["xgb", "meta_stack", "selector_simple"]
    rows: List[Dict[str, object]] = []
    rolling_rows: List[Dict[str, object]] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model_kind in models:
        model_output = _model_output_path(args.output, model_kind, rolling=False)
        cmd = [
            sys.executable,
            "-m",
            "src.scripts.run_walkforward_validation",
            "--dataset-path",
            str(args.dataset_path),
            "--y-key",
            str(args.y_key),
            "--folds",
            str(int(folds_eff)),
            "--train-size",
            str(int(train_eff)),
            "--val-size",
            str(int(val_eff)),
            "--test-size",
            str(int(test_eff)),
            "--gap",
            str(int(args.gap)),
            "--purge-size",
            str(int(args.purge_size)),
            "--embargo-size",
            str(int(args.embargo_size)),
            "--mode",
            str(args.mode),
            "--model-kind",
            model_kind,
            "--signal-threshold",
            str(float(args.signal_threshold)),
            "--fee-bps",
            str(float(args.fee_bps)),
            "--slippage-bps",
            str(float(args.slippage_bps)),
            "--output",
            str(model_output),
        ]
        _run(cmd)
        payload = _read_json(model_output)
        rows.append(
            {
                "model_kind": model_kind,
                "auc_mean": float(payload.get("auc_mean", float("nan"))),
                "cum_ret_net_total": float(payload.get("cum_ret_net_total", float("nan"))),
                "trade_count_total": int(payload.get("trade_count_total", 0) or 0),
                "path": str(model_output),
            }
        )

    best = _best(rows)
    selected_model_kind = str(best.get("model_kind")) if str(args.selection_policy) == "best_cum_ret" else (
        "xgb" if _find_row(rows, "xgb") is not None else str(best.get("model_kind"))
    )

    rolling_summary: Dict[str, object] | None = None
    if bool(args.rolling_guard):
        for model_kind in models:
            model_output = _model_output_path(args.output, model_kind, rolling=True)
            cmd = [
                sys.executable,
                "-m",
                "src.scripts.run_walkforward_validation",
                "--dataset-path",
                str(args.dataset_path),
                "--y-key",
                str(args.y_key),
                "--folds",
                str(int(folds_eff)),
                "--train-size",
                str(int(train_eff)),
                "--val-size",
                str(int(val_eff)),
                "--test-size",
                str(int(test_eff)),
                "--gap",
                str(int(args.gap)),
                "--purge-size",
                str(int(args.purge_size)),
                "--embargo-size",
                str(int(args.embargo_size)),
                "--mode",
                "rolling",
                "--model-kind",
                model_kind,
                "--signal-threshold",
                str(float(args.signal_threshold)),
                "--fee-bps",
                str(float(args.fee_bps)),
                "--slippage-bps",
                str(float(args.slippage_bps)),
                "--output",
                str(model_output),
            ]
            _run(cmd)
            payload = _read_json(model_output)
            rolling_rows.append(
                {
                    "model_kind": model_kind,
                    "auc_mean": float(payload.get("auc_mean", float("nan"))),
                    "cum_ret_net_total": float(payload.get("cum_ret_net_total", float("nan"))),
                    "trade_count_total": int(payload.get("trade_count_total", 0) or 0),
                    "path": str(model_output),
                }
            )

        rolling_best = _best(rolling_rows)
        rolling_best_simple = _best_simple(rolling_rows)
        rolling_summary = {
            "rows": rolling_rows,
            "best_model_kind": rolling_best.get("model_kind"),
            "best_simple_model_kind": rolling_best_simple.get("model_kind"),
            "meta_guard_margin": float(args.meta_margin),
        }

        rolling_summary["meta_min_rolling_trades"] = int(args.meta_min_rolling_trades)

    if str(args.selection_policy) == "incumbent_guarded":
        # XGB remains incumbent unless meta_stack clears both expanding(OOF) and rolling checks.
        xgb_expanding = _find_row(rows, "xgb")
        meta_expanding = _find_row(rows, "meta_stack")
        if xgb_expanding is None:
            selected_model_kind = str(best.get("model_kind"))
        elif meta_expanding is not None:
            meta_beats_expanding = (
                float(meta_expanding.get("cum_ret_net_total", float("-inf")))
                >= float(xgb_expanding.get("cum_ret_net_total", float("-inf"))) + float(args.meta_margin)
                and float(meta_expanding.get("auc_mean", float("-inf")))
                >= float(xgb_expanding.get("auc_mean", float("-inf")))
            )
            meta_beats_rolling = True
            if bool(args.rolling_guard):
                xgb_rolling = _find_row(rolling_rows, "xgb")
                meta_rolling = _find_row(rolling_rows, "meta_stack")
                if xgb_rolling is None or meta_rolling is None:
                    meta_beats_rolling = False
                else:
                    meta_beats_rolling = (
                        float(meta_rolling.get("cum_ret_net_total", float("-inf")))
                        >= float(xgb_rolling.get("cum_ret_net_total", float("-inf"))) + float(args.meta_margin)
                        and float(meta_rolling.get("auc_mean", float("-inf")))
                        >= float(xgb_rolling.get("auc_mean", float("-inf")))
                        and int(meta_rolling.get("trade_count_total", 0) or 0) >= int(args.meta_min_rolling_trades)
                    )
            if meta_beats_expanding and meta_beats_rolling:
                selected_model_kind = "meta_stack"

        # Optional selector fallback if it clearly dominates XGB and meta isn't promoted.
        if selected_model_kind == "xgb":
            selector = _find_row(rows, "selector_simple")
            if selector is not None:
                selector_net = float(selector.get("cum_ret_net_total", float("-inf")))
                xgb_net = float(_find_row(rows, "xgb").get("cum_ret_net_total", float("-inf"))) if _find_row(rows, "xgb") else float("-inf")
                if selector_net > xgb_net + float(args.meta_margin):
                    selected_model_kind = "selector_simple"

    summary = {
        "rows": rows,
        "selected_model_kind": selected_model_kind,
        "selection_metric": "cum_ret_net_total",
        "selection_policy": str(args.selection_policy),
        "resolved_walkforward": {
            "dataset_rows": int(n_rows),
            "folds": int(folds_eff),
            "train_size": int(train_eff),
            "val_size": int(val_eff),
            "test_size": int(test_eff),
            "gap": int(args.gap),
            "purge_size": int(args.purge_size),
            "embargo_size": int(args.embargo_size),
            "min_train_size": int(args.min_train_size),
            "min_val_size": int(args.min_val_size),
            "min_test_size": int(args.min_test_size),
        },
    }
    if rolling_summary is not None:
        summary["rolling_guard"] = rolling_summary
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
