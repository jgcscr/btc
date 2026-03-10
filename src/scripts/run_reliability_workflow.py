from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class StepResult:
    name: str
    command: List[str]
    returncode: int
    log_path: Path


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _annotate_monitoring_with_regime(monitoring_path: Path, regime_payload: Dict[str, Any]) -> None:
    if not monitoring_path.exists():
        return
    current = json.loads(monitoring_path.read_text(encoding="utf-8"))
    if not isinstance(current, dict):
        return
    current["regime"] = regime_payload
    monitoring_path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Workflow config must be a mapping: {path}")
    return payload


def _join_horizons(horizons: List[int]) -> str:
    return ",".join(str(v) for v in horizons)


def _count_npz_rows(npz_path: Path) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        required = ("y_train", "y_val", "y_test")
        if not all(k in data.files for k in required):
            raise KeyError(f"Dataset {npz_path} missing keys {required}")
        return int(len(data["y_train"]) + len(data["y_val"]) + len(data["y_test"]))


def _count_npz_rows_from_x(npz_path: Path) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        required = ("X_train", "X_val", "X_test")
        if not all(k in data.files for k in required):
            raise KeyError(f"Dataset {npz_path} missing keys {required}")
        return int(len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"]))


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
        test = max(10, available // 5)
        val = max(10, available // 4)
        train = max(20, available - val - test)
    else:
        test = max(20, min(int(test_size), max(20, available // 5)))
        val = max(20, min(int(val_size), max(20, available // 4)))
        train = max(30, min(int(train_size), available - val - test))

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


def _run_step(
    name: str,
    cmd: List[str],
    log_path: Path,
    dry_run: bool,
    *,
    allowed_returncodes: Sequence[int] | None = None,
) -> StepResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(shlex.quote(part) for part in cmd)
    allowed = set(int(code) for code in (allowed_returncodes or [0]))
    if dry_run:
        log_path.write_text(f"[dry-run] {rendered}\n", encoding="utf-8")
        print(f"[dry-run] {name}: {rendered}")
        return StepResult(name=name, command=cmd, returncode=0, log_path=log_path)

    print(f"\n>>> {name}")
    print(rendered)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode not in allowed:
        raise RuntimeError(f"Step '{name}' failed (exit={process.returncode}). See {log_path}")
    if process.returncode != 0:
        print(
            f"Warning: step '{name}' returned non-zero exit {process.returncode} but is configured as allowed.",
            file=sys.stderr,
        )
    return StepResult(name=name, command=cmd, returncode=process.returncode, log_path=log_path)


def _walkforward_depth_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    folds_obj = payload.get("folds", [])
    folds: List[Dict[str, Any]] = folds_obj if isinstance(folds_obj, list) else []
    n_folds = int(len(folds))
    test_rows = int(
        np.nansum(
            np.asarray(
                [float(f.get("n_test", 0.0) or 0.0) for f in folds if isinstance(f, dict)],
                dtype=float,
            )
        )
    )
    trade_count_total = int(payload.get("trade_count_total", 0) or 0)
    return {
        "path": str(path),
        "n_folds": n_folds,
        "test_rows_total": test_rows,
        "trade_count_total": trade_count_total,
    }


def _apply_joint_tuning_to_thresholds(
    thresholds_path: Path,
    *,
    joint_tuning_payload: Dict[str, Any],
    horizon: int,
) -> bool:
    if not thresholds_path.exists():
        return False
    best_obj = joint_tuning_payload.get("best")
    if not isinstance(best_obj, dict):
        return False

    p_up_min = best_obj.get("p_up_min")
    ret_min = best_obj.get("ret_min")
    if p_up_min is None or ret_min is None:
        return False

    payload = _load_json(thresholds_path)
    horizons_obj = payload.get("horizons")
    if not isinstance(horizons_obj, dict):
        return False

    key = str(int(horizon))
    current_obj = horizons_obj.get(key)
    if not isinstance(current_obj, dict):
        return False

    updated = dict(current_obj)
    updated["p_up_min"] = float(p_up_min)
    updated["ret_min"] = float(ret_min)
    updated["joint_tuning_applied"] = True
    updated["joint_tuning_source"] = "overlap_slice"
    updated["joint_tuning_best"] = {
        "p_up_min": float(p_up_min),
        "ret_min": float(ret_min),
        "direction_threshold": float(best_obj.get("direction_threshold", 0.5) or 0.5),
        "selection_value": float(best_obj.get("selection_value", best_obj.get("cum_ret", 0.0)) or 0.0),
        "economics_score": float(best_obj.get("economics_score", 0.0) or 0.0),
    }
    horizons_obj[key] = updated
    thresholds_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def _xgb_dataset_for_horizon(horizon: int) -> str:
    return (
        "artifacts/datasets/btc_features_1h_splits.npz"
        if horizon == 1
        else "artifacts/datasets/btc_features_multi_horizon_splits.npz"
    )


def _direction_dataset_for_horizon(horizon: int) -> str:
    return (
        "artifacts/datasets/btc_features_1h_direction_splits.npz"
        if horizon == 1
        else "artifacts/datasets/btc_features_multi_horizon_splits.npz"
    )


def _load_numeric_returns(path: Path, col: str) -> np.ndarray:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {path}")
    return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)


def _paired_diff_stats(
    *,
    baseline_path: Path,
    candidate_path: Path,
    baseline_col: str,
    candidate_col: str,
) -> Dict[str, Any]:
    baseline = _load_numeric_returns(baseline_path, baseline_col)
    candidate = _load_numeric_returns(candidate_path, candidate_col)
    n = int(min(candidate.size, baseline.size))
    if n <= 0:
        return {
            "paired_rows": 0,
            "non_zero_paired_rows": 0,
            "pairwise_diff_std": float("nan"),
        }

    diff = candidate[-n:] - baseline[-n:]
    non_zero = int(np.count_nonzero(np.abs(diff) > 1e-12))
    std = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    return {
        "paired_rows": int(n),
        "non_zero_paired_rows": non_zero,
        "pairwise_diff_std": std,
    }


def _find_latest_profile_snapshot_run(
    *,
    run_root: Path,
    current_run_id: str,
    profile_id: str,
    snapshot_name: str = "live_predictions_snapshot.json",
) -> tuple[str, Path] | None:
    if not run_root.exists():
        return None

    candidates = sorted((p for p in run_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    for run_path in candidates:
        run_id = run_path.name
        if run_id == current_run_id:
            continue

        summary_dir = run_path / "summary"
        manifest_path = summary_dir / "workflow_manifest.json"
        snapshot_path = summary_dir / snapshot_name
        if not manifest_path.exists() or not snapshot_path.exists():
            continue

        try:
            manifest_payload = _load_json(manifest_path)
        except Exception:
            continue

        profile = manifest_payload.get("profile") if isinstance(manifest_payload.get("profile"), dict) else {}
        if str(profile.get("id", "")).strip() != profile_id:
            continue
        return run_id, snapshot_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full reliability workflow: CV+Optuna, calibration, thresholds, ensemble, monitoring, paper-live.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reliability_workflow.default.yaml"),
        help="Workflow YAML config.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/reliability"),
        help="Root output directory for workflow artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip CV+Optuna searches (steps 1-2).",
    )
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration step (step 3).")
    parser.add_argument("--skip-thresholds", action="store_true", help="Skip threshold search step (step 4).")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip meta-ensemble training step (step 5).")
    parser.add_argument("--skip-monitoring", action="store_true", help="Skip reliability trigger checks (step 6).")
    parser.add_argument(
        "--skip-quality-evals",
        action="store_true",
        help="Skip optional walk-forward, model-quality, hygiene, and promotion-gate checks.",
    )
    parser.add_argument(
        "--continue-on-promotion-fail",
        action="store_true",
        help=(
            "Continue workflow when promotion gate returns exit 3 (promote=false). "
            "Useful for keeping paper-live/shadow steps running while gate blocks promotion."
        ),
    )
    parser.add_argument("--skip-paper-live", action="store_true", help="Skip paper-live refresh run (step 7).")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    cv_cfg = config.get("cv", {})
    search_cfg = config.get("search", {})
    monitoring_cfg = config.get("monitoring", {})
    cadence_cfg = config.get("cadence", {})
    quality_cfg = config.get("quality", {})
    rolling_ab_cfg = quality_cfg.get("rolling_ab", {}) if isinstance(quality_cfg, dict) else {}
    tuning_cfg = quality_cfg.get("joint_threshold_tuning", {}) if isinstance(quality_cfg, dict) else {}
    calibration_cfg = quality_cfg.get("calibration_robustness", {}) if isinstance(quality_cfg, dict) else {}
    no_trade_cfg = quality_cfg.get("no_trade", {}) if isinstance(quality_cfg, dict) else {}
    leakage_cfg = quality_cfg.get("leakage_audit", {}) if isinstance(quality_cfg, dict) else {}
    cv_stress_cfg = quality_cfg.get("cv_stress_sweep", {}) if isinstance(quality_cfg, dict) else {}
    feature_rel_cfg = quality_cfg.get("feature_reliability", {}) if isinstance(quality_cfg, dict) else {}
    champ_cfg = quality_cfg.get("champion_challenger", {}) if isinstance(quality_cfg, dict) else {}
    label_ablation_cfg = quality_cfg.get("label_ablation", {}) if isinstance(quality_cfg, dict) else {}
    trade_decision_cfg = quality_cfg.get("trade_decision_model", {}) if isinstance(quality_cfg, dict) else {}
    compare_cfg = quality_cfg.get("walkforward_model_compare", {}) if isinstance(quality_cfg, dict) else {}
    canonical_cfg = quality_cfg.get("canonical_direction_dataset", {}) if isinstance(quality_cfg, dict) else {}
    reconcile_cfg = quality_cfg.get("walkforward_labeled_reconciliation", {}) if isinstance(quality_cfg, dict) else {}
    overlap_pre_tuning_cfg = quality_cfg.get("overlap_pre_tuning", {}) if isinstance(quality_cfg, dict) else {}
    regime_weakness_cfg = quality_cfg.get("regime_weakness", {}) if isinstance(quality_cfg, dict) else {}
    profile_cfg_obj = config.get("profile", {}) if isinstance(config, dict) else {}
    profile_cfg = profile_cfg_obj if isinstance(profile_cfg_obj, dict) else {}
    run_profile_id = str(profile_cfg.get("id", "default_runtime")).strip() or "default_runtime"
    run_profile_name = str(profile_cfg.get("name", run_profile_id)).strip() or run_profile_id

    horizons = [int(v) for v in search_cfg.get("horizons", [1, 4, 8, 12])]
    n_trials = int(search_cfg.get("n_trials", 25))
    timeout = search_cfg.get("timeout")

    cv_folds = int(cv_cfg.get("folds", 4))
    cv_train_size = int(cv_cfg.get("train_size", 2400))
    cv_val_size = int(cv_cfg.get("val_size", 400))
    cv_test_size = int(cv_cfg.get("test_size", 400))
    cv_gap = int(cv_cfg.get("gap", 24))
    cv_purge_size = int(cv_cfg.get("purge_size", 0))
    cv_embargo_size = int(cv_cfg.get("embargo_size", 0))
    cv_mode = str(cv_cfg.get("mode", "expanding"))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_root / timestamp
    logs_dir = run_dir / "logs"
    model_dir = run_dir / "models"
    summary_dir = run_dir / "summary"
    for directory in (logs_dir, model_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results: List[StepResult] = []
    edge_trustworthy_for_paper_live = True
    overlap_as_primary = bool(reconcile_cfg.get("use_as_primary", True))
    overlap_pruning_allows_tuning = True
    overlap_feature_reliability_path = summary_dir / "overlap_feature_reliability.json"
    overlap_model_pruning_path = summary_dir / "overlap_model_pruning.json"

    python = sys.executable

    if not args.skip_optuna:
        for horizon in horizons:
            direction_dataset = Path(_direction_dataset_for_horizon(horizon))
            optuna_cv_folds = cv_folds
            optuna_cv_train_size = cv_train_size
            optuna_cv_val_size = cv_val_size
            optuna_cv_test_size = cv_test_size
            if direction_dataset.exists() and not args.dry_run:
                try:
                    n_rows = _count_npz_rows_from_x(direction_dataset)
                    (
                        optuna_cv_folds,
                        optuna_cv_train_size,
                        optuna_cv_val_size,
                        optuna_cv_test_size,
                    ) = _compute_feasible_walkforward(
                        n_rows,
                        folds=cv_folds,
                        train_size=cv_train_size,
                        val_size=cv_val_size,
                        test_size=cv_test_size,
                        gap=cv_gap,
                        purge=cv_purge_size,
                        embargo=cv_embargo_size,
                    )
                except Exception as exc:
                    print(f"Warning: failed to auto-resize optuna CV windows: {exc}", file=sys.stderr)

            xgb_out = model_dir / f"xgb_dir{horizon}h_optuna"
            xgb_cmd = [
                python,
                "-m",
                "src.scripts.search_xgb_optuna",
                "--mode",
                "dir",
                "--horizon",
                str(horizon),
                "--dataset-path",
                _direction_dataset_for_horizon(horizon),
                "--output-dir",
                str(xgb_out),
                "--n-trials",
                str(n_trials),
                "--cv-folds",
                str(optuna_cv_folds),
                "--cv-train-size",
                str(optuna_cv_train_size),
                "--cv-val-size",
                str(optuna_cv_val_size),
                "--cv-test-size",
                str(optuna_cv_test_size),
                "--cv-gap",
                str(cv_gap),
                "--cv-purge-size",
                str(cv_purge_size),
                "--cv-embargo-size",
                str(cv_embargo_size),
                "--cv-mode",
                cv_mode,
            ]
            if timeout is not None:
                xgb_cmd.extend(["--timeout", str(timeout)])
            results.append(_run_step(f"xgb_dir_optuna_{horizon}h", xgb_cmd, logs_dir / f"xgb_dir_optuna_{horizon}h.log", args.dry_run))

            if horizon == 1:
                lstm_out = model_dir / "lstm_dir1h_optuna"
                lstm_cmd = [
                    python,
                    "-m",
                    "src.scripts.search_lstm_optuna",
                    "--horizon",
                    "1",
                    "--dataset-path",
                    _direction_dataset_for_horizon(1),
                    "--output-dir",
                    str(lstm_out),
                    "--n-trials",
                    str(n_trials),
                    "--cv-folds",
                    str(optuna_cv_folds),
                    "--cv-train-size",
                    str(optuna_cv_train_size),
                    "--cv-val-size",
                    str(optuna_cv_val_size),
                    "--cv-test-size",
                    str(optuna_cv_test_size),
                    "--cv-gap",
                    str(cv_gap),
                    "--cv-purge-size",
                    str(cv_purge_size),
                    "--cv-embargo-size",
                    str(cv_embargo_size),
                    "--cv-mode",
                    cv_mode,
                ]
                if timeout is not None:
                    lstm_cmd.extend(["--timeout", str(int(timeout))])
                results.append(_run_step("lstm_dir_optuna_1h", lstm_cmd, logs_dir / "lstm_dir_optuna_1h.log", args.dry_run))

                transformer_out = model_dir / "transformer_dir1h_optuna"
                transformer_cmd = [
                    python,
                    "-m",
                    "src.scripts.search_transformer_optuna",
                    "--horizon",
                    "1",
                    "--dataset-path",
                    _direction_dataset_for_horizon(1),
                    "--output-dir",
                    str(transformer_out),
                    "--n-trials",
                    str(n_trials),
                    "--cv-folds",
                    str(optuna_cv_folds),
                    "--cv-train-size",
                    str(optuna_cv_train_size),
                    "--cv-val-size",
                    str(optuna_cv_val_size),
                    "--cv-test-size",
                    str(optuna_cv_test_size),
                    "--cv-gap",
                    str(cv_gap),
                    "--cv-purge-size",
                    str(cv_purge_size),
                    "--cv-embargo-size",
                    str(cv_embargo_size),
                    "--cv-mode",
                    cv_mode,
                ]
                if timeout is not None:
                    transformer_cmd.extend(["--timeout", str(timeout)])
                results.append(_run_step("transformer_dir_optuna_1h", transformer_cmd, logs_dir / "transformer_dir_optuna_1h.log", args.dry_run))

    if not args.skip_calibration:
        calibr_cmd = [
            python,
            "-m",
            "src.scripts.train_platt_calibration",
            "--horizons",
            *[str(h) for h in horizons],
            "--output-path",
            str(summary_dir / "platt_calibration.json"),
            "--method",
            str(calibration_cfg.get("method", "platt")),
        ]
        results.append(_run_step("platt_calibration", calibr_cmd, logs_dir / "platt_calibration.log", args.dry_run))

    thresholds_path = summary_dir / "calibrated_thresholds.json"
    trade_decision_model_path = summary_dir / "trade_decision_model.json"
    trade_decision_deploy_ready = False
    joint_tuning_output_path = summary_dir / "joint_threshold_tuning.json"
    joint_tuning_accepted: bool | None = None
    last_deployable_thresholds_path = Path(
        str(
            tuning_cfg.get(
                "last_deployable_thresholds_path",
                "artifacts/monitoring/calibrated_thresholds_last_deployable.json",
            )
        )
    )
    use_last_deployable_on_joint_reject = bool(tuning_cfg.get("fallback_to_last_deployable_on_reject", True))
    used_last_deployable_thresholds = False
    if not args.skip_thresholds:
        threshold_objective = str(search_cfg.get("threshold_objective", "cumret_with_dd_constraint"))
        thresholds_cmd = [
            python,
            "-m",
            "src.scripts.search_ensemble_thresholds",
            "--targets",
            _join_horizons(horizons),
            "--objective",
            threshold_objective,
            "--max-dd",
            str(search_cfg.get("max_drawdown", -0.08)),
            "--min-trades",
            str(search_cfg.get("min_trades", 10)),
            "--output-dir",
            str(summary_dir / "threshold_search"),
            "--output",
            str(thresholds_path),
        ]
        if threshold_objective == "utility_stability":
            thresholds_cmd.extend(
                [
                    "--utility-dd-penalty",
                    str(float(search_cfg.get("utility_dd_penalty", 2.0))),
                    "--utility-std-penalty",
                    str(float(search_cfg.get("utility_std_penalty", 0.5))),
                    "--utility-turnover-penalty",
                    str(float(search_cfg.get("utility_turnover_penalty", 0.5))),
                ]
            )
        results.append(_run_step("threshold_search", thresholds_cmd, logs_dir / "threshold_search.log", args.dry_run))

    if not args.skip_ensemble:
        default_meta_inputs = [
            Path("artifacts/backtests/historical_1h_pup060_full_simplified/backtest_signals.csv"),
            Path("artifacts/backtests/historical_1h_pup060_full/backtest_signals.csv"),
            Path("artifacts/backtests/historical_1h_pup060_full/backtest_signals.csv"),
        ]
        missing_meta_inputs = [str(path) for path in default_meta_inputs if not path.exists()]
        if missing_meta_inputs:
            print(
                "Warning: skipping meta-ensemble training because required inputs are missing: "
                + ", ".join(missing_meta_inputs),
                file=sys.stderr,
            )
        else:
            meta_cmd = [
                python,
                "-m",
                "src.scripts.train_meta_ensemble",
                "--output-csv",
                str(summary_dir / "backtest_signals_meta_ensemble.csv"),
                "--config-path",
                str(summary_dir / "meta_ensemble_config.json"),
                "--weight-threshold",
                str(search_cfg.get("meta_weight_threshold", 0.5)),
            ]
            results.append(_run_step("meta_ensemble_train", meta_cmd, logs_dir / "meta_ensemble_train.log", args.dry_run))

    if not args.skip_quality_evals and bool(quality_cfg.get("enabled", False)):
        quality_input = Path(quality_cfg.get("quality_input") or (summary_dir / "backtest_signals_meta_ensemble.csv"))
        labeled_snapshot_csv = summary_dir / "labeled_backtest.snapshot.csv"
        labeled_meta_output = summary_dir / "labeled_backtest_meta.json"
        labeled_snapshot_meta = summary_dir / "labeled_backtest_meta.snapshot.json"
        candidate_quality_input_cfg = quality_cfg.get("candidate_quality_input")
        candidate_quality_input = (
            Path(str(candidate_quality_input_cfg))
            if candidate_quality_input_cfg
            else (summary_dir / "backtest_signals_meta_ensemble.csv")
        )
        if not args.dry_run and not candidate_quality_input.exists():
            candidate_quality_input = quality_input
        candidate_gate_input = candidate_quality_input
        pinned_labeled_csv_cfg = quality_cfg.get("pinned_labeled_csv_path")
        pinned_labeled_meta_cfg = quality_cfg.get("pinned_labeled_meta_path")
        pinned_labeled_csv = (
            Path(str(pinned_labeled_csv_cfg))
            if str(pinned_labeled_csv_cfg or "").strip()
            else None
        )
        pinned_labeled_meta = (
            Path(str(pinned_labeled_meta_cfg))
            if str(pinned_labeled_meta_cfg or "").strip()
            else None
        )
        use_pinned_labeled_csv = pinned_labeled_csv is not None
        pinned_canonical_dataset_cfg = canonical_cfg.get("pinned_dataset_path")
        pinned_canonical_meta_cfg = canonical_cfg.get("pinned_meta_path")
        pinned_canonical_dataset = (
            Path(str(pinned_canonical_dataset_cfg))
            if str(pinned_canonical_dataset_cfg or "").strip()
            else None
        )
        pinned_canonical_meta = (
            Path(str(pinned_canonical_meta_cfg))
            if str(pinned_canonical_meta_cfg or "").strip()
            else None
        )
        use_pinned_canonical_dataset = pinned_canonical_dataset is not None

        if use_pinned_canonical_dataset and not args.dry_run:
            if not pinned_canonical_dataset.exists():
                raise FileNotFoundError(
                    f"Pinned canonical direction dataset not found: {pinned_canonical_dataset}"
                )
            if pinned_canonical_meta is not None and not pinned_canonical_meta.exists():
                raise FileNotFoundError(
                    f"Pinned canonical direction meta not found: {pinned_canonical_meta}"
                )
        if use_pinned_labeled_csv and not args.dry_run:
            if not pinned_labeled_csv.exists():
                raise FileNotFoundError(f"Pinned labeled backtest CSV not found: {pinned_labeled_csv}")
            if pinned_labeled_meta is not None and not pinned_labeled_meta.exists():
                raise FileNotFoundError(f"Pinned labeled backtest meta not found: {pinned_labeled_meta}")

        # Rebuild canonical hourly and direction datasets so walk-forward and audits use the latest expanded Binance history.
        if bool(canonical_cfg.get("enabled", True)) and not use_pinned_canonical_dataset:
            label_policy = str(canonical_cfg.get("labeling_scheme", "binary"))
            if bool(canonical_cfg.get("enforce_binary_label_policy", True)) and label_policy != "binary":
                raise ValueError(
                    f"quality.canonical_direction_dataset.labeling_scheme must be binary when enforce_binary_label_policy=true; got {label_policy}",
                )
            if bool(canonical_cfg.get("rebuild_hourly", True)):
                build_hourly_cmd = [
                    python,
                    "-m",
                    "src.scripts.build_training_dataset",
                    "--output-dir",
                    str(canonical_cfg.get("output_dir", "artifacts/datasets")),
                ]
                results.append(
                    _run_step(
                        "build_canonical_hourly_dataset",
                        build_hourly_cmd,
                        logs_dir / "build_canonical_hourly_dataset.log",
                        args.dry_run,
                    )
                )

            build_direction_cmd = [
                python,
                "-m",
                "src.scripts.build_training_dataset_direction",
                "--output-dir",
                str(canonical_cfg.get("output_dir", "artifacts/datasets")),
                "--threshold",
                str(float(canonical_cfg.get("threshold", 0.0))),
                "--labeling-scheme",
                label_policy,
                "--no-trade-abs-ret",
                str(float(canonical_cfg.get("no_trade_abs_ret", 0.0))),
                "--no-trade-vol-mult",
                str(float(canonical_cfg.get("no_trade_vol_mult", 0.0))),
                "--meta-path",
                str(canonical_cfg.get("meta_path", "artifacts/datasets/btc_features_1h_direction_meta.json")),
                "--tb-horizon-steps",
                str(int(canonical_cfg.get("tb_horizon_steps", 3))),
                "--tb-vol-window",
                str(int(canonical_cfg.get("tb_vol_window", 24))),
                "--tb-upper-mult",
                str(float(canonical_cfg.get("tb_upper_mult", 1.5))),
                "--tb-lower-mult",
                str(float(canonical_cfg.get("tb_lower_mult", 1.5))),
            ]
            results.append(
                _run_step(
                    "build_canonical_direction_dataset",
                    build_direction_cmd,
                    logs_dir / "build_canonical_direction_dataset.log",
                    args.dry_run,
                )
            )
        elif use_pinned_canonical_dataset and not args.dry_run:
            print(
                f"Using pinned canonical direction dataset: {pinned_canonical_dataset}",
                file=sys.stderr,
            )

        if use_pinned_labeled_csv and not args.dry_run:
            shutil.copyfile(pinned_labeled_csv, labeled_snapshot_csv)
            if pinned_labeled_meta is not None:
                shutil.copyfile(pinned_labeled_meta, labeled_meta_output)
                shutil.copyfile(pinned_labeled_meta, labeled_snapshot_meta)
            quality_input = labeled_snapshot_csv
            if not candidate_quality_input_cfg:
                candidate_quality_input = quality_input
            elif not candidate_quality_input.exists():
                candidate_quality_input = quality_input
            candidate_gate_input = candidate_quality_input
            print(f"Using pinned labeled backtest CSV: {pinned_labeled_csv}", file=sys.stderr)
        elif bool(quality_cfg.get("build_labeled_dataset", True)):
            labeled_cmd = [
                python,
                "-m",
                "src.scripts.build_labeled_backtest_from_history",
                "--output",
                str(quality_input),
                "--meta-output",
                str(labeled_meta_output),
                "--fold-size",
                str(int(quality_cfg.get("fold_size", 12))),
                "--lookback-rows",
                str(int(quality_cfg.get("lookback_rows", 2000))),
                "--lookback-hours",
                str(int(quality_cfg.get("lookback_hours", 0))),
                "--min-rows",
                str(int(quality_cfg.get("min_labeled_rows", 200))),
            ]
            backtest_csv = quality_cfg.get("backtest_csv")
            if backtest_csv:
                labeled_cmd.extend(["--backtest-csv", str(backtest_csv)])
            if bool(quality_cfg.get("prefer_backtest", True)):
                labeled_cmd.append("--prefer-backtest")
            else:
                labeled_cmd.append("--no-prefer-backtest")
            results.append(
                _run_step(
                    "build_labeled_dataset",
                    labeled_cmd,
                    logs_dir / "build_labeled_dataset.log",
                    args.dry_run,
                )
            )

        if not args.dry_run and quality_input.exists() and quality_input != labeled_snapshot_csv:
            shutil.copyfile(quality_input, labeled_snapshot_csv)
            quality_input = labeled_snapshot_csv
            if not candidate_quality_input_cfg:
                candidate_quality_input = quality_input
            elif not candidate_quality_input.exists():
                candidate_quality_input = quality_input
            candidate_gate_input = candidate_quality_input
        if not args.dry_run and labeled_meta_output.exists() and labeled_meta_output != labeled_snapshot_meta:
            shutil.copyfile(labeled_meta_output, labeled_snapshot_meta)

        walkforward_horizon = int(quality_cfg.get("walkforward_horizon", 1))
        walkforward_dataset = Path(
            quality_cfg.get("walkforward_dataset") or _direction_dataset_for_horizon(walkforward_horizon)
        )
        walkforward_target = str(quality_cfg.get("walkforward_y_key") or "y")
        walkforward_output = summary_dir / "walkforward_validation.json"
        selected_model_kind = str(quality_cfg.get("walkforward_model_kind", "meta_stack"))
        labeled_overlap_dataset: Path | None = None
        labeled_overlap_meta: Path | None = None

        if bool(canonical_cfg.get("enabled", True)):
            canonical_output_dir = Path(str(canonical_cfg.get("output_dir", "artifacts/datasets")))
            canonical_dataset = (
                pinned_canonical_dataset
                if pinned_canonical_dataset is not None
                else (canonical_output_dir / "btc_features_1h_direction_splits.npz")
            )
            canonical_meta = (
                pinned_canonical_meta
                if pinned_canonical_dataset is not None
                else Path(str(canonical_cfg.get("meta_path", "artifacts/datasets/btc_features_1h_direction_meta.json")))
            )
            snapshot_dataset = summary_dir / "btc_features_1h_direction_splits.snapshot.npz"
            snapshot_meta = summary_dir / "btc_features_1h_direction_meta.snapshot.json"
            walkforward_dataset = snapshot_dataset

            if not args.dry_run and canonical_dataset.exists():
                shutil.copyfile(canonical_dataset, snapshot_dataset)
            if not args.dry_run and canonical_meta is not None and canonical_meta.exists():
                shutil.copyfile(canonical_meta, snapshot_meta)

            if (not args.dry_run) and walkforward_dataset.exists() and snapshot_meta.exists():
                try:
                    meta_payload = _load_json(snapshot_meta)
                    meta_rows = int(meta_payload.get("row_count", 0) or 0)
                    npz_rows = _count_npz_rows(walkforward_dataset)
                    consistency_payload = {
                        "dataset_path": str(walkforward_dataset),
                        "meta_path": str(snapshot_meta),
                        "meta_row_count": meta_rows,
                        "npz_row_count": npz_rows,
                        "rows_match": bool(meta_rows == npz_rows),
                    }
                    (summary_dir / "canonical_dataset_consistency.json").write_text(
                        json.dumps(consistency_payload, indent=2),
                        encoding="utf-8",
                    )
                    if meta_rows != npz_rows:
                        print(
                            "Warning: canonical dataset snapshot mismatch "
                            f"(meta rows={meta_rows}, npz rows={npz_rows}).",
                            file=sys.stderr,
                        )
                except Exception as exc:
                    print(f"Warning: failed canonical dataset consistency check: {exc}", file=sys.stderr)

        if bool(reconcile_cfg.get("enabled", False)) and (quality_input.exists() or args.dry_run):
            labeled_overlap_dataset = summary_dir / "btc_features_1h_direction_splits.labeled_overlap.npz"
            labeled_overlap_meta = summary_dir / "walkforward_labeled_overlap_meta.json"
            overlap_cmd = [
                python,
                "-m",
                "src.scripts.slice_direction_dataset_by_timestamps",
                "--dataset",
                str(walkforward_dataset),
                "--labeled-csv",
                str(quality_input),
                "--ts-col",
                str(reconcile_cfg.get("ts_col", "ts")),
                "--min-rows",
                str(int(reconcile_cfg.get("min_rows", 120))),
                "--output-dataset",
                str(labeled_overlap_dataset),
                "--output-meta",
                str(labeled_overlap_meta),
            ]
            results.append(
                _run_step(
                    "walkforward_labeled_overlap_dataset",
                    overlap_cmd,
                    logs_dir / "walkforward_labeled_overlap_dataset.log",
                    args.dry_run,
                )
            )

        if (
            bool(overlap_pre_tuning_cfg.get("enabled", True))
            and bool(overlap_pre_tuning_cfg.get("feature_reliability", {}).get("enabled", True))
            and labeled_overlap_dataset is not None
            and (labeled_overlap_dataset.exists() or args.dry_run)
        ):
            overlap_feature_cfg = overlap_pre_tuning_cfg.get("feature_reliability", {})
            overlap_feature_cmd = [
                python,
                "-m",
                "src.scripts.evaluate_feature_reliability",
                "--input",
                str(labeled_overlap_dataset),
                "--baseline-window",
                str(int(overlap_feature_cfg.get("baseline_window", 80))),
                "--recent-window",
                str(int(overlap_feature_cfg.get("recent_window", 40))),
                "--min-score",
                str(float(overlap_feature_cfg.get("min_score", 0.55))),
                "--max-features",
                str(int(overlap_feature_cfg.get("max_features", 0))),
                "--output",
                str(overlap_feature_reliability_path),
            ]
            results.append(
                _run_step(
                    "overlap_feature_reliability",
                    overlap_feature_cmd,
                    logs_dir / "overlap_feature_reliability.log",
                    args.dry_run,
                )
            )

        if walkforward_dataset.exists() or args.dry_run:
            if bool(compare_cfg.get("enabled", True)):
                compare_output = summary_dir / "walkforward_model_compare.json"
                full_min_train = int(compare_cfg.get("min_train_size", 30))
                full_min_val = int(compare_cfg.get("min_val_size", 20))
                full_min_test = int(compare_cfg.get("min_test_size", 20))
                compare_cmd = [
                    python,
                    "-m",
                    "src.scripts.compare_walkforward_models",
                    "--dataset-path",
                    str(walkforward_dataset),
                    "--y-key",
                    walkforward_target,
                    "--folds",
                    str(int(compare_cfg.get("folds", cv_folds))),
                    "--train-size",
                    str(int(compare_cfg.get("train_size", cv_train_size))),
                    "--val-size",
                    str(int(compare_cfg.get("val_size", cv_val_size))),
                    "--test-size",
                    str(int(compare_cfg.get("test_size", cv_test_size))),
                    "--gap",
                    str(int(compare_cfg.get("gap", cv_gap))),
                    "--purge-size",
                    str(int(compare_cfg.get("purge_size", cv_purge_size))),
                    "--embargo-size",
                    str(int(compare_cfg.get("embargo_size", cv_embargo_size))),
                    "--mode",
                    str(compare_cfg.get("mode", cv_mode)),
                    "--min-train-size",
                    str(int(full_min_train)),
                    "--min-val-size",
                    str(int(full_min_val)),
                    "--min-test-size",
                    str(int(full_min_test)),
                    "--signal-threshold",
                    str(float(compare_cfg.get("signal_threshold", quality_cfg.get("walkforward_signal_threshold", 0.5)))),
                    "--fee-bps",
                    str(float(compare_cfg.get("fee_bps", quality_cfg.get("walkforward_fee_bps", 2.0)))),
                    "--slippage-bps",
                    str(float(compare_cfg.get("slippage_bps", quality_cfg.get("walkforward_slippage_bps", 1.0)))),
                    "--rolling-guard" if bool(compare_cfg.get("rolling_guard", True)) else "",
                    "--meta-margin",
                    str(float(compare_cfg.get("meta_margin", 0.0))),
                    "--meta-min-rolling-trades",
                    str(int(compare_cfg.get("meta_min_rolling_trades", 0))),
                    "--selection-policy",
                    str(compare_cfg.get("selection_policy", "incumbent_guarded")),
                    "--output",
                    str(compare_output),
                ]
                compare_cmd = [part for part in compare_cmd if part != ""]
                results.append(
                    _run_step(
                        "walkforward_model_compare",
                        compare_cmd,
                        logs_dir / "walkforward_model_compare.log",
                        args.dry_run,
                    )
                )
                if compare_output.exists() and not args.dry_run:
                    compare_payload = _load_json(compare_output)
                    selected_model_kind = str(compare_payload.get("selected_model_kind", selected_model_kind))
                    selected_path = None
                    rows = compare_payload.get("rows", [])
                    if isinstance(rows, list):
                        for row in rows:
                            if not isinstance(row, dict):
                                continue
                            if str(row.get("model_kind")) == selected_model_kind:
                                selected_path = row.get("path")
                                break
                    if selected_path:
                        selected_file = Path(str(selected_path))
                        if selected_file.exists():
                            shutil.copyfile(selected_file, walkforward_output)

                if labeled_overlap_dataset is not None and (labeled_overlap_dataset.exists() or args.dry_run):
                    overlap_compare_output = summary_dir / "walkforward_model_compare_labeled_overlap.json"
                    overlap_compare_cfg_obj = reconcile_cfg.get("overlap_compare", {}) if isinstance(reconcile_cfg, dict) else {}
                    overlap_compare_cfg = overlap_compare_cfg_obj if isinstance(overlap_compare_cfg_obj, dict) else {}
                    overlap_folds = int(overlap_compare_cfg.get("folds", compare_cfg.get("folds", cv_folds)))
                    overlap_train_size = int(overlap_compare_cfg.get("train_size", compare_cfg.get("train_size", cv_train_size)))
                    overlap_val_size = int(overlap_compare_cfg.get("val_size", compare_cfg.get("val_size", cv_val_size)))
                    overlap_test_size = int(overlap_compare_cfg.get("test_size", compare_cfg.get("test_size", cv_test_size)))
                    overlap_gap = int(overlap_compare_cfg.get("gap", compare_cfg.get("gap", cv_gap)))
                    overlap_purge = int(overlap_compare_cfg.get("purge_size", compare_cfg.get("purge_size", cv_purge_size)))
                    overlap_embargo = int(overlap_compare_cfg.get("embargo_size", compare_cfg.get("embargo_size", cv_embargo_size)))
                    overlap_mode = str(overlap_compare_cfg.get("mode", compare_cfg.get("mode", cv_mode)))
                    overlap_min_train = int(overlap_compare_cfg.get("min_train_size", 20))
                    overlap_min_val = int(overlap_compare_cfg.get("min_val_size", 10))
                    overlap_min_test = int(overlap_compare_cfg.get("min_test_size", 10))
                    overlap_compare_cmd = [
                        python,
                        "-m",
                        "src.scripts.compare_walkforward_models",
                        "--dataset-path",
                        str(labeled_overlap_dataset),
                        "--y-key",
                        walkforward_target,
                        "--folds",
                        str(int(overlap_folds)),
                        "--train-size",
                        str(int(overlap_train_size)),
                        "--val-size",
                        str(int(overlap_val_size)),
                        "--test-size",
                        str(int(overlap_test_size)),
                        "--gap",
                        str(int(overlap_gap)),
                        "--purge-size",
                        str(int(overlap_purge)),
                        "--embargo-size",
                        str(int(overlap_embargo)),
                        "--mode",
                        str(overlap_mode),
                        "--min-train-size",
                        str(int(overlap_min_train)),
                        "--min-val-size",
                        str(int(overlap_min_val)),
                        "--min-test-size",
                        str(int(overlap_min_test)),
                        "--signal-threshold",
                        str(float(compare_cfg.get("signal_threshold", quality_cfg.get("walkforward_signal_threshold", 0.5)))),
                        "--fee-bps",
                        str(float(compare_cfg.get("fee_bps", quality_cfg.get("walkforward_fee_bps", 2.0)))),
                        "--slippage-bps",
                        str(float(compare_cfg.get("slippage_bps", quality_cfg.get("walkforward_slippage_bps", 1.0)))),
                        "--rolling-guard" if bool(compare_cfg.get("rolling_guard", True)) else "",
                        "--meta-margin",
                        str(float(compare_cfg.get("meta_margin", 0.0))),
                        "--meta-min-rolling-trades",
                        str(int(compare_cfg.get("meta_min_rolling_trades", 0))),
                        "--selection-policy",
                        str(overlap_compare_cfg.get("selection_policy", "best_cum_ret")),
                        "--output",
                        str(overlap_compare_output),
                    ]
                    overlap_compare_cmd = [part for part in overlap_compare_cmd if part != ""]
                    results.append(
                        _run_step(
                            "walkforward_model_compare_labeled_overlap",
                            overlap_compare_cmd,
                            logs_dir / "walkforward_model_compare_labeled_overlap.log",
                            args.dry_run,
                        )
                    )

                    if not args.dry_run and compare_output.exists() and overlap_compare_output.exists():
                        full_payload = _load_json(compare_output)
                        overlap_payload = _load_json(overlap_compare_output)

                        def _selected_row(payload: Dict[str, Any]) -> Dict[str, Any]:
                            selected_kind = str(payload.get("selected_model_kind", ""))
                            rows_obj = payload.get("rows", [])
                            if not isinstance(rows_obj, list):
                                return {}
                            for item in rows_obj:
                                if isinstance(item, dict) and str(item.get("model_kind", "")) == selected_kind:
                                    return item
                            return {}

                        def _row_by_model_kind(payload: Dict[str, Any], model_kind: str) -> Dict[str, Any]:
                            rows_obj = payload.get("rows", [])
                            if not isinstance(rows_obj, list):
                                return {}
                            for item in rows_obj:
                                if isinstance(item, dict) and str(item.get("model_kind", "")) == str(model_kind):
                                    return item
                            return {}

                        overlap_selection_cfg_obj = reconcile_cfg.get("overlap_model_selection", {}) if isinstance(reconcile_cfg, dict) else {}
                        overlap_selection_cfg = overlap_selection_cfg_obj if isinstance(overlap_selection_cfg_obj, dict) else {}
                        overlap_selection_enabled = bool(overlap_selection_cfg.get("enabled", False))
                        overlap_selection_primary = str(overlap_selection_cfg.get("primary_model", "xgb"))
                        overlap_selection_fallback = str(overlap_selection_cfg.get("fallback_model", "meta_stack"))
                        overlap_selection_min_ret_improvement = float(overlap_selection_cfg.get("min_ret_improvement", 0.0))
                        overlap_selection_only_when_primary_negative = bool(
                            overlap_selection_cfg.get("only_when_primary_negative", True)
                        )
                        overlap_selection_require_auc_non_worse = bool(
                            overlap_selection_cfg.get("require_fallback_auc_non_worse", False)
                        )
                        overlap_selection_min_fallback_trades = int(overlap_selection_cfg.get("min_fallback_trades", 0))

                        overlap_selection_override_payload: Dict[str, Any] | None = None
                        if overlap_selection_enabled:
                            primary_row = _row_by_model_kind(overlap_payload, overlap_selection_primary)
                            fallback_row = _row_by_model_kind(overlap_payload, overlap_selection_fallback)
                            if primary_row and fallback_row:
                                primary_ret = float(primary_row.get("cum_ret_net_total", 0.0) or 0.0)
                                fallback_ret = float(fallback_row.get("cum_ret_net_total", 0.0) or 0.0)
                                primary_auc = float(primary_row.get("auc_mean", float("nan")))
                                fallback_auc = float(fallback_row.get("auc_mean", float("nan")))
                                fallback_trades = int(fallback_row.get("trade_count_total", 0) or 0)

                                ret_improvement_ok = fallback_ret >= (primary_ret + overlap_selection_min_ret_improvement)
                                primary_negative_ok = (primary_ret < 0.0) if overlap_selection_only_when_primary_negative else True
                                auc_ok = (fallback_auc >= primary_auc) if overlap_selection_require_auc_non_worse else True
                                trades_ok = fallback_trades >= overlap_selection_min_fallback_trades
                                selected_override = bool(ret_improvement_ok and primary_negative_ok and auc_ok and trades_ok)

                                if selected_override:
                                    overlap_payload["selected_model_kind"] = overlap_selection_fallback

                                overlap_selection_override_payload = {
                                    "enabled": True,
                                    "primary_model": overlap_selection_primary,
                                    "fallback_model": overlap_selection_fallback,
                                    "selected_override": bool(selected_override),
                                    "constraints": {
                                        "min_ret_improvement": overlap_selection_min_ret_improvement,
                                        "only_when_primary_negative": overlap_selection_only_when_primary_negative,
                                        "require_fallback_auc_non_worse": overlap_selection_require_auc_non_worse,
                                        "min_fallback_trades": overlap_selection_min_fallback_trades,
                                    },
                                    "metrics": {
                                        "primary_ret": primary_ret,
                                        "fallback_ret": fallback_ret,
                                        "primary_auc": primary_auc,
                                        "fallback_auc": fallback_auc,
                                        "fallback_trade_count": fallback_trades,
                                    },
                                    "checks": {
                                        "ret_improvement_ok": bool(ret_improvement_ok),
                                        "primary_negative_ok": bool(primary_negative_ok),
                                        "auc_ok": bool(auc_ok),
                                        "trades_ok": bool(trades_ok),
                                    },
                                }
                            else:
                                overlap_selection_override_payload = {
                                    "enabled": True,
                                    "primary_model": overlap_selection_primary,
                                    "fallback_model": overlap_selection_fallback,
                                    "selected_override": False,
                                    "reason": "missing_model_rows",
                                }

                        full_row = _selected_row(full_payload)
                        overlap_row = _selected_row(overlap_payload)
                        overlap_rows_obj = overlap_payload.get("rows", [])
                        overlap_rows = overlap_rows_obj if isinstance(overlap_rows_obj, list) else []

                        overlap_pruning_cfg_obj = overlap_pre_tuning_cfg.get("model_pruning", {}) if isinstance(overlap_pre_tuning_cfg, dict) else {}
                        overlap_pruning_cfg = overlap_pruning_cfg_obj if isinstance(overlap_pruning_cfg_obj, dict) else {}
                        overlap_pruning_enabled = bool(overlap_pre_tuning_cfg.get("enabled", True)) and bool(
                            overlap_pruning_cfg.get("enabled", True)
                        )
                        overlap_min_model_cum_ret = float(overlap_pruning_cfg.get("min_cum_ret", 0.0))
                        overlap_min_model_trades = int(overlap_pruning_cfg.get("min_trade_count", 10))

                        pruned_rows: List[Dict[str, Any]] = []
                        rejected_rows: List[Dict[str, Any]] = []
                        for row in overlap_rows:
                            if not isinstance(row, dict):
                                continue
                            row_ret = float(row.get("cum_ret_net_total", float("nan")))
                            row_trades = int(row.get("trade_count_total", 0) or 0)
                            reasons: List[str] = []
                            if row_ret < overlap_min_model_cum_ret:
                                reasons.append("min_cum_ret")
                            if row_trades < overlap_min_model_trades:
                                reasons.append("min_trade_count")
                            if reasons:
                                rejected_rows.append(
                                    {
                                        "model_kind": row.get("model_kind"),
                                        "cum_ret_net_total": row_ret,
                                        "trade_count_total": row_trades,
                                        "reasons": reasons,
                                    }
                                )
                            else:
                                pruned_rows.append(row)

                        pruned_selected_model_kind = None
                        pruned_selected_row: Dict[str, Any] | None = None
                        if overlap_pruning_enabled and pruned_rows:
                            pruned_rows_sorted = sorted(
                                pruned_rows,
                                key=lambda r: (
                                    float(r.get("cum_ret_net_total", float("-inf"))),
                                    int(r.get("trade_count_total", 0) or 0),
                                    float(r.get("auc_mean", float("-inf"))),
                                ),
                                reverse=True,
                            )
                            pruned_selected_row = pruned_rows_sorted[0]
                            pruned_selected_model_kind = str(pruned_selected_row.get("model_kind", ""))
                            overlap_payload["selected_model_kind"] = pruned_selected_model_kind
                            overlap_row = pruned_selected_row
                        require_viable_for_tuning = bool(overlap_pruning_cfg.get("require_viable_model_for_tuning", True))
                        overlap_pruning_allows_tuning = (not overlap_pruning_enabled) or bool(pruned_rows) or (not require_viable_for_tuning)

                        overlap_pruning_payload = {
                            "enabled": bool(overlap_pruning_enabled),
                            "source_compare_path": str(overlap_compare_output),
                            "constraints": {
                                "min_cum_ret": overlap_min_model_cum_ret,
                                "min_trade_count": overlap_min_model_trades,
                                "require_viable_model_for_tuning": require_viable_for_tuning,
                            },
                            "selected_model_from_compare": overlap_payload.get("selected_model_kind"),
                            "pruned_selected_model": pruned_selected_model_kind,
                            "pruned_selected_row": pruned_selected_row,
                            "viable_rows": pruned_rows,
                            "rejected_rows": rejected_rows,
                            "allows_tuning": bool(overlap_pruning_allows_tuning),
                        }
                        overlap_model_pruning_path.write_text(
                            json.dumps(overlap_pruning_payload, indent=2),
                            encoding="utf-8",
                        )

                        full_ret = float(full_row.get("cum_ret_net_total", 0.0) or 0.0)
                        overlap_ret = float(overlap_row.get("cum_ret_net_total", 0.0) or 0.0)
                        full_depth = _walkforward_depth_metrics(Path(str(full_row.get("path", "")))) if full_row else {}
                        overlap_depth = _walkforward_depth_metrics(Path(str(overlap_row.get("path", "")))) if overlap_row else {}
                        same_sign = bool((full_ret == 0.0 and overlap_ret == 0.0) or (full_ret * overlap_ret > 0.0))
                        abs_gap = float(abs(full_ret - overlap_ret))
                        min_overlap = float(reconcile_cfg.get("min_overlap_cum_ret", -0.005))
                        require_non_negative_overlap = bool(reconcile_cfg.get("require_non_negative_overlap_selected", True))
                        overlap_non_negative_ok = (overlap_ret >= 0.0) if require_non_negative_overlap else True
                        max_abs_gap = float(reconcile_cfg.get("max_abs_cum_ret_gap", 0.03))
                        min_overlap_folds = int(reconcile_cfg.get("min_overlap_folds", 2))
                        min_overlap_test_rows = int(reconcile_cfg.get("min_overlap_test_rows", 40))
                        overlap_folds_ok = int(overlap_depth.get("n_folds", 0) or 0) >= min_overlap_folds
                        overlap_test_rows_ok = int(overlap_depth.get("test_rows_total", 0) or 0) >= min_overlap_test_rows
                        overlap_depth_ok = bool(overlap_folds_ok and overlap_test_rows_ok)
                        trustworthy = bool(
                            same_sign
                            and overlap_ret >= min_overlap
                            and overlap_non_negative_ok
                            and abs_gap <= max_abs_gap
                            and overlap_depth_ok
                        )
                        failed_checks = []
                        if not same_sign:
                            failed_checks.append("same_return_sign")
                        if overlap_ret < min_overlap:
                            failed_checks.append("min_overlap_cum_ret")
                        if not overlap_non_negative_ok:
                            failed_checks.append("overlap_selected_non_negative")
                        if abs_gap > max_abs_gap:
                            failed_checks.append("max_abs_cum_ret_gap")
                        if not overlap_folds_ok:
                            failed_checks.append("min_overlap_folds")
                        if not overlap_test_rows_ok:
                            failed_checks.append("min_overlap_test_rows")

                        reconciliation_payload = {
                            "full_compare_path": str(compare_output),
                            "labeled_overlap_compare_path": str(overlap_compare_output),
                            "labeled_overlap_meta_path": str(labeled_overlap_meta) if labeled_overlap_meta else None,
                            "overlap_model_pruning_path": str(overlap_model_pruning_path),
                            "overlap_model_selection_override": overlap_selection_override_payload,
                            "full_selected_model": full_payload.get("selected_model_kind"),
                            "overlap_selected_model": overlap_payload.get("selected_model_kind"),
                            "full_selected_row": full_row,
                            "overlap_selected_row": overlap_row,
                            "full_depth": full_depth,
                            "overlap_depth": overlap_depth,
                            "agreement": {
                                "same_return_sign": same_sign,
                                "abs_cum_ret_gap": abs_gap,
                                "min_overlap_cum_ret": min_overlap,
                                "require_non_negative_overlap_selected": bool(require_non_negative_overlap),
                                "overlap_non_negative_ok": bool(overlap_non_negative_ok),
                                "max_abs_cum_ret_gap": max_abs_gap,
                                "min_overlap_folds": int(min_overlap_folds),
                                "min_overlap_test_rows": int(min_overlap_test_rows),
                                "overlap_folds_ok": bool(overlap_folds_ok),
                                "overlap_test_rows_ok": bool(overlap_test_rows_ok),
                                "overlap_depth_ok": bool(overlap_depth_ok),
                                "edge_trustworthy": trustworthy,
                                "failed_checks": failed_checks,
                            },
                        }
                        (summary_dir / "walkforward_labeled_reconciliation.json").write_text(
                            json.dumps(reconciliation_payload, indent=2),
                            encoding="utf-8",
                        )
                        overlap_stability_output = summary_dir / "overlap_trust_stability.json"
                        full_selected_path = Path(str(full_row.get("path", ""))) if full_row else None
                        overlap_selected_path = Path(str(overlap_row.get("path", ""))) if overlap_row else None
                        if (
                            quality_input.exists()
                            and labeled_overlap_dataset is not None
                            and labeled_overlap_dataset.exists()
                            and full_selected_path is not None
                            and overlap_selected_path is not None
                            and full_selected_path.exists()
                            and overlap_selected_path.exists()
                        ):
                            overlap_feature_sources = [summary_dir / "backtest_signals_meta_ensemble_decision_aligned.csv"]
                            overlap_feature_sources.append(quality_input)
                            overlap_diag_cmd = [
                                python,
                                "-m",
                                "src.scripts.analyze_overlap_trust_stability",
                                "--full-walkforward",
                                str(full_selected_path),
                                "--overlap-walkforward",
                                str(overlap_selected_path),
                                "--overlap-dataset",
                                str(labeled_overlap_dataset),
                                "--labeled-csv",
                                str(quality_input),
                                "--feature-source",
                                str(overlap_feature_sources[0]),
                                "--feature-source",
                                str(overlap_feature_sources[1]),
                                "--ts-col",
                                str(reconcile_cfg.get("ts_col", "ts")),
                                "--return-col",
                                str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                "--signal-col",
                                str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                                "--output",
                                str(overlap_stability_output),
                            ]
                            results.append(
                                _run_step(
                                    "overlap_trust_stability",
                                    overlap_diag_cmd,
                                    logs_dir / "overlap_trust_stability.log",
                                    args.dry_run,
                                )
                            )
                            reconciliation_payload["overlap_trust_stability_path"] = str(overlap_stability_output)
                            (summary_dir / "walkforward_labeled_reconciliation.json").write_text(
                                json.dumps(reconciliation_payload, indent=2),
                                encoding="utf-8",
                            )
                        edge_trustworthiness_payload = {
                            "edge_trustworthy": bool(trustworthy),
                            "source": "walkforward_labeled_reconciliation",
                            "reconciliation_path": str(summary_dir / "walkforward_labeled_reconciliation.json"),
                            "enforce_for_paper_live": bool(reconcile_cfg.get("enforce_for_paper_live", True)),
                        }
                        (summary_dir / "edge_trustworthiness.json").write_text(
                            json.dumps(edge_trustworthiness_payload, indent=2),
                            encoding="utf-8",
                        )
                        if bool(reconcile_cfg.get("enforce_for_paper_live", True)) and not bool(trustworthy):
                            edge_trustworthy_for_paper_live = False

                        if overlap_as_primary:
                            selected_model_kind = str(overlap_payload.get("selected_model_kind", selected_model_kind))
                            overlap_rows = overlap_payload.get("rows", [])
                            if isinstance(overlap_rows, list):
                                for row in overlap_rows:
                                    if not isinstance(row, dict):
                                        continue
                                    if str(row.get("model_kind")) != selected_model_kind:
                                        continue
                                    selected_path = row.get("path")
                                    if selected_path:
                                        selected_file = Path(str(selected_path))
                                        if selected_file.exists():
                                            shutil.copyfile(selected_file, walkforward_output)
                                    break
                            walkforward_dataset = labeled_overlap_dataset
            else:
                walkforward_cmd = [
                    python,
                    "-m",
                    "src.scripts.run_walkforward_validation",
                    "--dataset-path",
                    str(walkforward_dataset),
                    "--y-key",
                    walkforward_target,
                    "--folds",
                    str(cv_folds),
                    "--train-size",
                    str(cv_train_size),
                    "--val-size",
                    str(cv_val_size),
                    "--test-size",
                    str(cv_test_size),
                    "--gap",
                    str(cv_gap),
                    "--purge-size",
                    str(cv_purge_size),
                    "--embargo-size",
                    str(cv_embargo_size),
                    "--mode",
                    cv_mode,
                    "--model-kind",
                    selected_model_kind,
                    "--signal-threshold",
                    str(float(quality_cfg.get("walkforward_signal_threshold", 0.5))),
                    "--fee-bps",
                    str(float(quality_cfg.get("walkforward_fee_bps", 2.0))),
                    "--slippage-bps",
                    str(float(quality_cfg.get("walkforward_slippage_bps", 1.0))),
                    "--output",
                    str(walkforward_output),
                ]
                results.append(_run_step("walkforward_validation", walkforward_cmd, logs_dir / "walkforward_validation.log", args.dry_run))

        if bool(label_ablation_cfg.get("enabled", False)):
            label_ablation_cmd = [
                python,
                "-m",
                "src.scripts.run_label_ablation",
                "--output-dir",
                str(summary_dir / "label_ablation"),
                "--threshold",
                str(float(label_ablation_cfg.get("threshold", 0.0))),
                "--walkforward-folds",
                str(int(label_ablation_cfg.get("folds", cv_folds))),
                "--walkforward-train-size",
                str(int(label_ablation_cfg.get("train_size", cv_train_size))),
                "--walkforward-val-size",
                str(int(label_ablation_cfg.get("val_size", cv_val_size))),
                "--walkforward-test-size",
                str(int(label_ablation_cfg.get("test_size", cv_test_size))),
                "--walkforward-gap",
                str(int(label_ablation_cfg.get("gap", cv_gap))),
                "--walkforward-purge",
                str(int(label_ablation_cfg.get("purge_size", cv_purge_size))),
                "--walkforward-embargo",
                str(int(label_ablation_cfg.get("embargo_size", cv_embargo_size))),
                "--walkforward-mode",
                str(label_ablation_cfg.get("mode", cv_mode)),
                "--model-kind",
                str(label_ablation_cfg.get("model_kind", selected_model_kind)),
                "--signal-threshold",
                str(float(label_ablation_cfg.get("signal_threshold", quality_cfg.get("walkforward_signal_threshold", 0.5)))),
                "--fee-bps",
                str(float(label_ablation_cfg.get("fee_bps", quality_cfg.get("walkforward_fee_bps", 2.0)))),
                "--slippage-bps",
                str(float(label_ablation_cfg.get("slippage_bps", quality_cfg.get("walkforward_slippage_bps", 1.0)))),
                "--economics-min-trades",
                str(float(label_ablation_cfg.get("economics_min_trades", quality_cfg.get("min_trade_count", 10)))),
                "--economics-min-cum-ret",
                str(float(label_ablation_cfg.get("economics_min_cum_ret", quality_cfg.get("min_net_return", 0.0)))),
                "--economics-turnover-penalty",
                str(float(label_ablation_cfg.get("economics_turnover_penalty", 0.002))),
                "--economics-downside-penalty",
                str(float(label_ablation_cfg.get("economics_downside_penalty", 2.0))),
            ]
            feature_rel_json = label_ablation_cfg.get("feature_reliability_json")
            if (
                not feature_rel_json
                and bool(overlap_pre_tuning_cfg.get("enabled", True))
                and bool(overlap_pre_tuning_cfg.get("use_overlap_feature_reliability_for_ablation", True))
                and overlap_feature_reliability_path.exists()
            ):
                feature_rel_json = overlap_feature_reliability_path
            if feature_rel_json:
                label_ablation_cmd.extend(["--feature-reliability-json", str(feature_rel_json)])
                label_ablation_cmd.extend(
                    [
                        "--feature-reliability-min-score",
                        str(
                            float(
                                label_ablation_cfg.get(
                                    "feature_reliability_min_score",
                                    overlap_pre_tuning_cfg.get("feature_reliability", {}).get("min_score", 0.55),
                                )
                            )
                        ),
                    ]
                )
            results.append(
                _run_step(
                    "label_ablation",
                    label_ablation_cmd,
                    logs_dir / "label_ablation.log",
                    args.dry_run,
                )
            )

        if bool(leakage_cfg.get("enabled", True)):
            leakage_cmd = [
                python,
                "-m",
                "src.scripts.audit_point_in_time_integrity",
                "--dataset-path",
                str(walkforward_dataset),
                "--y-key",
                str(leakage_cfg.get("y_key", walkforward_target)),
                "--leakage-corr-alert",
                str(float(leakage_cfg.get("corr_alert", 0.98))),
                "--output",
                str(summary_dir / "point_in_time_audit.json"),
            ]
            results.append(_run_step("point_in_time_audit", leakage_cmd, logs_dir / "point_in_time_audit.log", args.dry_run))

        if bool(cv_stress_cfg.get("enabled", True)):
            cv_stress_folds = int(cv_stress_cfg.get("folds", cv_folds))
            cv_stress_train = int(cv_stress_cfg.get("train_size", cv_train_size))
            cv_stress_val = int(cv_stress_cfg.get("val_size", cv_val_size))
            cv_stress_test = int(cv_stress_cfg.get("test_size", cv_test_size))
            if walkforward_dataset.exists() and not args.dry_run:
                try:
                    n_rows = _count_npz_rows_from_x(walkforward_dataset)
                    cv_stress_folds, cv_stress_train, cv_stress_val, cv_stress_test = _compute_feasible_walkforward(
                        n_rows,
                        folds=cv_stress_folds,
                        train_size=cv_stress_train,
                        val_size=cv_stress_val,
                        test_size=cv_stress_test,
                        gap=int(cv_stress_cfg.get("gap", cv_gap)),
                        purge=int(cv_purge_size),
                        embargo=int(cv_embargo_size),
                    )
                except Exception as exc:
                    print(f"Warning: failed to auto-resize cv stress windows: {exc}", file=sys.stderr)
                    cv_stress_folds = 0
            if cv_stress_folds < 2 and not args.dry_run:
                print(
                    "Skipping cv_stress_sweep: insufficient samples for at least 2 feasible folds.",
                    file=sys.stderr,
                )
            else:
                cv_stress_cmd = [
                python,
                "-m",
                "src.scripts.run_cv_stress_sweep",
                "--dataset-path",
                str(walkforward_dataset),
                "--y-key",
                str(cv_stress_cfg.get("y_key", walkforward_target)),
                "--folds",
                str(int(cv_stress_folds)),
                "--train-size",
                str(int(cv_stress_train)),
                "--val-size",
                str(int(cv_stress_val)),
                "--test-size",
                str(int(cv_stress_test)),
                "--gap",
                str(int(cv_stress_cfg.get("gap", cv_gap))),
                "--purge-list",
                str(cv_stress_cfg.get("purge_list", "0,12,24")),
                "--embargo-list",
                str(cv_stress_cfg.get("embargo_list", "0,12,24")),
                "--mode",
                str(cv_stress_cfg.get("mode", cv_mode)),
                "--output",
                str(summary_dir / "cv_stress_sweep.json"),
                ]
                results.append(_run_step("cv_stress_sweep", cv_stress_cmd, logs_dir / "cv_stress_sweep.log", args.dry_run))

        if candidate_quality_input.exists() or args.dry_run:
            quality_cmd = [
                python,
                "-m",
                "src.scripts.evaluate_model_quality",
                "--input",
                str(candidate_quality_input),
                "--output",
                str(summary_dir / "model_quality_candidate.json"),
            ]
            results.append(_run_step("model_quality_eval", quality_cmd, logs_dir / "model_quality_eval.log", args.dry_run))

            if bool(trade_decision_cfg.get("enabled", True)):
                decision_feature_input = summary_dir / "backtest_signals_meta_ensemble_decision_features.csv"
                enrich_candidate_cmd = [
                    python,
                    "-m",
                    "src.scripts.enrich_backtest_with_decision_features",
                    "--input",
                    str(candidate_quality_input),
                    "--output",
                    str(decision_feature_input),
                    "--meta-output",
                    str(summary_dir / "backtest_signals_meta_ensemble_decision_features_meta.json"),
                    "--auto-discover-sources",
                ]
                if quality_input.exists() or args.dry_run:
                    enrich_candidate_cmd.extend(["--feature-source", str(quality_input)])
                incumbent_backtest_for_features = quality_cfg.get("incumbent_backtest_csv")
                if incumbent_backtest_for_features:
                    enrich_candidate_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                results.append(
                    _run_step(
                        "enrich_candidate_decision_features",
                        enrich_candidate_cmd,
                        logs_dir / "enrich_candidate_decision_features.log",
                        args.dry_run,
                    )
                )
                decision_model_input = decision_feature_input if (decision_feature_input.exists() or args.dry_run) else candidate_quality_input

                decision_cmd = [
                    python,
                    "-m",
                    "src.scripts.train_trade_decision_model",
                    "--input",
                    str(decision_model_input),
                    "--target-col",
                    str(trade_decision_cfg.get("target_col", "ret_ensemble_net")),
                    "--signal-col",
                    str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                    "--threshold",
                    str(float(trade_decision_cfg.get("threshold", 0.55))),
                    "--min-rows",
                    str(int(trade_decision_cfg.get("min_rows", 200))),
                    "--ev-calibration-source",
                    str(trade_decision_cfg.get("ev_calibration_source", "hybrid")),
                    "--ev-min-candidate-rows",
                    str(int(trade_decision_cfg.get("ev_min_candidate_rows", 100))),
                    "--ev-bins",
                    str(int(trade_decision_cfg.get("ev_bins", 8))),
                    "--min-bin-samples",
                    str(int(trade_decision_cfg.get("min_bin_samples", 8))),
                    "--raw-ev-calibration-source",
                    str(trade_decision_cfg.get("raw_ev_calibration_source", "weighted_hybrid")),
                    "--raw-ev-candidate-weight",
                    str(float(trade_decision_cfg.get("raw_ev_candidate_weight", 4.0))),
                    "--output",
                    str(trade_decision_model_path),
                ]
                if bool(trade_decision_cfg.get("candidate_only", False)):
                    decision_cmd.append("--candidate-only")
                results.append(
                    _run_step(
                        "trade_decision_model",
                        decision_cmd,
                        logs_dir / "trade_decision_model.log",
                        args.dry_run,
                    )
                )
                if trade_decision_model_path.exists() and not args.dry_run:
                    decision_payload = _load_json(trade_decision_model_path)
                    trade_decision_deploy_ready = bool(decision_payload.get("deploy_ready", False))
                elif args.dry_run:
                    trade_decision_deploy_ready = bool(trade_decision_cfg.get("enabled", True))

                if trade_decision_deploy_ready and (candidate_quality_input.exists() or args.dry_run):
                    paper_live_cfg_path = Path(str(search_cfg.get("paper_live_config", "")))
                    trade_policy_cfg: Dict[str, Any] = {}
                    if paper_live_cfg_path.exists():
                        try:
                            paper_live_cfg = _load_yaml(paper_live_cfg_path)
                            trade_policy_obj = paper_live_cfg.get("trade_decision_policy", {})
                            if isinstance(trade_policy_obj, dict):
                                trade_policy_cfg = trade_policy_obj
                        except Exception as exc:
                            print(
                                f"Warning: failed to load trade_decision_policy from {paper_live_cfg_path}: {exc}",
                                file=sys.stderr,
                            )

                    weak_veto_cfg_obj = trade_decision_cfg.get("weak_band_candidate_only_veto", {})
                    weak_veto_cfg = weak_veto_cfg_obj if isinstance(weak_veto_cfg_obj, dict) else {}
                    weak_veto_enabled_official = bool(weak_veto_cfg.get("enabled", False))
                    weak_veto_official_mode = str(weak_veto_cfg.get("official_mode", "weak_band")).strip().lower()
                    if weak_veto_official_mode not in {"weak_band", "midband"}:
                        weak_veto_official_mode = "weak_band"
                    weak_veto_low = float(weak_veto_cfg.get("p_up_low", 0.55))
                    weak_veto_high = float(weak_veto_cfg.get("p_up_high", 0.60))
                    weak_veto_high_inclusive = bool(weak_veto_cfg.get("high_inclusive", False))
                    weak_veto_reference_path_cfg = weak_veto_cfg.get("incumbent_reference_path")
                    refined_veto_cfg_obj = weak_veto_cfg.get("refined", {})
                    refined_veto_cfg = refined_veto_cfg_obj if isinstance(refined_veto_cfg_obj, dict) else {}
                    refined_veto_low = float(refined_veto_cfg.get("p_up_low", weak_veto_low))
                    refined_veto_high = float(refined_veto_cfg.get("p_up_high", weak_veto_high))
                    refined_veto_high_inclusive = bool(refined_veto_cfg.get("high_inclusive", weak_veto_high_inclusive))
                    refined_veto_min_abs_ret_pred = float(refined_veto_cfg.get("min_abs_ret_pred", 0.001))
                    refined_veto_reference_path_cfg = refined_veto_cfg.get("incumbent_reference_path")
                    midband_veto_cfg_obj = weak_veto_cfg.get("midband", {})
                    midband_veto_cfg = midband_veto_cfg_obj if isinstance(midband_veto_cfg_obj, dict) else {}
                    midband_veto_low = float(midband_veto_cfg.get("p_up_low", weak_veto_low))
                    midband_veto_high = float(midband_veto_cfg.get("p_up_high", weak_veto_high))
                    midband_veto_high_inclusive = bool(midband_veto_cfg.get("high_inclusive", weak_veto_high_inclusive))
                    midband_veto_min_abs_ret_pred = float(midband_veto_cfg.get("min_abs_ret_pred", 0.0005))
                    midband_veto_max_abs_ret_pred = float(midband_veto_cfg.get("max_abs_ret_pred", 0.001))
                    midband_veto_reference_path_cfg = midband_veto_cfg.get("incumbent_reference_path")
                    raw_ev_veto_cfg_obj = weak_veto_cfg.get("raw_ev_sign", {})
                    raw_ev_veto_cfg = raw_ev_veto_cfg_obj if isinstance(raw_ev_veto_cfg_obj, dict) else {}
                    raw_ev_veto_low = float(raw_ev_veto_cfg.get("p_up_low", weak_veto_low))
                    raw_ev_veto_high = float(raw_ev_veto_cfg.get("p_up_high", weak_veto_high))
                    raw_ev_veto_high_inclusive = bool(raw_ev_veto_cfg.get("high_inclusive", weak_veto_high_inclusive))
                    raw_ev_veto_max = float(raw_ev_veto_cfg.get("raw_ev_sign_max", 0.0))
                    raw_ev_veto_reference_path_cfg = raw_ev_veto_cfg.get("incumbent_reference_path")
                    direction_align_veto_cfg_obj = weak_veto_cfg.get("direction_alignment", {})
                    direction_align_veto_cfg = (
                        direction_align_veto_cfg_obj if isinstance(direction_align_veto_cfg_obj, dict) else {}
                    )
                    direction_align_veto_low = float(direction_align_veto_cfg.get("p_up_low", weak_veto_low))
                    direction_align_veto_high = float(direction_align_veto_cfg.get("p_up_high", weak_veto_high))
                    direction_align_veto_high_inclusive = bool(
                        direction_align_veto_cfg.get("high_inclusive", weak_veto_high_inclusive)
                    )
                    direction_align_veto_require_aligned = bool(
                        direction_align_veto_cfg.get("require_aligned", False)
                    )
                    direction_align_veto_use_midband_slice = bool(
                        direction_align_veto_cfg.get("use_midband_slice", True)
                    )
                    direction_align_veto_min_abs_ret_pred = float(
                        direction_align_veto_cfg.get("min_abs_ret_pred", midband_veto_min_abs_ret_pred)
                    )
                    direction_align_veto_max_abs_ret_pred = float(
                        direction_align_veto_cfg.get("max_abs_ret_pred", midband_veto_max_abs_ret_pred)
                    )
                    direction_align_veto_reference_path_cfg = direction_align_veto_cfg.get("incumbent_reference_path")
                    joint_direction_midband_veto_cfg_obj = weak_veto_cfg.get("joint_direction_midband", {})
                    joint_direction_midband_veto_cfg = (
                        joint_direction_midband_veto_cfg_obj if isinstance(joint_direction_midband_veto_cfg_obj, dict) else {}
                    )
                    joint_direction_midband_veto_low = float(joint_direction_midband_veto_cfg.get("p_up_low", weak_veto_low))
                    joint_direction_midband_veto_high = float(joint_direction_midband_veto_cfg.get("p_up_high", weak_veto_high))
                    joint_direction_midband_veto_high_inclusive = bool(
                        joint_direction_midband_veto_cfg.get("high_inclusive", weak_veto_high_inclusive)
                    )
                    joint_direction_midband_veto_require_aligned = bool(
                        joint_direction_midband_veto_cfg.get("require_aligned", False)
                    )
                    joint_direction_midband_veto_min_abs_ret_pred = float(
                        joint_direction_midband_veto_cfg.get("min_abs_ret_pred", midband_veto_min_abs_ret_pred)
                    )
                    joint_direction_midband_veto_max_abs_ret_pred = float(
                        joint_direction_midband_veto_cfg.get("max_abs_ret_pred", midband_veto_max_abs_ret_pred)
                    )
                    joint_direction_midband_veto_reference_path_cfg = joint_direction_midband_veto_cfg.get("incumbent_reference_path")
                    regime_state_veto_cfg_obj = weak_veto_cfg.get("regime_state", {})
                    regime_state_veto_cfg = (
                        regime_state_veto_cfg_obj if isinstance(regime_state_veto_cfg_obj, dict) else {}
                    )
                    regime_state_veto_low = float(regime_state_veto_cfg.get("p_up_low", weak_veto_low))
                    regime_state_veto_high = float(regime_state_veto_cfg.get("p_up_high", weak_veto_high))
                    regime_state_veto_high_inclusive = bool(
                        regime_state_veto_cfg.get("high_inclusive", weak_veto_high_inclusive)
                    )
                    regime_state_veto_use_midband_slice = bool(
                        regime_state_veto_cfg.get("use_midband_slice", False)
                    )
                    regime_state_veto_min_abs_ret_pred = float(
                        regime_state_veto_cfg.get("min_abs_ret_pred", midband_veto_min_abs_ret_pred)
                    )
                    regime_state_veto_max_abs_ret_pred = float(
                        regime_state_veto_cfg.get("max_abs_ret_pred", midband_veto_max_abs_ret_pred)
                    )
                    regime_state_veto_min_regime_rows = int(regime_state_veto_cfg.get("min_regime_rows", 1))
                    regime_state_veto_reference_path_cfg = regime_state_veto_cfg.get("incumbent_reference_path")
                    regime_state_veto_override_obj = regime_state_veto_cfg.get("regime_states", [])
                    regime_state_veto_override = [
                        str(value).strip().lower()
                        for value in (regime_state_veto_override_obj if isinstance(regime_state_veto_override_obj, list) else [])
                        if str(value).strip()
                    ]
                    chop_high_vol_veto_cfg_obj = weak_veto_cfg.get("chop_high_volatility", {})
                    chop_high_vol_veto_cfg = (
                        chop_high_vol_veto_cfg_obj if isinstance(chop_high_vol_veto_cfg_obj, dict) else {}
                    )
                    chop_high_vol_veto_low = float(chop_high_vol_veto_cfg.get("p_up_low", weak_veto_low))
                    chop_high_vol_veto_high = float(chop_high_vol_veto_cfg.get("p_up_high", weak_veto_high))
                    chop_high_vol_veto_high_inclusive = bool(
                        chop_high_vol_veto_cfg.get("high_inclusive", weak_veto_high_inclusive)
                    )
                    chop_high_vol_veto_regime_state = str(
                        chop_high_vol_veto_cfg.get("regime_state", "chop")
                    ).strip().lower()
                    chop_high_vol_veto_volatility_col = str(
                        chop_high_vol_veto_cfg.get("volatility_col", "volatility_realized_24h")
                    )
                    chop_high_vol_veto_use_midband_slice = bool(
                        chop_high_vol_veto_cfg.get("use_midband_slice", False)
                    )
                    chop_high_vol_veto_min_abs_ret_pred = float(
                        chop_high_vol_veto_cfg.get("min_abs_ret_pred", midband_veto_min_abs_ret_pred)
                    )
                    chop_high_vol_veto_max_abs_ret_pred = float(
                        chop_high_vol_veto_cfg.get("max_abs_ret_pred", midband_veto_max_abs_ret_pred)
                    )
                    chop_high_vol_veto_reference_path_cfg = chop_high_vol_veto_cfg.get("incumbent_reference_path")
                    chop_high_vol_veto_min_volatility_cfg = chop_high_vol_veto_cfg.get("min_volatility")
                    volatility_only_veto_cfg_obj = weak_veto_cfg.get("volatility_only", {})
                    volatility_only_veto_cfg = (
                        volatility_only_veto_cfg_obj if isinstance(volatility_only_veto_cfg_obj, dict) else {}
                    )
                    volatility_only_veto_low = float(volatility_only_veto_cfg.get("p_up_low", weak_veto_low))
                    volatility_only_veto_high = float(volatility_only_veto_cfg.get("p_up_high", weak_veto_high))
                    volatility_only_veto_high_inclusive = bool(
                        volatility_only_veto_cfg.get("high_inclusive", weak_veto_high_inclusive)
                    )
                    volatility_only_veto_volatility_col = str(
                        volatility_only_veto_cfg.get("volatility_col", chop_high_vol_veto_volatility_col)
                    )
                    volatility_only_veto_use_midband_slice = bool(
                        volatility_only_veto_cfg.get("use_midband_slice", False)
                    )
                    volatility_only_veto_min_abs_ret_pred = float(
                        volatility_only_veto_cfg.get("min_abs_ret_pred", midband_veto_min_abs_ret_pred)
                    )
                    volatility_only_veto_max_abs_ret_pred = float(
                        volatility_only_veto_cfg.get("max_abs_ret_pred", midband_veto_max_abs_ret_pred)
                    )
                    volatility_only_veto_reference_path_cfg = volatility_only_veto_cfg.get("incumbent_reference_path")
                    volatility_only_veto_min_volatility_cfg = volatility_only_veto_cfg.get("min_volatility")

                    aligned_candidate_input = summary_dir / "backtest_signals_meta_ensemble_decision_aligned.csv"
                    common_align_args = [
                        "--threshold",
                        str(float(trade_policy_cfg.get("threshold", trade_decision_cfg.get("threshold", 0.55)))),
                        "--fee-bps",
                        str(float(quality_cfg.get("walkforward_fee_bps", 2.0))),
                        "--slippage-bps",
                        str(float(quality_cfg.get("walkforward_slippage_bps", 1.0))),
                        "--replace-threshold-rule",
                        "1" if bool(trade_policy_cfg.get("replace_threshold_rule", True)) else "0",
                        "--require-direction-ret-alignment",
                        "1" if bool(trade_policy_cfg.get("require_direction_ret_alignment", True)) else "0",
                        "--use-oof-expected-value",
                        "1" if bool(trade_policy_cfg.get("use_oof_expected_value", True)) else "0",
                        "--oof-expected-value-mode",
                        str(trade_policy_cfg.get("oof_expected_value_mode", "max_with_raw_calibrated")),
                        "--enforce-positive-oof-envelope",
                        "1" if bool(trade_policy_cfg.get("enforce_positive_oof_envelope", False)) else "0",
                        "--block-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("block_when_no_positive_oof_bin", True)) else "0",
                        "--positive-oof-min-samples",
                        str(int(trade_policy_cfg.get("positive_oof_min_samples", 4))),
                        "--allow-raw-ev-fallback-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False)) else "0",
                        "--raw-ev-fallback-quantile",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_quantile", 0.9))),
                        "--raw-ev-fallback-min-edge-over-fee",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_min_edge_over_fee", 0.0))),
                        "--min-expected-net",
                        str(float(trade_policy_cfg.get("min_expected_net", 0.0))),
                        "--min-edge-over-fee",
                        str(float(trade_policy_cfg.get("min_edge_over_fee", 0.0))),
                    ]

                    align_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(decision_model_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(aligned_candidate_input),
                        "--meta-output",
                        str(summary_dir / "backtest_signals_meta_ensemble_decision_aligned_meta.json"),
                    ]
                    align_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        align_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        align_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    if weak_veto_official_mode == "midband":
                        align_cmd.extend(
                            [
                                "--midband-candidate-only-veto",
                                "1" if weak_veto_enabled_official else "0",
                                "--midband-pup-low",
                                str(midband_veto_low),
                                "--midband-pup-high",
                                str(midband_veto_high),
                                "--midband-high-inclusive",
                                "1" if midband_veto_high_inclusive else "0",
                                "--midband-min-abs-ret-pred",
                                str(midband_veto_min_abs_ret_pred),
                                "--midband-max-abs-ret-pred",
                                str(midband_veto_max_abs_ret_pred),
                            ]
                        )
                        if midband_veto_reference_path_cfg:
                            align_cmd.extend(["--midband-incumbent-reference", str(midband_veto_reference_path_cfg)])
                    else:
                        align_cmd.extend(
                            [
                                "--weak-band-candidate-only-veto",
                                "1" if weak_veto_enabled_official else "0",
                                "--weak-band-pup-low",
                                str(weak_veto_low),
                                "--weak-band-pup-high",
                                str(weak_veto_high),
                                "--weak-band-high-inclusive",
                                "1" if weak_veto_high_inclusive else "0",
                            ]
                        )
                        if weak_veto_reference_path_cfg:
                            align_cmd.extend(["--weak-band-incumbent-reference", str(weak_veto_reference_path_cfg)])
                    results.append(
                        _run_step(
                            "trade_decision_align_candidate",
                            align_cmd,
                            logs_dir / "trade_decision_align_candidate.log",
                            args.dry_run,
                        )
                    )

                    raw_candidate_diag_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(decision_model_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(summary_dir / "backtest_signals_meta_ensemble_policy_diagnostics_candidate_raw.csv"),
                        "--diagnostics-output",
                        str(summary_dir / "trade_decision_diagnostics_candidate_raw.json"),
                        "--meta-output",
                        str(summary_dir / "trade_decision_diagnostics_candidate_raw_meta.json"),
                        "--diagnostics-only",
                    ]
                    raw_candidate_diag_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        raw_candidate_diag_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        raw_candidate_diag_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    results.append(
                        _run_step(
                            "trade_decision_diagnostics_candidate_raw",
                            raw_candidate_diag_cmd,
                            logs_dir / "trade_decision_diagnostics_candidate_raw.log",
                            args.dry_run,
                        )
                    )

                    if aligned_candidate_input.exists() or args.dry_run:
                        candidate_gate_input = aligned_candidate_input
                        aligned_quality_cmd = [
                            python,
                            "-m",
                            "src.scripts.evaluate_model_quality",
                            "--input",
                            str(candidate_gate_input),
                            "--output",
                            str(summary_dir / "model_quality_candidate.json"),
                        ]
                        results.append(
                            _run_step(
                                "model_quality_eval_decision_aligned",
                                aligned_quality_cmd,
                                logs_dir / "model_quality_eval_decision_aligned.log",
                                args.dry_run,
                            )
                        )

                        overlap_stability_output = summary_dir / "overlap_trust_stability.json"
                        if (
                            quality_input.exists()
                            and labeled_overlap_dataset is not None
                            and labeled_overlap_dataset.exists()
                            and full_selected_path is not None
                            and overlap_selected_path is not None
                            and full_selected_path.exists()
                            and overlap_selected_path.exists()
                        ):
                            overlap_feature_sources = [aligned_candidate_input, quality_input]
                            overlap_diag_cmd = [
                                python,
                                "-m",
                                "src.scripts.analyze_overlap_trust_stability",
                                "--full-walkforward",
                                str(full_selected_path),
                                "--overlap-walkforward",
                                str(overlap_selected_path),
                                "--overlap-dataset",
                                str(labeled_overlap_dataset),
                                "--labeled-csv",
                                str(quality_input),
                                "--feature-source",
                                str(overlap_feature_sources[0]),
                                "--feature-source",
                                str(overlap_feature_sources[1]),
                                "--ts-col",
                                str(reconcile_cfg.get("ts_col", "ts")),
                                "--return-col",
                                str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                                "--signal-col",
                                str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                                "--output",
                                str(overlap_stability_output),
                            ]
                            results.append(
                                _run_step(
                                    "overlap_trust_stability_decision_aligned_refresh",
                                    overlap_diag_cmd,
                                    logs_dir / "overlap_trust_stability_decision_aligned_refresh.log",
                                    args.dry_run,
                                )
                            )

            hygiene_cmd = [
                python,
                "-m",
                "src.scripts.analyze_ensemble_hygiene",
                "--input",
                str(quality_input),
                "--output",
                str(summary_dir / "ensemble_hygiene.json"),
            ]
            results.append(
                _run_step(
                    "ensemble_hygiene",
                    hygiene_cmd,
                    logs_dir / "ensemble_hygiene.log",
                    args.dry_run,
                    allowed_returncodes=[0, 2],
                )
            )

            if bool(tuning_cfg.get("enabled", True)):
                optimize_on_overlap = bool(tuning_cfg.get("optimize_on_labeled_overlap", False)) and bool(overlap_as_primary)
                overlap_pruning_cfg_obj = overlap_pre_tuning_cfg.get("model_pruning", {}) if isinstance(overlap_pre_tuning_cfg, dict) else {}
                overlap_pruning_cfg = overlap_pruning_cfg_obj if isinstance(overlap_pruning_cfg_obj, dict) else {}
                fallback_tuning_cfg_obj = overlap_pruning_cfg.get("fallback_tuning", {}) if isinstance(overlap_pruning_cfg, dict) else {}
                fallback_tuning_cfg = fallback_tuning_cfg_obj if isinstance(fallback_tuning_cfg_obj, dict) else {}
                fallback_tuning_enabled = bool(fallback_tuning_cfg.get("enabled", False))

                def _build_tuning_cmd(
                    *,
                    p_up_grid: str,
                    ret_min_grid: str,
                    direction_threshold_grid: str,
                    min_trades: int,
                    max_drawdown: float,
                    min_cum_ret: float,
                    selection_metric: str,
                    economics_turnover_penalty: float,
                    economics_downside_penalty: float,
                    stability_gap_penalty: float,
                    max_stability_gap: float,
                    min_overlap_rows: int,
                    strict_accept: bool,
                ) -> List[str]:
                    cmd = [
                        python,
                        "-m",
                        "src.scripts.tune_joint_signal_thresholds",
                        "--input",
                        str(quality_input),
                        f"--p-up-grid={p_up_grid}",
                        f"--ret-min-grid={ret_min_grid}",
                        f"--direction-threshold-grid={direction_threshold_grid}",
                        "--min-trades",
                        str(int(min_trades)),
                        "--max-dd",
                        str(float(max_drawdown)),
                        "--min-cum-ret",
                        str(float(min_cum_ret)),
                        "--selection-metric",
                        str(selection_metric),
                        "--economics-turnover-penalty",
                        str(float(economics_turnover_penalty)),
                        "--economics-downside-penalty",
                        str(float(economics_downside_penalty)),
                        "--stability-gap-penalty",
                        str(float(stability_gap_penalty)),
                        "--max-stability-gap",
                        str(float(max_stability_gap)),
                        "--min-overlap-rows",
                        str(int(min_overlap_rows)),
                        "--output",
                        str(joint_tuning_output_path),
                    ]
                    if optimize_on_overlap and labeled_overlap_dataset is not None:
                        cmd.extend(
                            [
                                "--overlap-dataset",
                                str(labeled_overlap_dataset),
                                "--ts-col",
                                str(reconcile_cfg.get("ts_col", "ts")),
                            ]
                        )
                    if bool(strict_accept):
                        cmd.append("--strict-accept")
                    return cmd

                run_tuning = True
                used_fallback_tuning = False
                if optimize_on_overlap and not overlap_pruning_allows_tuning:
                    if fallback_tuning_enabled:
                        used_fallback_tuning = True
                        print(
                            "Overlap model pruning found no viable candidates; running constrained fallback joint-threshold tuning.",
                            file=sys.stderr,
                        )
                        tuning_cmd = _build_tuning_cmd(
                            p_up_grid=str(fallback_tuning_cfg.get("p_up_grid", "0.58,0.60,0.62")),
                            ret_min_grid=str(fallback_tuning_cfg.get("ret_min_grid", "0.0,0.0001,0.0002")),
                            direction_threshold_grid=str(fallback_tuning_cfg.get("direction_threshold_grid", "0.54,0.56,0.58")),
                            min_trades=int(fallback_tuning_cfg.get("min_trades", tuning_cfg.get("min_trades", 10))),
                            max_drawdown=float(fallback_tuning_cfg.get("max_drawdown", tuning_cfg.get("max_drawdown", -0.12))),
                            min_cum_ret=float(fallback_tuning_cfg.get("min_cum_ret", tuning_cfg.get("min_cum_ret", 0.0))),
                            selection_metric=str(fallback_tuning_cfg.get("selection_metric", tuning_cfg.get("selection_metric", "economics_score"))),
                            economics_turnover_penalty=float(
                                fallback_tuning_cfg.get(
                                    "economics_turnover_penalty",
                                    tuning_cfg.get("economics_turnover_penalty", 0.002),
                                )
                            ),
                            economics_downside_penalty=float(
                                fallback_tuning_cfg.get(
                                    "economics_downside_penalty",
                                    tuning_cfg.get("economics_downside_penalty", 2.0),
                                )
                            ),
                            stability_gap_penalty=float(
                                fallback_tuning_cfg.get(
                                    "stability_gap_penalty",
                                    tuning_cfg.get("stability_gap_penalty", 0.0),
                                )
                            ),
                            max_stability_gap=float(
                                fallback_tuning_cfg.get(
                                    "max_stability_gap",
                                    tuning_cfg.get("max_stability_gap", 1e9),
                                )
                            ),
                            min_overlap_rows=int(
                                fallback_tuning_cfg.get(
                                    "min_overlap_rows",
                                    tuning_cfg.get("min_overlap_rows", 0),
                                )
                            ),
                            strict_accept=bool(
                                fallback_tuning_cfg.get(
                                    "enforce_deployable",
                                    tuning_cfg.get("enforce_deployable", True),
                                )
                            ),
                        )
                    else:
                        run_tuning = False
                        joint_tuning_accepted = False
                        blocked_payload = {
                            "rows": 0,
                            "constraints": {
                                "selection_metric": str(tuning_cfg.get("selection_metric", "cum_ret")),
                                "optimize_on_labeled_overlap": True,
                            },
                            "accepted": False,
                            "reason": "blocked_by_overlap_model_pruning",
                            "overlap_model_pruning_path": str(overlap_model_pruning_path),
                            "fallback_tuning_enabled": False,
                        }
                        joint_tuning_output_path.write_text(json.dumps(blocked_payload, indent=2), encoding="utf-8")
                        print(
                            "Skipping joint_threshold_tuning: overlap model pruning found no viable model candidates.",
                            file=sys.stderr,
                        )
                else:
                    tuning_cmd = _build_tuning_cmd(
                        p_up_grid=str(tuning_cfg.get("p_up_grid", "0.50,0.55,0.60,0.65")),
                        ret_min_grid=str(tuning_cfg.get("ret_min_grid", "-0.0002,0.0,0.0002,0.0005")),
                        direction_threshold_grid=str(tuning_cfg.get("direction_threshold_grid", "0.50,0.55,0.60")),
                        min_trades=int(tuning_cfg.get("min_trades", 10)),
                        max_drawdown=float(tuning_cfg.get("max_drawdown", -0.12)),
                        min_cum_ret=float(tuning_cfg.get("min_cum_ret", 0.0)),
                        selection_metric=str(tuning_cfg.get("selection_metric", "cum_ret")),
                        economics_turnover_penalty=float(tuning_cfg.get("economics_turnover_penalty", 0.002)),
                        economics_downside_penalty=float(tuning_cfg.get("economics_downside_penalty", 2.0)),
                        stability_gap_penalty=float(tuning_cfg.get("stability_gap_penalty", 0.0)),
                        max_stability_gap=float(tuning_cfg.get("max_stability_gap", 1e9)),
                        min_overlap_rows=int(tuning_cfg.get("min_overlap_rows", 0)),
                        strict_accept=bool(tuning_cfg.get("enforce_deployable", True)),
                    )

                if run_tuning:
                    results.append(
                        _run_step(
                            "joint_threshold_tuning_fallback_overlap" if used_fallback_tuning else "joint_threshold_tuning",
                            tuning_cmd,
                            logs_dir / ("joint_threshold_tuning_fallback_overlap.log" if used_fallback_tuning else "joint_threshold_tuning.log"),
                            args.dry_run,
                            allowed_returncodes=[0, 2],
                        )
                    )

                    if joint_tuning_output_path.exists() and not args.dry_run:
                        joint_tuning_payload = _load_json(joint_tuning_output_path)
                        if used_fallback_tuning:
                            joint_tuning_payload["fallback_tuning_used"] = True
                            joint_tuning_payload["fallback_tuning_source"] = "overlap_model_pruning"
                            joint_tuning_output_path.write_text(json.dumps(joint_tuning_payload, indent=2), encoding="utf-8")
                        joint_tuning_accepted = bool(joint_tuning_payload.get("accepted", False))

                        sweep_cfg_obj = tuning_cfg.get("overlap_threshold_sweep_fallback", {}) if isinstance(tuning_cfg, dict) else {}
                        sweep_cfg = sweep_cfg_obj if isinstance(sweep_cfg_obj, dict) else {}
                        if (
                            (not joint_tuning_accepted)
                            and optimize_on_overlap
                            and bool(sweep_cfg.get("enabled", True))
                            and labeled_overlap_dataset is not None
                            and labeled_overlap_dataset.exists()
                        ):
                            overlap_compare_path = summary_dir / "walkforward_model_compare_labeled_overlap.json"
                            overlap_selected_model = "xgb"
                            resolved_walkforward: Dict[str, Any] = {}
                            if overlap_compare_path.exists():
                                overlap_compare_payload = _load_json(overlap_compare_path)
                                overlap_selected_model = str(
                                    overlap_compare_payload.get("selected_model_kind", overlap_selected_model)
                                )
                                resolved_obj = overlap_compare_payload.get("resolved_walkforward", {})
                                if isinstance(resolved_obj, dict):
                                    resolved_walkforward = resolved_obj

                            threshold_grid_raw = str(sweep_cfg.get("threshold_grid", "0.48,0.50,0.52,0.55,0.58,0.60"))
                            threshold_grid = [
                                float(v.strip())
                                for v in threshold_grid_raw.split(",")
                                if v is not None and v.strip() != ""
                            ]
                            min_trades_sweep = int(sweep_cfg.get("min_trades", tuning_cfg.get("min_trades", 10)))
                            min_cum_ret_sweep = float(sweep_cfg.get("min_cum_ret", tuning_cfg.get("min_cum_ret", 0.0)))
                            folds_sweep = int(resolved_walkforward.get("folds", reconcile_cfg.get("overlap_compare", {}).get("folds", cv_folds)))
                            train_sweep = int(resolved_walkforward.get("train_size", reconcile_cfg.get("overlap_compare", {}).get("train_size", cv_train_size)))
                            val_sweep = int(resolved_walkforward.get("val_size", reconcile_cfg.get("overlap_compare", {}).get("val_size", cv_val_size)))
                            test_sweep = int(resolved_walkforward.get("test_size", reconcile_cfg.get("overlap_compare", {}).get("test_size", cv_test_size)))
                            gap_sweep = int(resolved_walkforward.get("gap", reconcile_cfg.get("overlap_compare", {}).get("gap", cv_gap)))
                            purge_sweep = int(resolved_walkforward.get("purge_size", reconcile_cfg.get("overlap_compare", {}).get("purge_size", cv_purge_size)))
                            embargo_sweep = int(resolved_walkforward.get("embargo_size", reconcile_cfg.get("overlap_compare", {}).get("embargo_size", cv_embargo_size)))
                            mode_sweep = str(reconcile_cfg.get("overlap_compare", {}).get("mode", cv_mode))

                            sweep_rows: List[Dict[str, Any]] = []
                            for threshold in threshold_grid:
                                sweep_name = str(threshold).replace(".", "p")
                                sweep_output = summary_dir / f"overlap_threshold_sweep_{sweep_name}.json"
                                sweep_cmd = [
                                    python,
                                    "-m",
                                    "src.scripts.run_walkforward_validation",
                                    "--dataset-path",
                                    str(labeled_overlap_dataset),
                                    "--y-key",
                                    str(walkforward_target),
                                    "--folds",
                                    str(int(folds_sweep)),
                                    "--train-size",
                                    str(int(train_sweep)),
                                    "--val-size",
                                    str(int(val_sweep)),
                                    "--test-size",
                                    str(int(test_sweep)),
                                    "--gap",
                                    str(int(gap_sweep)),
                                    "--purge-size",
                                    str(int(purge_sweep)),
                                    "--embargo-size",
                                    str(int(embargo_sweep)),
                                    "--mode",
                                    str(mode_sweep),
                                    "--model-kind",
                                    str(overlap_selected_model),
                                    "--signal-threshold",
                                    str(float(threshold)),
                                    "--fee-bps",
                                    str(float(compare_cfg.get("fee_bps", quality_cfg.get("walkforward_fee_bps", 2.0)))),
                                    "--slippage-bps",
                                    str(float(compare_cfg.get("slippage_bps", quality_cfg.get("walkforward_slippage_bps", 1.0)))),
                                    "--output",
                                    str(sweep_output),
                                ]
                                results.append(
                                    _run_step(
                                        f"overlap_threshold_sweep_{sweep_name}",
                                        sweep_cmd,
                                        logs_dir / f"overlap_threshold_sweep_{sweep_name}.log",
                                        args.dry_run,
                                    )
                                )
                                if sweep_output.exists():
                                    sweep_payload = _load_json(sweep_output)
                                    sweep_rows.append(
                                        {
                                            "signal_threshold": float(threshold),
                                            "cum_ret_net_total": float(sweep_payload.get("cum_ret_net_total", float("nan"))),
                                            "trade_count_total": int(sweep_payload.get("trade_count_total", 0) or 0),
                                            "auc_mean": float(sweep_payload.get("auc_mean", float("nan"))),
                                            "path": str(sweep_output),
                                        }
                                    )

                            deployable_rows = [
                                row
                                for row in sweep_rows
                                if int(row.get("trade_count_total", 0) or 0) >= int(min_trades_sweep)
                                and float(row.get("cum_ret_net_total", float("-inf"))) >= float(min_cum_ret_sweep)
                            ]
                            if deployable_rows:
                                best_sweep = max(
                                    deployable_rows,
                                    key=lambda r: (
                                        float(r.get("cum_ret_net_total", float("-inf"))),
                                        int(r.get("trade_count_total", 0) or 0),
                                    ),
                                )
                                horizon_key = str(int(walkforward_horizon))
                                ret_min_from_thresholds = 0.0
                                if thresholds_path.exists():
                                    current_thresholds_payload = _load_json(thresholds_path)
                                    horizons_payload = current_thresholds_payload.get("horizons", {})
                                    if isinstance(horizons_payload, dict):
                                        horizon_payload = horizons_payload.get(horizon_key, {})
                                        if isinstance(horizon_payload, dict):
                                            ret_min_from_thresholds = float(horizon_payload.get("ret_min", 0.0) or 0.0)

                                joint_tuning_payload = {
                                    "rows": int(len(sweep_rows)),
                                    "constraints": {
                                        "selection_metric": "cum_ret_net_total",
                                        "optimize_on_labeled_overlap": True,
                                        "min_trades": int(min_trades_sweep),
                                        "min_cum_ret": float(min_cum_ret_sweep),
                                        "source": "overlap_threshold_sweep_fallback",
                                    },
                                    "accepted": True,
                                    "best": {
                                        "p_up_min": float(best_sweep.get("signal_threshold", 0.5) or 0.5),
                                        "ret_min": float(ret_min_from_thresholds),
                                        "direction_threshold": 0.5,
                                        "n_trades": int(best_sweep.get("trade_count_total", 0) or 0),
                                        "cum_ret": float(best_sweep.get("cum_ret_net_total", 0.0) or 0.0),
                                        "full_cum_ret": float(best_sweep.get("cum_ret_net_total", 0.0) or 0.0),
                                        "stability_gap": 0.0,
                                        "max_drawdown": float("nan"),
                                        "economics_score": float(best_sweep.get("cum_ret_net_total", 0.0) or 0.0),
                                        "selection_value": float(best_sweep.get("cum_ret_net_total", 0.0) or 0.0),
                                    },
                                    "n_candidates": int(len(sweep_rows)),
                                    "n_feasible": int(len(deployable_rows)),
                                    "n_deployable": int(len(deployable_rows)),
                                    "fallback_tuning_used": True,
                                    "fallback_tuning_source": "overlap_threshold_sweep_fallback",
                                    "overlap_selected_model": str(overlap_selected_model),
                                    "sweep_rows": sweep_rows,
                                }
                                joint_tuning_output_path.write_text(json.dumps(joint_tuning_payload, indent=2), encoding="utf-8")
                                joint_tuning_accepted = True
                                print(
                                    "Joint threshold tuning fallback produced deployable overlap candidate via threshold sweep.",
                                    file=sys.stderr,
                                )

                        if joint_tuning_accepted and thresholds_path.exists():
                            applied = _apply_joint_tuning_to_thresholds(
                                thresholds_path,
                                joint_tuning_payload=joint_tuning_payload,
                                horizon=int(walkforward_horizon),
                            )
                            if not applied:
                                print(
                                    "Warning: joint threshold tuning accepted but thresholds artifact could not be patched.",
                                    file=sys.stderr,
                                )
                            last_deployable_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(thresholds_path, last_deployable_thresholds_path)
                        elif (not joint_tuning_accepted) and use_last_deployable_on_joint_reject:
                            if last_deployable_thresholds_path.exists():
                                thresholds_path = last_deployable_thresholds_path
                                used_last_deployable_thresholds = True
                            else:
                                print(
                                    "Joint threshold tuning was non-deployable and no last deployable thresholds artifact was found; "
                                    f"continuing with current run thresholds at {thresholds_path}.",
                                    file=sys.stderr,
                                )
                    elif args.dry_run:
                        joint_tuning_accepted = None

            if bool(feature_rel_cfg.get("enabled", True)):
                feature_rel_input = Path(feature_rel_cfg.get("input") or str(walkforward_dataset))
                feature_rel_cmd = [
                    python,
                    "-m",
                    "src.scripts.evaluate_feature_reliability",
                    "--input",
                    str(feature_rel_input),
                    "--baseline-window",
                    str(int(feature_rel_cfg.get("baseline_window", 240))),
                    "--recent-window",
                    str(int(feature_rel_cfg.get("recent_window", 120))),
                    "--min-score",
                    str(float(feature_rel_cfg.get("min_score", 0.55))),
                    "--max-features",
                    str(int(feature_rel_cfg.get("max_features", 0))),
                    "--output",
                    str(summary_dir / "feature_reliability.json"),
                ]
                results.append(
                    _run_step(
                        "feature_reliability",
                        feature_rel_cmd,
                        logs_dir / "feature_reliability.log",
                        args.dry_run,
                    )
                )

            if bool(calibration_cfg.get("enabled", True)):
                calibration_cmd = [
                    python,
                    "-m",
                    "src.scripts.evaluate_calibration_robustness",
                    "--input",
                    str(quality_input),
                    "--baseline-window",
                    str(int(calibration_cfg.get("baseline_window", 240))),
                    "--recent-window",
                    str(int(calibration_cfg.get("recent_window", 120))),
                    "--max-ece-drift",
                    str(float(calibration_cfg.get("max_ece_drift", 0.02))),
                    "--output",
                    str(summary_dir / "calibration_robustness.json"),
                ]
                results.append(_run_step("calibration_robustness", calibration_cmd, logs_dir / "calibration_robustness.log", args.dry_run))

                if bool(calibration_cfg.get("regime_aware", True)):
                    regime_calib_cmd = [
                        python,
                        "-m",
                        "src.scripts.train_platt_calibration",
                        "--horizons",
                        *[str(h) for h in horizons],
                        "--output-path",
                        str(summary_dir / "platt_calibration.json"),
                        "--method",
                        str(calibration_cfg.get("method", "platt")),
                        "--labeled-input",
                        str(quality_input),
                        "--regime-col",
                        str(calibration_cfg.get("regime_col", "regime_state")),
                        "--min-regime-rows",
                        str(int(calibration_cfg.get("min_regime_rows", 100))),
                    ]
                    results.append(
                        _run_step(
                            "platt_calibration_regime_aware",
                            regime_calib_cmd,
                            logs_dir / "platt_calibration_regime_aware.log",
                            args.dry_run,
                        )
                    )

            if bool(regime_weakness_cfg.get("enabled", True)):
                calibration_path = summary_dir / "calibration_robustness.json"
                if (calibration_path.exists() and walkforward_output.exists()) or args.dry_run:
                    regime_weakness_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_regime_weakness",
                        "--calibration",
                        str(calibration_path),
                        "--walkforward",
                        str(walkforward_output),
                        "--max-ece-drift",
                        str(float(regime_weakness_cfg.get("max_ece_drift", calibration_cfg.get("max_ece_drift", 0.02)))),
                        "--min-net-return",
                        str(float(regime_weakness_cfg.get("min_net_return", 0.0))),
                        "--output",
                        str(summary_dir / "regime_weakness.json"),
                    ]
                    results.append(
                        _run_step(
                            "regime_weakness",
                            regime_weakness_cmd,
                            logs_dir / "regime_weakness.log",
                            args.dry_run,
                        )
                    )

            if bool(rolling_ab_cfg.get("enabled", False)):
                baseline_input = rolling_ab_cfg.get("baseline_input")
                candidate_input = rolling_ab_cfg.get("candidate_input")
                if baseline_input and candidate_input:
                    rolling_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_rolling_ab",
                        "--baseline",
                        str(baseline_input),
                        "--candidate",
                        str(candidate_input),
                        "--window-size",
                        str(int(rolling_ab_cfg.get("window_size", 168))),
                        "--step-size",
                        str(int(rolling_ab_cfg.get("step_size", 24))),
                        "--min-window-trades",
                        str(int(rolling_ab_cfg.get("min_window_trades", 5))),
                        "--output",
                        str(summary_dir / "rolling_ab_report.json"),
                        "--output-md",
                        str(summary_dir / "rolling_ab_report.md"),
                    ]
                    if bool(rolling_ab_cfg.get("allow_no_trade_baseline", False)):
                        rolling_cmd.append("--allow-no-trade-baseline")
                    results.append(_run_step("rolling_ab_report", rolling_cmd, logs_dir / "rolling_ab_report.log", args.dry_run))
                else:
                    print(
                        "Warning: quality.rolling_ab.enabled=true but baseline_input/candidate_input not set; skipping rolling A/B.",
                        file=sys.stderr,
                    )

        incumbent_quality_path = quality_cfg.get("incumbent_quality_path")
        incumbent_backtest_csv = quality_cfg.get("incumbent_backtest_csv")
        if incumbent_quality_path and incumbent_backtest_csv:
            incumbent_labeled = summary_dir / "labeled_backtest_incumbent_1h.csv"
            incumbent_labeled_cmd = [
                python,
                "-m",
                "src.scripts.build_labeled_backtest_from_history",
                "--backtest-csv",
                str(incumbent_backtest_csv),
                "--prefer-backtest",
                "--output",
                str(incumbent_labeled),
                "--meta-output",
                str(summary_dir / "labeled_backtest_incumbent_meta.json"),
                "--fold-size",
                str(int(quality_cfg.get("fold_size", 12))),
                "--lookback-rows",
                str(int(quality_cfg.get("lookback_rows", 2000))),
                "--lookback-hours",
                str(int(quality_cfg.get("lookback_hours", 0))),
                "--min-rows",
                str(int(quality_cfg.get("min_labeled_rows", 200))),
            ]
            results.append(
                _run_step(
                    "build_incumbent_labeled_dataset",
                    incumbent_labeled_cmd,
                    logs_dir / "build_incumbent_labeled_dataset.log",
                    args.dry_run,
                )
            )

            incumbent_quality_cmd = [
                python,
                "-m",
                "src.scripts.evaluate_model_quality",
                "--input",
                str(incumbent_labeled),
                "--output",
                str(incumbent_quality_path),
            ]
            results.append(
                _run_step(
                    "incumbent_quality_eval",
                    incumbent_quality_cmd,
                    logs_dir / "incumbent_quality_eval.log",
                    args.dry_run,
                )
            )

        candidate_quality_path = summary_dir / "model_quality_candidate.json"
        champion_gate_payload: Dict[str, Any] | None = None
        if bool(champ_cfg.get("enabled", False)):
            baseline_raw = champ_cfg.get("baseline_input")
            candidate_raw = champ_cfg.get("candidate_input")

            baseline_input = (
                incumbent_backtest_csv
                if baseline_raw is None or str(baseline_raw).strip().lower() in {"", "auto"}
                else str(baseline_raw)
            )
            candidate_input = (
                str(candidate_gate_input)
                if candidate_raw is None or str(candidate_raw).strip().lower() in {"", "auto"}
                else str(candidate_raw)
            )

            baseline_resolution = {
                "champion_baseline_config": baseline_raw,
                "incumbent_backtest_csv": incumbent_backtest_csv,
                "resolved_baseline_input": baseline_input,
                "resolved_candidate_input": candidate_input,
                "baseline_exists": bool(Path(str(baseline_input)).exists()) if baseline_input else False,
                "candidate_exists": bool(Path(str(candidate_input)).exists()) if candidate_input else False,
                "baseline_matches_incumbent_backtest_csv": bool(
                    baseline_input and incumbent_backtest_csv and Path(str(baseline_input)).resolve() == Path(str(incumbent_backtest_csv)).resolve()
                ),
            }
            (summary_dir / "champion_baseline_resolution.json").write_text(
                json.dumps(baseline_resolution, indent=2),
                encoding="utf-8",
            )

            if baseline_input and candidate_input:
                if bool(trade_decision_cfg.get("enabled", True)) and bool(trade_decision_deploy_ready):
                    paper_live_cfg_path = Path(str(search_cfg.get("paper_live_config", "")))
                    trade_policy_cfg: Dict[str, Any] = {}
                    if paper_live_cfg_path.exists():
                        try:
                            paper_live_cfg = _load_yaml(paper_live_cfg_path)
                            trade_policy_obj = paper_live_cfg.get("trade_decision_policy", {})
                            if isinstance(trade_policy_obj, dict):
                                trade_policy_cfg = trade_policy_obj
                        except Exception as exc:
                            print(
                                f"Warning: failed to load trade_decision_policy from {paper_live_cfg_path}: {exc}",
                                file=sys.stderr,
                            )

                    baseline_diag_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(baseline_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(summary_dir / "backtest_signals_baseline_policy_diagnostics.csv"),
                        "--diagnostics-output",
                        str(summary_dir / "trade_decision_diagnostics_baseline.json"),
                        "--meta-output",
                        str(summary_dir / "trade_decision_diagnostics_baseline_meta.json"),
                        "--diagnostics-only",
                        "--threshold",
                        str(float(trade_policy_cfg.get("threshold", trade_decision_cfg.get("threshold", 0.55)))),
                        "--fee-bps",
                        str(float(quality_cfg.get("walkforward_fee_bps", 2.0))),
                        "--slippage-bps",
                        str(float(quality_cfg.get("walkforward_slippage_bps", 1.0))),
                        "--replace-threshold-rule",
                        "1" if bool(trade_policy_cfg.get("replace_threshold_rule", True)) else "0",
                        "--require-direction-ret-alignment",
                        "1" if bool(trade_policy_cfg.get("require_direction_ret_alignment", True)) else "0",
                        "--use-oof-expected-value",
                        "1" if bool(trade_policy_cfg.get("use_oof_expected_value", True)) else "0",
                        "--oof-expected-value-mode",
                        str(trade_policy_cfg.get("oof_expected_value_mode", "max_with_raw_calibrated")),
                        "--enforce-positive-oof-envelope",
                        "1" if bool(trade_policy_cfg.get("enforce_positive_oof_envelope", False)) else "0",
                        "--block-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("block_when_no_positive_oof_bin", True)) else "0",
                        "--positive-oof-min-samples",
                        str(int(trade_policy_cfg.get("positive_oof_min_samples", 4))),
                        "--allow-raw-ev-fallback-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False)) else "0",
                        "--raw-ev-fallback-quantile",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_quantile", 0.9))),
                        "--raw-ev-fallback-min-edge-over-fee",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_min_edge_over_fee", 0.0))),
                        "--min-expected-net",
                        str(float(trade_policy_cfg.get("min_expected_net", 0.0))),
                        "--min-edge-over-fee",
                        str(float(trade_policy_cfg.get("min_edge_over_fee", 0.0))),
                    ]
                    if quality_input.exists() or args.dry_run:
                        baseline_diag_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_csv:
                        baseline_diag_cmd.extend(["--feature-source", str(incumbent_backtest_csv)])
                    if candidate_input:
                        baseline_diag_cmd.extend(["--feature-source", str(candidate_input)])
                    results.append(
                        _run_step(
                            "trade_decision_diagnostics_baseline",
                            baseline_diag_cmd,
                            logs_dir / "trade_decision_diagnostics_baseline.log",
                            args.dry_run,
                        )
                    )

                    # Build a policy-aligned incumbent artifact for informational (non-gating) companion analysis.
                    incumbent_aligned_path = summary_dir / "backtest_signals_incumbent_decision_aligned.csv"
                    incumbent_aligned_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(baseline_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(incumbent_aligned_path),
                        "--meta-output",
                        str(summary_dir / "backtest_signals_incumbent_decision_aligned_meta.json"),
                        "--threshold",
                        str(float(trade_policy_cfg.get("threshold", trade_decision_cfg.get("threshold", 0.55)))),
                        "--fee-bps",
                        str(float(quality_cfg.get("walkforward_fee_bps", 2.0))),
                        "--slippage-bps",
                        str(float(quality_cfg.get("walkforward_slippage_bps", 1.0))),
                        "--replace-threshold-rule",
                        "1" if bool(trade_policy_cfg.get("replace_threshold_rule", True)) else "0",
                        "--require-direction-ret-alignment",
                        "1" if bool(trade_policy_cfg.get("require_direction_ret_alignment", True)) else "0",
                        "--use-oof-expected-value",
                        "1" if bool(trade_policy_cfg.get("use_oof_expected_value", True)) else "0",
                        "--oof-expected-value-mode",
                        str(trade_policy_cfg.get("oof_expected_value_mode", "max_with_raw_calibrated")),
                        "--enforce-positive-oof-envelope",
                        "1" if bool(trade_policy_cfg.get("enforce_positive_oof_envelope", False)) else "0",
                        "--block-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("block_when_no_positive_oof_bin", True)) else "0",
                        "--positive-oof-min-samples",
                        str(int(trade_policy_cfg.get("positive_oof_min_samples", 4))),
                        "--allow-raw-ev-fallback-when-no-positive-oof-bin",
                        "1" if bool(trade_policy_cfg.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False)) else "0",
                        "--raw-ev-fallback-quantile",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_quantile", 0.9))),
                        "--raw-ev-fallback-min-edge-over-fee",
                        str(float(trade_policy_cfg.get("raw_ev_fallback_min_edge_over_fee", 0.0))),
                        "--min-expected-net",
                        str(float(trade_policy_cfg.get("min_expected_net", 0.0))),
                        "--min-edge-over-fee",
                        str(float(trade_policy_cfg.get("min_edge_over_fee", 0.0))),
                    ]
                    if quality_input.exists() or args.dry_run:
                        incumbent_aligned_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_csv:
                        incumbent_aligned_cmd.extend(["--feature-source", str(incumbent_backtest_csv)])
                    if candidate_input:
                        incumbent_aligned_cmd.extend(["--feature-source", str(candidate_input)])
                    results.append(
                        _run_step(
                            "trade_decision_align_incumbent_companion",
                            incumbent_aligned_cmd,
                            logs_dir / "trade_decision_align_incumbent_companion.log",
                            args.dry_run,
                        )
                    )

                    companion_candidate_input = str(candidate_input)
                    companion_baseline_input = str(incumbent_aligned_path)
                    official_profile_summary_path = summary_dir / "official_profile_policy_metrics.json"
                    companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(companion_candidate_input),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_companion",
                            companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_companion.log",
                            args.dry_run,
                        )
                    )

                    official_profile_summary_cmd = [
                        python,
                        "-m",
                        "src.scripts.summarize_policy_aligned_profile_metrics",
                        "--candidate",
                        str(companion_candidate_input),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--candidate-meta",
                        str(summary_dir / "backtest_signals_meta_ensemble_decision_aligned_meta.json"),
                        "--profile-id",
                        str(run_profile_id),
                        "--profile-name",
                        str(run_profile_name),
                        "--run-id",
                        str(run_dir.name),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(official_profile_summary_path),
                    ]
                    results.append(
                        _run_step(
                            "official_profile_policy_metrics",
                            official_profile_summary_cmd,
                            logs_dir / "official_profile_policy_metrics.log",
                            args.dry_run,
                        )
                    )

                    if weak_veto_enabled_official and weak_veto_official_mode == "midband":
                        paper_longitudinal_output_path = args.run_root / "midband_shadow_longitudinal.json"
                        paper_longitudinal_cmd = [
                            python,
                            "-m",
                            "src.scripts.update_midband_shadow_longitudinal",
                            "--run-id",
                            str(run_dir.name),
                            "--comparison",
                            str(official_profile_summary_path),
                            "--track",
                            "paper_profile",
                            "--output",
                            str(paper_longitudinal_output_path),
                        ]
                        results.append(
                            _run_step(
                                "midband_paper_profile_longitudinal_update",
                                paper_longitudinal_cmd,
                                logs_dir / "midband_paper_profile_longitudinal_update.log",
                                args.dry_run,
                            )
                        )

                    paired_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(companion_candidate_input),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned",
                            paired_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned.log",
                            args.dry_run,
                        )
                    )

                    overlap_triggered_trade_diag_path = summary_dir / "overlap_triggered_trade_diagnostics.json"
                    companion_candidate_input_path = Path(str(companion_candidate_input))
                    if (
                        labeled_overlap_dataset is not None
                        and labeled_overlap_dataset.exists()
                        and companion_candidate_input_path.exists()
                    ):
                        overlap_triggered_trade_diag_cmd = [
                            python,
                            "-m",
                            "src.scripts.analyze_overlap_triggered_trade_diagnostics",
                            "--candidate",
                            str(companion_candidate_input_path),
                            "--overlap-dataset",
                            str(labeled_overlap_dataset),
                            "--signal-col",
                            str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                            "--return-col",
                            str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                            "--output",
                            str(overlap_triggered_trade_diag_path),
                        ]
                        results.append(
                            _run_step(
                                "overlap_triggered_trade_diagnostics",
                                overlap_triggered_trade_diag_cmd,
                                logs_dir / "overlap_triggered_trade_diagnostics.log",
                                args.dry_run,
                            )
                        )

                    weak_band_regime_diag_path = summary_dir / "weak_band_candidate_only_regime_diagnostics.json"
                    weak_band_regime_diag_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_weak_band_candidate_regime_diagnostics",
                        "--candidate",
                        str(companion_candidate_input),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--incumbent-signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--return-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--p-up-low",
                        str(regime_state_veto_low),
                        "--p-up-high",
                        str(regime_state_veto_high),
                        "--high-inclusive",
                        "1" if regime_state_veto_high_inclusive else "0",
                        "--use-midband-slice",
                        "1" if regime_state_veto_use_midband_slice else "0",
                        "--min-abs-ret-pred",
                        str(regime_state_veto_min_abs_ret_pred),
                        "--max-abs-ret-pred",
                        str(regime_state_veto_max_abs_ret_pred),
                        "--min-regime-rows",
                        str(regime_state_veto_min_regime_rows),
                        "--output",
                        str(weak_band_regime_diag_path),
                    ]
                    results.append(
                        _run_step(
                            "weak_band_candidate_only_regime_diagnostics",
                            weak_band_regime_diag_cmd,
                            logs_dir / "weak_band_candidate_only_regime_diagnostics.log",
                            args.dry_run,
                        )
                    )

                    selected_regime_states = list(regime_state_veto_override)
                    if (not selected_regime_states) and weak_band_regime_diag_path.exists() and (not args.dry_run):
                        try:
                            regime_diag_payload = _load_json(weak_band_regime_diag_path)
                            selected_obj = regime_diag_payload.get("selected_harmful_regimes", [])
                            if isinstance(selected_obj, list):
                                selected_regime_states = [
                                    str(value).strip().lower()
                                    for value in selected_obj
                                    if str(value).strip()
                                ]
                        except Exception as exc:
                            print(
                                f"Warning: failed to load selected harmful regimes from {weak_band_regime_diag_path}: {exc}",
                                file=sys.stderr,
                            )

                    selected_high_volatility_threshold = None
                    if chop_high_vol_veto_min_volatility_cfg is not None:
                        selected_high_volatility_threshold = float(chop_high_vol_veto_min_volatility_cfg)
                    elif weak_band_regime_diag_path.exists() and (not args.dry_run):
                        try:
                            regime_diag_payload = _load_json(weak_band_regime_diag_path)
                            vol_rule_obj = regime_diag_payload.get("selected_high_volatility_rule", {})
                            if isinstance(vol_rule_obj, dict):
                                threshold_obj = vol_rule_obj.get("threshold")
                                if threshold_obj is not None:
                                    threshold_value = float(threshold_obj)
                                    if np.isfinite(threshold_value):
                                        selected_high_volatility_threshold = threshold_value
                        except Exception as exc:
                            print(
                                f"Warning: failed to load selected high-volatility threshold from {weak_band_regime_diag_path}: {exc}",
                                file=sys.stderr,
                            )
                    volatility_only_min_volatility = selected_high_volatility_threshold
                    if volatility_only_veto_min_volatility_cfg is not None:
                        volatility_only_min_volatility = float(volatility_only_veto_min_volatility_cfg)

                    broad_shadow_candidate_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_veto.csv"
                    broad_shadow_meta_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_veto_meta.json"
                    broad_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(broad_shadow_candidate_path),
                        "--meta-output",
                        str(broad_shadow_meta_path),
                    ]
                    broad_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        broad_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        broad_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    broad_shadow_build_cmd.extend(
                        [
                            "--weak-band-candidate-only-veto",
                            "1",
                            "--weak-band-pup-low",
                            str(weak_veto_low),
                            "--weak-band-pup-high",
                            str(weak_veto_high),
                            "--weak-band-high-inclusive",
                            "1" if weak_veto_high_inclusive else "0",
                            "--weak-band-incumbent-reference",
                            str(companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_veto_candidate_band",
                            broad_shadow_build_cmd,
                            logs_dir / "build_shadow_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    broad_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(broad_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_veto_companion",
                            broad_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    broad_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(broad_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_veto",
                            broad_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_veto.log",
                            args.dry_run,
                        )
                    )

                    refined_shadow_candidate_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_refined_veto.csv"
                    refined_shadow_meta_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_refined_veto_meta.json"
                    refined_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(refined_shadow_candidate_path),
                        "--meta-output",
                        str(refined_shadow_meta_path),
                    ]
                    refined_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        refined_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        refined_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    refined_shadow_build_cmd.extend(
                        [
                            "--refined-candidate-only-veto",
                            "1",
                            "--refined-pup-low",
                            str(refined_veto_low),
                            "--refined-pup-high",
                            str(refined_veto_high),
                            "--refined-high-inclusive",
                            "1" if refined_veto_high_inclusive else "0",
                            "--refined-min-abs-ret-pred",
                            str(refined_veto_min_abs_ret_pred),
                            "--refined-incumbent-reference",
                            str(refined_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_refined_veto_candidate_band",
                            refined_shadow_build_cmd,
                            logs_dir / "build_shadow_refined_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    refined_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(refined_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_refined_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_refined_veto_companion",
                            refined_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_refined_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    refined_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(refined_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_refined_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_refined_veto",
                            refined_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_refined_veto.log",
                            args.dry_run,
                        )
                    )

                    midband_shadow_candidate_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_midband_veto.csv"
                    midband_shadow_meta_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_midband_veto_meta.json"
                    midband_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(midband_shadow_candidate_path),
                        "--meta-output",
                        str(midband_shadow_meta_path),
                    ]
                    midband_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        midband_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        midband_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    midband_shadow_build_cmd.extend(
                        [
                            "--midband-candidate-only-veto",
                            "1",
                            "--midband-pup-low",
                            str(midband_veto_low),
                            "--midband-pup-high",
                            str(midband_veto_high),
                            "--midband-high-inclusive",
                            "1" if midband_veto_high_inclusive else "0",
                            "--midband-min-abs-ret-pred",
                            str(midband_veto_min_abs_ret_pred),
                            "--midband-max-abs-ret-pred",
                            str(midband_veto_max_abs_ret_pred),
                            "--midband-incumbent-reference",
                            str(midband_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_midband_veto_candidate_band",
                            midband_shadow_build_cmd,
                            logs_dir / "build_shadow_midband_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    midband_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(midband_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_midband_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_midband_veto_companion",
                            midband_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_midband_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    midband_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(midband_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_midband_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_midband_veto",
                            midband_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_midband_veto.log",
                            args.dry_run,
                        )
                    )

                    raw_ev_shadow_candidate_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_raw_ev_veto.csv"
                    raw_ev_shadow_meta_path = summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_raw_ev_veto_meta.json"
                    raw_ev_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(raw_ev_shadow_candidate_path),
                        "--meta-output",
                        str(raw_ev_shadow_meta_path),
                    ]
                    raw_ev_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        raw_ev_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        raw_ev_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    raw_ev_shadow_build_cmd.extend(
                        [
                            "--raw-ev-sign-candidate-only-veto",
                            "1",
                            "--raw-ev-sign-pup-low",
                            str(raw_ev_veto_low),
                            "--raw-ev-sign-pup-high",
                            str(raw_ev_veto_high),
                            "--raw-ev-sign-high-inclusive",
                            "1" if raw_ev_veto_high_inclusive else "0",
                            "--raw-ev-sign-max",
                            str(raw_ev_veto_max),
                            "--raw-ev-sign-incumbent-reference",
                            str(raw_ev_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_raw_ev_veto_candidate_band",
                            raw_ev_shadow_build_cmd,
                            logs_dir / "build_shadow_raw_ev_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    raw_ev_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(raw_ev_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_raw_ev_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_raw_ev_veto_companion",
                            raw_ev_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_raw_ev_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    raw_ev_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(raw_ev_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_raw_ev_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_raw_ev_veto",
                            raw_ev_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_raw_ev_veto.log",
                            args.dry_run,
                        )
                    )

                    direction_align_shadow_candidate_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_direction_align_veto.csv"
                    )
                    direction_align_shadow_meta_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_direction_align_veto_meta.json"
                    )
                    direction_align_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(direction_align_shadow_candidate_path),
                        "--meta-output",
                        str(direction_align_shadow_meta_path),
                    ]
                    direction_align_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        direction_align_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        direction_align_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    direction_align_shadow_build_cmd.extend(
                        [
                            "--direction-align-candidate-only-veto",
                            "1",
                            "--direction-align-pup-low",
                            str(direction_align_veto_low),
                            "--direction-align-pup-high",
                            str(direction_align_veto_high),
                            "--direction-align-high-inclusive",
                            "1" if direction_align_veto_high_inclusive else "0",
                            "--direction-align-require-aligned",
                            "1" if direction_align_veto_require_aligned else "0",
                            "--direction-align-use-midband-slice",
                            "1" if direction_align_veto_use_midband_slice else "0",
                            "--direction-align-min-abs-ret-pred",
                            str(direction_align_veto_min_abs_ret_pred),
                            "--direction-align-max-abs-ret-pred",
                            str(direction_align_veto_max_abs_ret_pred),
                            "--direction-align-incumbent-reference",
                            str(direction_align_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_direction_align_veto_candidate_band",
                            direction_align_shadow_build_cmd,
                            logs_dir / "build_shadow_direction_align_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    direction_align_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(direction_align_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_direction_align_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_direction_align_veto_companion",
                            direction_align_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_direction_align_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    direction_align_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(direction_align_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_direction_align_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_direction_align_veto",
                            direction_align_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_direction_align_veto.log",
                            args.dry_run,
                        )
                    )

                    joint_direction_midband_shadow_candidate_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_joint_direction_midband_veto.csv"
                    )
                    joint_direction_midband_shadow_meta_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_joint_direction_midband_veto_meta.json"
                    )
                    joint_direction_midband_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(joint_direction_midband_shadow_candidate_path),
                        "--meta-output",
                        str(joint_direction_midband_shadow_meta_path),
                    ]
                    joint_direction_midband_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        joint_direction_midband_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        joint_direction_midband_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    joint_direction_midband_shadow_build_cmd.extend(
                        [
                            "--joint-direction-midband-candidate-only-veto",
                            "1",
                            "--joint-direction-midband-pup-low",
                            str(joint_direction_midband_veto_low),
                            "--joint-direction-midband-pup-high",
                            str(joint_direction_midband_veto_high),
                            "--joint-direction-midband-high-inclusive",
                            "1" if joint_direction_midband_veto_high_inclusive else "0",
                            "--joint-direction-midband-require-aligned",
                            "1" if joint_direction_midband_veto_require_aligned else "0",
                            "--joint-direction-midband-min-abs-ret-pred",
                            str(joint_direction_midband_veto_min_abs_ret_pred),
                            "--joint-direction-midband-max-abs-ret-pred",
                            str(joint_direction_midband_veto_max_abs_ret_pred),
                            "--joint-direction-midband-incumbent-reference",
                            str(joint_direction_midband_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_joint_direction_midband_veto_candidate_band",
                            joint_direction_midband_shadow_build_cmd,
                            logs_dir / "build_shadow_joint_direction_midband_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    joint_direction_midband_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(joint_direction_midband_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_joint_direction_midband_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_joint_direction_midband_veto_companion",
                            joint_direction_midband_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_joint_direction_midband_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    joint_direction_midband_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(joint_direction_midband_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_joint_direction_midband_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_joint_direction_midband_veto",
                            joint_direction_midband_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_joint_direction_midband_veto.log",
                            args.dry_run,
                        )
                    )

                    regime_state_shadow_candidate_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_regime_state_veto.csv"
                    )
                    regime_state_shadow_meta_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_regime_state_veto_meta.json"
                    )
                    regime_state_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(regime_state_shadow_candidate_path),
                        "--meta-output",
                        str(regime_state_shadow_meta_path),
                    ]
                    regime_state_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        regime_state_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        regime_state_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    regime_state_shadow_build_cmd.extend(
                        [
                            "--regime-state-candidate-only-veto",
                            "1",
                            "--regime-state-pup-low",
                            str(regime_state_veto_low),
                            "--regime-state-pup-high",
                            str(regime_state_veto_high),
                            "--regime-state-high-inclusive",
                            "1" if regime_state_veto_high_inclusive else "0",
                            "--regime-state-regimes",
                            ",".join(selected_regime_states),
                            "--regime-state-use-midband-slice",
                            "1" if regime_state_veto_use_midband_slice else "0",
                            "--regime-state-min-abs-ret-pred",
                            str(regime_state_veto_min_abs_ret_pred),
                            "--regime-state-max-abs-ret-pred",
                            str(regime_state_veto_max_abs_ret_pred),
                            "--regime-state-incumbent-reference",
                            str(regime_state_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_regime_state_veto_candidate_band",
                            regime_state_shadow_build_cmd,
                            logs_dir / "build_shadow_regime_state_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    regime_state_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(regime_state_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_regime_state_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_regime_state_veto_companion",
                            regime_state_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_regime_state_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    regime_state_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(regime_state_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_regime_state_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_regime_state_veto",
                            regime_state_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_regime_state_veto.log",
                            args.dry_run,
                        )
                    )

                    chop_high_vol_shadow_candidate_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_chop_high_vol_veto.csv"
                    )
                    chop_high_vol_shadow_meta_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_chop_high_vol_veto_meta.json"
                    )
                    chop_high_vol_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(chop_high_vol_shadow_candidate_path),
                        "--meta-output",
                        str(chop_high_vol_shadow_meta_path),
                    ]
                    chop_high_vol_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        chop_high_vol_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        chop_high_vol_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    chop_high_vol_shadow_build_cmd.extend(
                        [
                            "--chop-high-vol-candidate-only-veto",
                            "1",
                            "--chop-high-vol-pup-low",
                            str(chop_high_vol_veto_low),
                            "--chop-high-vol-pup-high",
                            str(chop_high_vol_veto_high),
                            "--chop-high-vol-high-inclusive",
                            "1" if chop_high_vol_veto_high_inclusive else "0",
                            "--chop-high-vol-regime-state",
                            str(chop_high_vol_veto_regime_state),
                            "--chop-high-vol-volatility-col",
                            str(chop_high_vol_veto_volatility_col),
                            "--chop-high-vol-min-volatility",
                            str(selected_high_volatility_threshold if selected_high_volatility_threshold is not None else "nan"),
                            "--chop-high-vol-use-midband-slice",
                            "1" if chop_high_vol_veto_use_midband_slice else "0",
                            "--chop-high-vol-min-abs-ret-pred",
                            str(chop_high_vol_veto_min_abs_ret_pred),
                            "--chop-high-vol-max-abs-ret-pred",
                            str(chop_high_vol_veto_max_abs_ret_pred),
                            "--chop-high-vol-incumbent-reference",
                            str(chop_high_vol_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_chop_high_vol_veto_candidate_band",
                            chop_high_vol_shadow_build_cmd,
                            logs_dir / "build_shadow_chop_high_vol_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    chop_high_vol_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(chop_high_vol_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_chop_high_vol_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_chop_high_vol_veto_companion",
                            chop_high_vol_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_chop_high_vol_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    chop_high_vol_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(chop_high_vol_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_chop_high_vol_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_chop_high_vol_veto",
                            chop_high_vol_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_chop_high_vol_veto.log",
                            args.dry_run,
                        )
                    )

                    volatility_only_shadow_candidate_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_volatility_only_veto.csv"
                    )
                    volatility_only_shadow_meta_path = (
                        summary_dir / "backtest_signals_meta_ensemble_decision_aligned_shadow_volatility_only_veto_meta.json"
                    )
                    volatility_only_shadow_build_cmd = [
                        python,
                        "-m",
                        "src.scripts.apply_trade_decision_policy_to_backtest",
                        "--input",
                        str(companion_candidate_input),
                        "--model",
                        str(trade_decision_model_path),
                        "--output",
                        str(volatility_only_shadow_candidate_path),
                        "--meta-output",
                        str(volatility_only_shadow_meta_path),
                    ]
                    volatility_only_shadow_build_cmd.extend(common_align_args)
                    if quality_input.exists() or args.dry_run:
                        volatility_only_shadow_build_cmd.extend(["--feature-source", str(quality_input)])
                    if incumbent_backtest_for_features:
                        volatility_only_shadow_build_cmd.extend(["--feature-source", str(incumbent_backtest_for_features)])
                    volatility_only_shadow_build_cmd.extend(
                        [
                            "--volatility-only-candidate-only-veto",
                            "1",
                            "--volatility-only-pup-low",
                            str(volatility_only_veto_low),
                            "--volatility-only-pup-high",
                            str(volatility_only_veto_high),
                            "--volatility-only-high-inclusive",
                            "1" if volatility_only_veto_high_inclusive else "0",
                            "--volatility-only-volatility-col",
                            str(volatility_only_veto_volatility_col),
                            "--volatility-only-min-volatility",
                            str(volatility_only_min_volatility if volatility_only_min_volatility is not None else "nan"),
                            "--volatility-only-use-midband-slice",
                            "1" if volatility_only_veto_use_midband_slice else "0",
                            "--volatility-only-min-abs-ret-pred",
                            str(volatility_only_veto_min_abs_ret_pred),
                            "--volatility-only-max-abs-ret-pred",
                            str(volatility_only_veto_max_abs_ret_pred),
                            "--volatility-only-incumbent-reference",
                            str(volatility_only_veto_reference_path_cfg or companion_baseline_input),
                        ]
                    )
                    results.append(
                        _run_step(
                            "build_shadow_volatility_only_veto_candidate_band",
                            volatility_only_shadow_build_cmd,
                            logs_dir / "build_shadow_volatility_only_veto_candidate_band.log",
                            args.dry_run,
                        )
                    )

                    volatility_only_shadow_companion_cmd = [
                        python,
                        "-m",
                        "src.scripts.evaluate_champion_challenger",
                        "--baseline",
                        str(companion_baseline_input),
                        "--candidate",
                        str(volatility_only_shadow_candidate_path),
                        "--baseline-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--n-boot",
                        str(int(champ_cfg.get("n_boot", 2000))),
                        "--alpha",
                        str(float(champ_cfg.get("alpha", 0.05))),
                        "--seed",
                        str(int(champ_cfg.get("seed", 42))),
                        "--output",
                        str(summary_dir / "champion_challenger_policy_aligned_shadow_volatility_only_veto_companion.json"),
                    ]
                    results.append(
                        _run_step(
                            "champion_challenger_policy_aligned_shadow_volatility_only_veto_companion",
                            volatility_only_shadow_companion_cmd,
                            logs_dir / "champion_challenger_policy_aligned_shadow_volatility_only_veto_companion.log",
                            args.dry_run,
                        )
                    )

                    volatility_only_shadow_overlap_cmd = [
                        python,
                        "-m",
                        "src.scripts.analyze_paired_trigger_overlap",
                        "--candidate",
                        str(volatility_only_shadow_candidate_path),
                        "--incumbent",
                        str(companion_baseline_input),
                        "--candidate-col",
                        str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                        "--incumbent-col",
                        str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                        "--signal-col",
                        str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                        "--output",
                        str(summary_dir / "paired_trigger_overlap_policy_aligned_shadow_volatility_only_veto.json"),
                    ]
                    results.append(
                        _run_step(
                            "paired_trigger_overlap_policy_aligned_shadow_volatility_only_veto",
                            volatility_only_shadow_overlap_cmd,
                            logs_dir / "paired_trigger_overlap_policy_aligned_shadow_volatility_only_veto.log",
                            args.dry_run,
                        )
                    )

                    weak_veto_retro_cfg_obj = weak_veto_cfg.get("retrospective", {})
                    weak_veto_retro_cfg = weak_veto_retro_cfg_obj if isinstance(weak_veto_retro_cfg_obj, dict) else {}
                    if bool(weak_veto_retro_cfg.get("enabled", True)):
                        midband_shadow_comparison_path = summary_dir / "midband_shadow_comparison.json"
                        weak_veto_comparison_cmd = [
                            python,
                            "-m",
                            "src.scripts.evaluate_midband_shadow_retrospective",
                            "--default-candidate",
                            str(companion_candidate_input),
                            "--midband-shadow-candidate",
                            str(midband_shadow_candidate_path),
                            "--incumbent",
                            str(companion_baseline_input),
                            "--candidate-col",
                            str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                            "--incumbent-col",
                            str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                            "--signal-col",
                            str(trade_decision_cfg.get("signal_col", "signal_ensemble")),
                            "--window-size",
                            str(int(weak_veto_retro_cfg.get("window_size", 120))),
                            "--step-size",
                            str(int(weak_veto_retro_cfg.get("step_size", 24))),
                            "--min-rows",
                            str(int(weak_veto_retro_cfg.get("min_rows", 80))),
                            "--n-boot",
                            str(int(weak_veto_retro_cfg.get("n_boot", 1000))),
                            "--seed",
                            str(int(weak_veto_retro_cfg.get("seed", champ_cfg.get("seed", 42)))),
                            "--midband-shadow-meta",
                            str(midband_shadow_meta_path),
                            "--output",
                            str(midband_shadow_comparison_path),
                        ]
                        results.append(
                            _run_step(
                                "midband_shadow_retrospective_comparison",
                                weak_veto_comparison_cmd,
                                logs_dir / "midband_shadow_retrospective_comparison.log",
                                args.dry_run,
                            )
                        )

                        longitudinal_output_cfg = weak_veto_retro_cfg.get("longitudinal_output")
                        if isinstance(longitudinal_output_cfg, str) and longitudinal_output_cfg.strip():
                            longitudinal_output_path = Path(longitudinal_output_cfg)
                        else:
                            longitudinal_output_path = args.run_root / "midband_shadow_longitudinal.json"
                        if not longitudinal_output_path.is_absolute():
                            longitudinal_output_path = Path.cwd() / longitudinal_output_path

                        midband_longitudinal_cmd = [
                            python,
                            "-m",
                            "src.scripts.update_midband_shadow_longitudinal",
                            "--run-id",
                            str(run_dir.name),
                            "--comparison",
                            str(midband_shadow_comparison_path),
                            "--track",
                            "shadow_retrospective",
                            "--output",
                            str(longitudinal_output_path),
                        ]
                        results.append(
                            _run_step(
                                "midband_shadow_longitudinal_update",
                                midband_longitudinal_cmd,
                                logs_dir / "midband_shadow_longitudinal_update.log",
                                args.dry_run,
                            )
                        )

                champ_cmd = [
                    python,
                    "-m",
                    "src.scripts.evaluate_champion_challenger",
                    "--baseline",
                    str(baseline_input),
                    "--candidate",
                    str(candidate_input),
                    "--baseline-col",
                    str(champ_cfg.get("baseline_col", "ret_ensemble_net")),
                    "--candidate-col",
                    str(champ_cfg.get("candidate_col", "ret_ensemble_net")),
                    "--n-boot",
                    str(int(champ_cfg.get("n_boot", 2000))),
                    "--alpha",
                    str(float(champ_cfg.get("alpha", 0.05))),
                    "--seed",
                    str(int(champ_cfg.get("seed", 42))),
                    "--output",
                    str(summary_dir / "champion_challenger_gate.json"),
                ]
                results.append(
                    _run_step(
                        "champion_challenger_gate",
                        champ_cmd,
                        logs_dir / "champion_challenger_gate.log",
                        args.dry_run,
                    )
                )
                gate_path = summary_dir / "champion_challenger_gate.json"
                if gate_path.exists() and not args.dry_run:
                    champion_gate_payload = _load_json(gate_path)

        champion_blocked = bool(
            champion_gate_payload
            and bool(champ_cfg.get("enforce", False))
            and not bool(champion_gate_payload.get("promote", False))
        )

        if incumbent_quality_path and (candidate_quality_path.exists() or args.dry_run):
            if champion_blocked and not args.dry_run:
                synthetic_gate = {
                    "promote": False,
                    "reason": "champion_challenger_blocked",
                    "champion_challenger": champion_gate_payload,
                }
                gate_out = summary_dir / "promotion_gate.json"
                gate_out.write_text(json.dumps(synthetic_gate, indent=2), encoding="utf-8")
                print("Promotion gate blocked by champion-challenger significance gate.", file=sys.stderr)
            else:
                promote_cmd = [
                    python,
                    "-m",
                    "src.scripts.evaluate_shadow_promotion",
                    "--incumbent",
                    str(incumbent_quality_path),
                    "--candidate",
                    str(candidate_quality_path),
                    "--min-auc-delta",
                    str(float(quality_cfg.get("min_auc_delta", 0.002))),
                    "--max-brier-increase",
                    str(float(quality_cfg.get("max_brier_increase", 0.0))),
                    "--max-ece-increase",
                    str(float(quality_cfg.get("max_ece_increase", 0.01))),
                    "--min-trade-count",
                    str(int(quality_cfg.get("min_trade_count", 10))),
                    "--trade-count-key",
                    str(quality_cfg.get("trade_count_key", "trade_count")),
                    "--min-net-return",
                    str(float(quality_cfg.get("min_net_return", 0.0))),
                    "--net-return-key",
                    str(quality_cfg.get("net_return_key", "net_return_total")),
                    "--output",
                    str(summary_dir / "promotion_gate.json"),
                ]
                if bool(quality_cfg.get("require_champion_significance", False)) and bool(champ_cfg.get("enabled", False)):
                    promote_cmd.extend(
                        [
                            "--champion-gate",
                            str(summary_dir / "champion_challenger_gate.json"),
                        ]
                    )
                promotion_allowed = [0, 3] if args.continue_on_promotion_fail else [0]
                results.append(
                    _run_step(
                        "promotion_gate",
                        promote_cmd,
                        logs_dir / "promotion_gate.log",
                        args.dry_run,
                        allowed_returncodes=promotion_allowed,
                    )
                )
        elif not args.dry_run:
            # Emit an explicit non-evaluated promotion artifact so downstream
            # matched-cycle summaries can compare promotion evidence on both profiles.
            if not incumbent_quality_path:
                skip_reason = "not_evaluated_incumbent_quality_missing"
            elif not candidate_quality_path.exists():
                skip_reason = "not_evaluated_candidate_quality_missing"
            else:
                skip_reason = "not_evaluated"

            skipped_gate = {
                "promote": False,
                "evaluated": False,
                "reason": skip_reason,
                "incumbent_quality_path": str(incumbent_quality_path) if incumbent_quality_path else None,
                "candidate_quality_path": str(candidate_quality_path),
                "candidate_quality_exists": bool(candidate_quality_path.exists()),
            }
            (summary_dir / "promotion_gate.json").write_text(
                json.dumps(skipped_gate, indent=2),
                encoding="utf-8",
            )

        # If quality indicates a no-trade or low-trade regime, trigger a relaxed threshold search for paper-live continuity.
        no_trade_regime = {
            "detected": False,
            "trigger_trade_count": None,
            "min_trade_count": int(quality_cfg.get("min_trade_count", 10)),
            "used_fallback_thresholds": False,
            "fallback_thresholds_path": None,
        }
        quality_candidate_path = summary_dir / "model_quality_candidate.json"
        if bool(no_trade_cfg.get("enabled", True)) and (quality_candidate_path.exists() or args.dry_run):
            trigger_below = int(no_trade_cfg.get("trigger_trade_count_below", quality_cfg.get("min_trade_count", 10)))
            trade_count = 0
            if quality_candidate_path.exists():
                quality_payload = _load_json(quality_candidate_path)
                trade_count = int(quality_payload.get("trade_count", 0) or 0)
            no_trade_regime["trigger_trade_count"] = int(trade_count)

            if (args.dry_run or trade_count < trigger_below) and not used_last_deployable_thresholds:
                no_trade_regime["detected"] = True
                fallback_thresholds_path = summary_dir / "calibrated_thresholds_fallback.json"
                fallback_cmd = [
                    python,
                    "-m",
                    "src.scripts.search_ensemble_thresholds",
                    "--targets",
                    _join_horizons(horizons),
                    "--objective",
                    str(no_trade_cfg.get("objective", "cumret_with_dd_constraint")),
                    "--max-dd",
                    str(float(no_trade_cfg.get("max_drawdown", -0.15))),
                    "--min-trades",
                    str(int(no_trade_cfg.get("min_trades", 1))),
                    f"--p-up-grid={str(no_trade_cfg.get('p_up_grid', '0.45,0.50,0.55,0.60'))}",
                    f"--ret-min-grid={str(no_trade_cfg.get('ret_min_grid', '-0.0005,-0.0002,0.0,0.0002'))}",
                    "--output-dir",
                    str(summary_dir / "threshold_search_fallback"),
                    "--output",
                    str(fallback_thresholds_path),
                ]
                results.append(
                    _run_step(
                        "fallback_threshold_search",
                        fallback_cmd,
                        logs_dir / "fallback_threshold_search.log",
                        args.dry_run,
                    )
                )
                thresholds_path = fallback_thresholds_path
                no_trade_regime["used_fallback_thresholds"] = True
                no_trade_regime["fallback_thresholds_path"] = str(fallback_thresholds_path)

            no_trade_regime["joint_threshold_accepted"] = joint_tuning_accepted
            no_trade_regime["used_last_deployable_thresholds"] = bool(used_last_deployable_thresholds)
            no_trade_regime["last_deployable_thresholds_path"] = str(last_deployable_thresholds_path)
            no_trade_regime["paper_live_thresholds_path"] = str(thresholds_path)

        regime_path = summary_dir / "regime_diagnostics.json"
        regime_path.write_text(json.dumps(no_trade_regime, indent=2), encoding="utf-8")

    if not args.skip_monitoring:
        monitoring_horizons = monitoring_cfg.get("horizons", ["1h", "4h", "8h", "12h"])
        monitoring_cmd = [
            python,
            "-m",
            "src.scripts.check_reliability_triggers",
            "--config-path",
            str(args.config),
            "--horizons",
            ",".join(str(h) for h in monitoring_horizons),
            "--output-path",
            str(summary_dir / "reliability_triggers.json"),
        ]
        results.append(_run_step("reliability_trigger_check", monitoring_cmd, logs_dir / "reliability_trigger_check.log", args.dry_run))

    if not args.skip_paper_live:
        paper_live_config = str(search_cfg.get("paper_live_config", "configs/run_refresh_and_predict.default.yaml"))
        paper_cmd = [
            python,
            "-m",
            "src.scripts.run_refresh_and_predict",
            "--config",
            paper_live_config,
            "--targets",
            _join_horizons(horizons),
            "--thresholds-json",
            str(thresholds_path),
            "--platt-calibration",
            str(summary_dir / "platt_calibration.json"),
            "--write-artifacts",
        ]
        if not edge_trustworthy_for_paper_live and not args.dry_run:
            # Force conservative hold behavior when overlap evidence disagrees with canonical walk-forward results.
            paper_cmd.extend(["--p-up-min", "0.99", "--ret-min", "0.005"])
            print(
                "Edge trustworthiness check failed; paper-live run forced to conservative hold thresholds.",
                file=sys.stderr,
            )
        if bool(trade_decision_cfg.get("enabled", True)) and bool(trade_decision_deploy_ready):
            paper_cmd.extend(["--trade-decision-enabled", "--trade-decision-model", str(trade_decision_model_path)])
            if trade_decision_cfg.get("threshold") is not None:
                paper_cmd.extend(["--trade-decision-threshold", str(float(trade_decision_cfg.get("threshold")))])
        elif bool(trade_decision_cfg.get("enabled", True)) and not args.dry_run:
            print(
                "Skipping trade-decision gate in paper-live because trained sample size is below configured minimum.",
                file=sys.stderr,
            )
        if bool(search_cfg.get("paper_live_dry_run", False)):
            paper_cmd.append("--dry-run")
        results.append(_run_step("paper_live_refresh", paper_cmd, logs_dir / "paper_live_refresh.log", args.dry_run))

        # Mark monitoring artifact with no-trade/fallback regime diagnostics.
        regime_path = summary_dir / "regime_diagnostics.json"
        if regime_path.exists() and not args.dry_run:
            try:
                regime_payload = _load_json(regime_path)
                _annotate_monitoring_with_regime(Path("artifacts/monitoring/latest.json"), regime_payload)
            except Exception as exc:  # pragma: no cover - best effort annotation
                print(f"Warning: failed to annotate monitoring latest with regime diagnostics: {exc}", file=sys.stderr)

        live_snapshot_path = summary_dir / "live_predictions_snapshot.json"
        live_snapshot_cmd = [
            python,
            "-m",
            "src.scripts.snapshot_live_predictions",
            "--run-id",
            str(run_dir.name),
            "--profile-id",
            str(run_profile_id),
            "--profile-name",
            str(run_profile_name),
            "--predictions-latest",
            "artifacts/predictions/latest.json",
            "--monitoring-latest",
            "artifacts/monitoring/latest.json",
            "--output",
            str(live_snapshot_path),
        ]
        results.append(
            _run_step(
                "live_prediction_snapshot",
                live_snapshot_cmd,
                logs_dir / "live_prediction_snapshot.log",
                args.dry_run,
            )
        )

        if run_profile_id == "midband_paper_evaluation":
            default_snapshot = _find_latest_profile_snapshot_run(
                run_root=args.run_root,
                current_run_id=str(run_dir.name),
                profile_id="default_runtime",
            )
            if default_snapshot is None:
                print(
                    "Warning: no prior default-runtime live snapshot found; skipping default-vs-midband live comparison.",
                    file=sys.stderr,
                )
            else:
                _, default_snapshot_path = default_snapshot
                live_comparison_cmd = [
                    python,
                    "-m",
                    "src.scripts.compare_default_vs_midband_paper_live_snapshots",
                    "--default-snapshot",
                    str(default_snapshot_path),
                    "--midband-snapshot",
                    str(live_snapshot_path),
                    "--output",
                    str(summary_dir / "default_vs_midband_paper_live_comparison.json"),
                ]
                results.append(
                    _run_step(
                        "default_vs_midband_paper_live_comparison",
                        live_comparison_cmd,
                        logs_dir / "default_vs_midband_paper_live_comparison.log",
                        args.dry_run,
                    )
                )

            longitudinal_live_cmd = [
                python,
                "-m",
                "src.scripts.build_default_vs_midband_paper_live_longitudinal",
                "--run-root",
                str(args.run_root),
                "--include-run-id",
                str(run_dir.name),
                "--include-profile-id",
                str(run_profile_id),
                "--include-profile-name",
                str(run_profile_name),
                "--include-snapshot",
                str(live_snapshot_path),
                "--output",
                str(summary_dir / "default_vs_midband_paper_live_longitudinal.json"),
            ]
            results.append(
                _run_step(
                    "default_vs_midband_paper_live_longitudinal",
                    longitudinal_live_cmd,
                    logs_dir / "default_vs_midband_paper_live_longitudinal.log",
                    args.dry_run,
                )
            )

            watchlist_output = summary_dir / "default_vs_midband_paper_live_watchlist.json"
            watchlist_cmd = [
                python,
                "-m",
                "src.scripts.build_default_vs_midband_paper_live_watchlist",
                "--longitudinal-input",
                str(summary_dir / "default_vs_midband_paper_live_longitudinal.json"),
                "--output",
                str(watchlist_output),
                "--target-matched-pairs",
                "8",
                "--early-operational-streak",
                "2",
                "--early-actionable-asymmetry-streak",
                "2",
            ]
            results.append(
                _run_step(
                    "default_vs_midband_paper_live_watchlist",
                    watchlist_cmd,
                    logs_dir / "default_vs_midband_paper_live_watchlist.log",
                    args.dry_run,
                )
            )

            canonical_watchlist_cmd = [
                python,
                "-m",
                "src.scripts.build_default_vs_midband_paper_live_watchlist",
                "--longitudinal-input",
                str(summary_dir / "default_vs_midband_paper_live_longitudinal.json"),
                "--output",
                str(args.run_root / "default_vs_midband_paper_live_watchlist.json"),
                "--target-matched-pairs",
                "8",
                "--early-operational-streak",
                "2",
                "--early-actionable-asymmetry-streak",
                "2",
            ]
            results.append(
                _run_step(
                    "default_vs_midband_paper_live_watchlist_canonical",
                    canonical_watchlist_cmd,
                    logs_dir / "default_vs_midband_paper_live_watchlist_canonical.log",
                    args.dry_run,
                )
            )

    cadence_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "monthly_retrain_day": int(cadence_cfg.get("monthly_retrain_day", 1)),
        "weekly_recalibration_weekday": str(cadence_cfg.get("weekly_recalibration_weekday", "mon")),
        "trigger_file": str(summary_dir / "reliability_triggers.json"),
        "notes": "Trigger immediate retrain when reliability_triggers.global_trigger = true.",
    }
    cadence_path = summary_dir / "cadence_plan.json"
    cadence_path.write_text(json.dumps(cadence_payload, indent=2), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": str(args.config),
        "profile": {
            "id": run_profile_id,
            "name": run_profile_name,
        },
        "run_dir": str(run_dir),
        "steps": [
            {
                "name": step.name,
                "returncode": step.returncode,
                "log": str(step.log_path),
                "command": step.command,
            }
            for step in results
        ],
        "cadence_plan": str(cadence_path),
    }
    manifest_path = summary_dir / "workflow_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nReliability workflow finished. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
