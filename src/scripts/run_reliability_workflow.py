from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

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

    python = sys.executable

    if not args.skip_optuna:
        for horizon in horizons:
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
                str(cv_folds),
                "--cv-train-size",
                str(cv_train_size),
                "--cv-val-size",
                str(cv_val_size),
                "--cv-test-size",
                str(cv_test_size),
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
                    str(cv_folds),
                    "--cv-train-size",
                    str(cv_train_size),
                    "--cv-val-size",
                    str(cv_val_size),
                    "--cv-test-size",
                    str(cv_test_size),
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
                    str(cv_folds),
                    "--cv-train-size",
                    str(cv_train_size),
                    "--cv-val-size",
                    str(cv_val_size),
                    "--cv-test-size",
                    str(cv_test_size),
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
        ]
        results.append(_run_step("platt_calibration", calibr_cmd, logs_dir / "platt_calibration.log", args.dry_run))

    thresholds_path = summary_dir / "calibrated_thresholds.json"
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

        if bool(quality_cfg.get("build_labeled_dataset", True)):
            labeled_cmd = [
                python,
                "-m",
                "src.scripts.build_labeled_backtest_from_history",
                "--output",
                str(quality_input),
                "--meta-output",
                str(summary_dir / "labeled_backtest_meta.json"),
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

        walkforward_horizon = int(quality_cfg.get("walkforward_horizon", 1))
        walkforward_dataset = Path(
            quality_cfg.get("walkforward_dataset") or _direction_dataset_for_horizon(walkforward_horizon)
        )
        walkforward_target = str(quality_cfg.get("walkforward_y_key") or "y")
        if walkforward_dataset.exists() or args.dry_run:
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
                "--output",
                str(summary_dir / "walkforward_validation.json"),
            ]
            results.append(_run_step("walkforward_validation", walkforward_cmd, logs_dir / "walkforward_validation.log", args.dry_run))

        if quality_input.exists() or args.dry_run:
            quality_cmd = [
                python,
                "-m",
                "src.scripts.evaluate_model_quality",
                "--input",
                str(quality_input),
                "--output",
                str(summary_dir / "model_quality_candidate.json"),
            ]
            results.append(_run_step("model_quality_eval", quality_cmd, logs_dir / "model_quality_eval.log", args.dry_run))

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
                tuning_cmd = [
                    python,
                    "-m",
                    "src.scripts.tune_joint_signal_thresholds",
                    "--input",
                    str(quality_input),
                    f"--p-up-grid={str(tuning_cfg.get('p_up_grid', '0.50,0.55,0.60,0.65'))}",
                    f"--ret-min-grid={str(tuning_cfg.get('ret_min_grid', '-0.0002,0.0,0.0002,0.0005'))}",
                    f"--direction-threshold-grid={str(tuning_cfg.get('direction_threshold_grid', '0.50,0.55,0.60'))}",
                    "--min-trades",
                    str(int(tuning_cfg.get("min_trades", 10))),
                    "--max-dd",
                    str(float(tuning_cfg.get("max_drawdown", -0.12))),
                    "--output",
                    str(summary_dir / "joint_threshold_tuning.json"),
                ]
                results.append(_run_step("joint_threshold_tuning", tuning_cmd, logs_dir / "joint_threshold_tuning.log", args.dry_run))

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
        if incumbent_quality_path and (candidate_quality_path.exists() or args.dry_run):
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
                "--output",
                str(summary_dir / "promotion_gate.json"),
            ]
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

            if args.dry_run or trade_count < trigger_below:
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
