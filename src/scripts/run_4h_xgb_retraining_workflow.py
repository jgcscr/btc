from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


DEFAULT_DATASET_PATH = "artifacts/datasets/btc_features_multi_horizon_splits.npz"
DEFAULT_SEARCH_OUTPUT_DIR = "artifacts/analysis/4h_xgb_optuna_search"
DEFAULT_CANDIDATE_MODEL_ROOT = "artifacts/models_4h_candidate"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/4h_xgb_retraining_workflow_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/4h_xgb_retraining_workflow_latest.md"
INCUMBENT_SUMMARY_PATH = Path("artifacts/models/xgb_dir4h_v1/summary.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a focused 4h XGB remediation workflow: Optuna search, candidate retrain, "
            "4h calibration fit, and incumbent-vs-candidate reporting."
        )
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--search-output-dir", default=DEFAULT_SEARCH_OUTPUT_DIR)
    parser.add_argument("--candidate-model-root", default=DEFAULT_CANDIDATE_MODEL_ROOT)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--cv-train-size", type=int, default=1500)
    parser.add_argument("--cv-val-size", type=int, default=300)
    parser.add_argument("--cv-test-size", type=int, default=300)
    parser.add_argument("--cv-gap", type=int, default=24)
    parser.add_argument("--cv-mode", choices=("expanding", "rolling"), default="expanding")
    parser.add_argument("--max-train-val-gap", type=float, default=0.03)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _train_val_gap(summary: Mapping[str, Any]) -> float | None:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), Mapping) else {}
    train = metrics.get("train") if isinstance(metrics.get("train"), Mapping) else {}
    val = metrics.get("val") if isinstance(metrics.get("val"), Mapping) else {}
    train_acc = _safe_float(train.get("accuracy"))
    val_acc = _safe_float(val.get("accuracy"))
    if train_acc is None or val_acc is None:
        return None
    return abs(train_acc - val_acc)


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _render_markdown(payload: Mapping[str, Any]) -> str:
    inc = payload.get("incumbent", {})
    cand = payload.get("candidate", {})
    search = payload.get("search", {})
    lines = ["# 4h XGB Retraining Workflow", ""]
    lines.append("## Recommendation")
    lines.append(f"- Candidate ready for trust review: {payload.get('candidate_ready_for_trust_review')}")
    lines.append(f"- Keep live artifacts unchanged: {payload.get('keep_live_artifacts_unchanged')}")
    lines.append("")
    lines.append("## Search")
    lines.append(f"- Trials: {search.get('n_trials')}")
    lines.append(f"- Best validation objective: {search.get('best_value')}")
    lines.append(f"- Candidate walkforward test AUC: {((search.get('summary') or {}).get('test_metrics') or {}).get('test_auc')}")
    lines.append("")
    lines.append("## Incumbent")
    lines.append(f"- Summary path: {inc.get('summary_path')}")
    lines.append(f"- Train/val gap: {inc.get('train_val_gap')}")
    lines.append(f"- Test accuracy: {inc.get('test_accuracy')}")
    lines.append("")
    lines.append("## Candidate")
    lines.append(f"- Summary path: {cand.get('summary_path')}")
    lines.append(f"- Train/val gap: {cand.get('train_val_gap')}")
    lines.append(f"- Allowed train/val gap: {cand.get('max_train_val_gap')}")
    lines.append(f"- Test accuracy: {cand.get('test_accuracy')}")
    lines.append(f"- Calibration path: {cand.get('calibration_path')}")
    lines.append("")
    lines.append("## Notes")
    for note in payload.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root()
    dataset_path = Path(args.dataset_path)
    search_output_dir = Path(args.search_output_dir)
    candidate_model_root = Path(args.candidate_model_root)
    candidate_model_dir = candidate_model_root / "xgb_dir4h_v1"
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not INCUMBENT_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Incumbent summary not found: {INCUMBENT_SUMMARY_PATH}")

    search_output_dir.mkdir(parents=True, exist_ok=True)
    candidate_model_dir.mkdir(parents=True, exist_ok=True)

    search_cmd = [
        sys.executable,
        "-m",
        "src.scripts.search_xgb_optuna",
        "--mode",
        "dir",
        "--horizon",
        "4",
        "--dataset-path",
        str(dataset_path),
        "--n-trials",
        str(int(args.n_trials)),
        "--output-dir",
        str(search_output_dir),
        "--cv-folds",
        str(int(args.cv_folds)),
        "--cv-train-size",
        str(int(args.cv_train_size)),
        "--cv-val-size",
        str(int(args.cv_val_size)),
        "--cv-test-size",
        str(int(args.cv_test_size)),
        "--cv-gap",
        str(int(args.cv_gap)),
        "--cv-mode",
        str(args.cv_mode),
    ]
    if args.timeout is not None:
        search_cmd.extend(["--timeout", str(float(args.timeout))])
    _run(search_cmd, cwd=repo_root)

    search_summary = _read_json(search_output_dir / "summary.json")
    best_params = search_summary.get("best_params")
    if not isinstance(best_params, Mapping):
        raise ValueError("Optuna search did not produce best_params")
    params_path = search_output_dir / "best_params.json"
    params_path.write_text(json.dumps(dict(best_params), indent=2), encoding="utf-8")

    train_cmd = [
        sys.executable,
        "-m",
        "src.scripts.train_xgb_dir4h_v1",
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(candidate_model_dir),
        "--horizon",
        "4",
        "--params-json",
        str(params_path),
    ]
    _run(train_cmd, cwd=repo_root)

    calibration_path = candidate_model_root / "platt_calibration_4h_candidate.json"
    calibrate_cmd = [
        sys.executable,
        "-m",
        "src.scripts.train_platt_calibration",
        "--model-root",
        str(candidate_model_root),
        "--dataset-multi",
        str(dataset_path),
        "--horizons",
        "4",
        "--output-path",
        str(calibration_path),
    ]
    _run(calibrate_cmd, cwd=repo_root)

    incumbent_summary = _read_json(INCUMBENT_SUMMARY_PATH)
    candidate_summary = _read_json(candidate_model_dir / "summary.json")
    incumbent_gap = _train_val_gap(incumbent_summary)
    candidate_gap = _train_val_gap(candidate_summary)

    incumbent_test_accuracy = _safe_float(((incumbent_summary.get("metrics") or {}).get("test") or {}).get("accuracy"))
    candidate_test_accuracy = _safe_float(((candidate_summary.get("metrics") or {}).get("test") or {}).get("accuracy"))

    candidate_ready = (
        candidate_gap is not None
        and incumbent_gap is not None
        and candidate_gap < incumbent_gap
        and candidate_gap <= float(args.max_train_val_gap)
        and candidate_test_accuracy is not None
        and incumbent_test_accuracy is not None
        and candidate_test_accuracy >= incumbent_test_accuracy
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dataset_path": str(dataset_path),
            "n_trials": int(args.n_trials),
            "timeout": float(args.timeout) if args.timeout is not None else None,
            "cv_folds": int(args.cv_folds),
            "cv_train_size": int(args.cv_train_size),
            "cv_val_size": int(args.cv_val_size),
            "cv_test_size": int(args.cv_test_size),
            "cv_gap": int(args.cv_gap),
            "cv_mode": str(args.cv_mode),
            "max_train_val_gap": float(args.max_train_val_gap),
        },
        "search": {
            "output_dir": str(search_output_dir),
            "n_trials": int(args.n_trials),
            "best_value": search_summary.get("best_value"),
            "summary": search_summary,
            "best_params_path": str(params_path),
        },
        "incumbent": {
            "summary_path": str(INCUMBENT_SUMMARY_PATH),
            "train_val_gap": incumbent_gap,
            "test_accuracy": incumbent_test_accuracy,
        },
        "candidate": {
            "model_root": str(candidate_model_root),
            "summary_path": str(candidate_model_dir / "summary.json"),
            "train_val_gap": candidate_gap,
            "max_train_val_gap": float(args.max_train_val_gap),
            "test_accuracy": candidate_test_accuracy,
            "calibration_path": str(calibration_path),
        },
        "candidate_ready_for_trust_review": bool(candidate_ready),
        "keep_live_artifacts_unchanged": True,
        "notes": [
            "This workflow stages a separate 4h XGB candidate and calibration artifact without modifying live model directories.",
            "Trust review still requires the candidate train/val gap to clear the configured absolute threshold and a later live-style replay to inspect probability divergence.",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(payload), encoding="utf-8")

    print(f"Wrote workflow JSON: {output_json}")
    print(f"Wrote workflow memo: {output_md}")


if __name__ == "__main__":
    main()