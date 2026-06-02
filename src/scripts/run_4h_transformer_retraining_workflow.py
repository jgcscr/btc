from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


DEFAULT_DATASET_PATH = "artifacts/datasets/btc_features_multi_horizon_splits.npz"
DEFAULT_CANDIDATE_MODEL_DIR = "artifacts/models_4h_candidate/transformer_dir4h_v1"
DEFAULT_OUTPUT_JSON = "artifacts/analysis/4h_transformer_retraining_workflow_latest.json"
DEFAULT_OUTPUT_MD = "artifacts/analysis/4h_transformer_retraining_workflow_latest.md"
INCUMBENT_SUMMARY_PATH = Path("artifacts/models/transformer_dir4h_v1/summary.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a staged 4h transformer retraining workflow with complete summary metrics, "
            "without modifying live artifacts."
        )
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--candidate-model-dir", default=DEFAULT_CANDIDATE_MODEL_DIR)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--preset", choices=("base", "large"), default="large")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-val-gap", type=float, default=0.03)
    parser.add_argument("--min-test-accuracy", type=float, default=0.55)
    parser.add_argument("--min-test-auc", type=float, default=0.52)
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


def _test_metrics(summary: Mapping[str, Any]) -> Dict[str, float | None]:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), Mapping) else {}
    test = metrics.get("test") if isinstance(metrics.get("test"), Mapping) else {}
    return {
        "test_accuracy": _safe_float(test.get("accuracy")),
        "test_auc": _safe_float(test.get("roc_auc") if test.get("roc_auc") is not None else test.get("auc")),
        "test_f1": _safe_float(test.get("f1")),
    }


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _render_markdown(payload: Mapping[str, Any]) -> str:
    inc = payload.get("incumbent", {})
    cand = payload.get("candidate", {})
    lines = ["# 4h Transformer Retraining Workflow", ""]
    lines.append("## Recommendation")
    lines.append(f"- candidate_ready_for_trust_review: {payload.get('candidate_ready_for_trust_review')}")
    lines.append(f"- keep_live_artifacts_unchanged: {payload.get('keep_live_artifacts_unchanged')}")
    lines.append("")
    lines.append("## Incumbent")
    lines.append(f"- Summary path: {inc.get('summary_path')}")
    lines.append(f"- Train/val gap: {inc.get('train_val_gap')}")
    lines.append(f"- Test accuracy: {inc.get('test_accuracy')}")
    lines.append(f"- Test auc: {inc.get('test_auc')}")
    lines.append("")
    lines.append("## Candidate")
    lines.append(f"- Summary path: {cand.get('summary_path')}")
    lines.append(f"- Train/val gap: {cand.get('train_val_gap')}")
    lines.append(f"- Allowed train/val gap: {cand.get('max_train_val_gap')}")
    lines.append(f"- Test accuracy: {cand.get('test_accuracy')}")
    lines.append(f"- Test auc: {cand.get('test_auc')}")
    lines.append(f"- Test f1: {cand.get('test_f1')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root()
    dataset_path = Path(args.dataset_path)
    candidate_model_dir = Path(args.candidate_model_dir)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not INCUMBENT_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Incumbent summary not found: {INCUMBENT_SUMMARY_PATH}")

    candidate_model_dir.mkdir(parents=True, exist_ok=True)
    train_cmd = [
        sys.executable,
        "-m",
        "src.scripts.train_transformer_dir1h",
        "--dataset-path",
        str(dataset_path),
        "--horizon",
        "4",
        "--preset",
        str(args.preset),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--patience",
        str(int(args.patience)),
        "--output-dir",
        str(candidate_model_dir),
    ]
    if args.max_steps is not None:
        train_cmd.extend(["--max-steps", str(int(args.max_steps))])
    _run(train_cmd, cwd=repo_root)

    incumbent_summary = _read_json(INCUMBENT_SUMMARY_PATH)
    candidate_summary = _read_json(candidate_model_dir / "summary.json")

    incumbent_gap = _train_val_gap(incumbent_summary)
    candidate_gap = _train_val_gap(candidate_summary)
    incumbent_test = _test_metrics(incumbent_summary)
    candidate_test = _test_metrics(candidate_summary)

    candidate_ready = (
        candidate_gap is not None
        and candidate_gap <= float(args.max_train_val_gap)
        and candidate_test["test_accuracy"] is not None
        and candidate_test["test_accuracy"] >= float(args.min_test_accuracy)
        and (
            candidate_test["test_auc"] is None
            or candidate_test["test_auc"] >= float(args.min_test_auc)
        )
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dataset_path": str(dataset_path),
            "candidate_model_dir": str(candidate_model_dir),
            "preset": str(args.preset),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "patience": int(args.patience),
            "max_steps": int(args.max_steps) if args.max_steps is not None else None,
            "max_train_val_gap": float(args.max_train_val_gap),
            "min_test_accuracy": float(args.min_test_accuracy),
            "min_test_auc": float(args.min_test_auc),
        },
        "incumbent": {
            "summary_path": str(INCUMBENT_SUMMARY_PATH),
            "train_val_gap": incumbent_gap,
            **incumbent_test,
        },
        "candidate": {
            "summary_path": str(candidate_model_dir / "summary.json"),
            "train_val_gap": candidate_gap,
            "max_train_val_gap": float(args.max_train_val_gap),
            **candidate_test,
        },
        "candidate_ready_for_trust_review": bool(candidate_ready),
        "keep_live_artifacts_unchanged": True,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote workflow JSON: {output_json}")
    print(f"Wrote workflow memo: {output_md}")


if __name__ == "__main__":
    main()