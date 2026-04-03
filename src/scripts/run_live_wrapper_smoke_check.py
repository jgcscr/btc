from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from src.scripts.run_live_inference import DEFAULT_LIVE_CONFIG


REQUIRED_TRUST_FIELDS = {
    "trust_status",
    "trust_reasons",
    "excluded_from_voting",
    "voting_weight_after_trust",
    "trust_hardening_changed_outcome",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _latest_matching_live_run(artifact_root: Path, *, expected_config: str, started_at: float) -> Path:
    candidates: List[Path] = []
    for entry in artifact_root.iterdir():
        if not entry.is_dir() or not entry.name.startswith("live-"):
            continue
        if entry.stat().st_mtime < started_at:
            continue
        request_path = entry / "request.json"
        if not request_path.exists():
            continue
        try:
            request_payload = json.loads(request_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(request_payload.get("config")) == expected_config:
            candidates.append(entry)
    if not candidates:
        raise RuntimeError(
            "No live runtime run matched the wrapper smoke-check config path. "
            "Check artifacts/runtime_runs and wrapper execution output."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _validate_trust_behavior(predictions_payload: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    predictions = predictions_payload.get("predictions") if isinstance(predictions_payload, Mapping) else None
    if not isinstance(predictions, Mapping):
        return ["predictions.json missing top-level 'predictions' mapping"]

    # Required telemetry must exist for every emitted horizon.
    for horizon_label, entry in predictions.items():
        if not isinstance(entry, Mapping):
            errors.append(f"{horizon_label}: prediction entry is not a mapping")
            continue
        missing = REQUIRED_TRUST_FIELDS.difference(entry.keys())
        if missing:
            errors.append(f"{horizon_label}: missing trust fields: {sorted(missing)}")

    four_h = predictions.get("4h")
    eight_h = predictions.get("8h")
    if not isinstance(four_h, Mapping):
        errors.append("4h horizon missing from predictions.json")
    else:
        if four_h.get("trust_status") != "low_trust":
            errors.append("4h trust_status expected low_trust")
        if four_h.get("trust_hardening_action") != "deweight":
            errors.append("4h trust_hardening_action expected deweight")
        if bool(four_h.get("excluded_from_voting")):
            errors.append("4h expected excluded_from_voting=false under deweight")
        try:
            weight = float(four_h.get("voting_weight_after_trust"))
            if abs(weight - 0.5) > 1e-9:
                errors.append("4h voting_weight_after_trust expected 0.5")
        except (TypeError, ValueError):
            errors.append("4h voting_weight_after_trust is not numeric")

    if isinstance(eight_h, Mapping):
        errors.append("8h horizon should not be emitted for the current live policy")

    missing_metadata_hits = 0
    for entry in predictions.values():
        if not isinstance(entry, Mapping):
            continue
        for reason in entry.get("trust_reasons") or []:
            if str(reason) == "missing_required_trust_metadata":
                missing_metadata_hits += 1
    if missing_metadata_hits > 0:
        errors.append(f"unexpected missing_required_trust_metadata hits: {missing_metadata_hits}")

    return errors


def run_smoke_check(config_path: Path, *, timeout_seconds: int, artifact_root: Path) -> Dict[str, Any]:
    repo_root = _repo_root()
    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg["dry_run"] = True

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        tmp_config = Path(handle.name)

    started_at = time.time()
    cmd = [sys.executable, "-m", "src.scripts.run_live_inference", "--config", str(tmp_config)]
    env = dict(os.environ)
    env["PYTHONPATH"] = ".:src"

    try:
        run_result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        run_dir = _latest_matching_live_run(
            artifact_root,
            expected_config=str(tmp_config),
            started_at=started_at,
        )

        summary_path = run_dir / "summary.json"
        events_path = run_dir / "events.jsonl"
        predictions_path = run_dir / "predictions.json"
        monitoring_path = run_dir / "monitoring.json"

        errors: List[str] = []
        if run_result.returncode != 0:
            errors.append(f"wrapper process exited with non-zero code: {run_result.returncode}")

        if not summary_path.exists():
            errors.append("summary.json missing in runtime run")
            summary_payload: Dict[str, Any] = {}
        else:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_payload.get("status") != "succeeded":
                errors.append(f"summary status is not succeeded: {summary_payload.get('status')}")

        prediction_stage_completed = False
        if not events_path.exists():
            errors.append("events.jsonl missing in runtime run")
        else:
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("stage") == "prediction" and event.get("status") == "completed":
                    prediction_stage_completed = True
                    break
            if not prediction_stage_completed:
                errors.append("prediction stage did not complete")

        if not predictions_path.exists():
            errors.append("predictions.json missing in runtime run")
            predictions_payload: Dict[str, Any] = {}
        else:
            predictions_payload = json.loads(predictions_path.read_text(encoding="utf-8"))
            errors.extend(_validate_trust_behavior(predictions_payload))

        if not monitoring_path.exists():
            errors.append("monitoring.json missing in runtime run")

        return {
            "ok": not errors,
            "errors": errors,
            "run_dir": str(run_dir),
            "config_used": str(tmp_config),
            "wrapper_return_code": run_result.returncode,
            "stdout_tail": run_result.stdout[-2000:],
            "stderr_tail": run_result.stderr[-2000:],
            "prediction_stage_completed": prediction_stage_completed,
        }
    finally:
        try:
            tmp_config.unlink(missing_ok=True)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke check: default BTC live wrapper reaches prediction stage and emits trust telemetry.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_LIVE_CONFIG),
        help="Base live config. Smoke check forces dry_run=true in a temporary copy.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=240,
        help="Timeout for wrapper execution.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/runtime_runs"),
        help="Runtime run artifact directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_smoke_check(
        args.config,
        timeout_seconds=int(args.timeout_seconds),
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result, indent=2))
    if not result.get("ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
