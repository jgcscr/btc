from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from src.scripts.run_refresh_and_predict import _classify_execution_tier, _resolve_execution_policy


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_execution_policy(config_path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"Expected YAML mapping at {config_path}")
    return _resolve_execution_policy(config.get("execution_policy"))


def _is_horizon_label(value: str) -> bool:
    if not isinstance(value, str) or len(value) < 2:
        return False
    suffix = value[-1].lower()
    if suffix not in {"h", "m"}:
        return False
    try:
        float(value[:-1])
    except ValueError:
        return False
    return True


def _extract_snapshot_rows(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None = None,
    profile_label: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    predictions = payload.get("predictions")
    if isinstance(predictions, Mapping):
        return {str(label): dict(row) for label, row in predictions.items() if _is_horizon_label(str(label)) and isinstance(row, Mapping)}

    if profile_label:
        profile_payload = payload.get(profile_label)
        if not isinstance(profile_payload, Mapping):
            raise ValueError(f"Profile label {profile_label!r} was not found in comparison snapshot")
        snapshot_ref = profile_payload.get("snapshot")
        if not isinstance(snapshot_ref, str) or not snapshot_ref.strip():
            raise ValueError(f"Profile label {profile_label!r} does not include a source snapshot path")
        snapshot_path = Path(snapshot_ref)
        if not snapshot_path.is_absolute():
            cwd_candidate = (Path.cwd() / snapshot_path).resolve()
            if cwd_candidate.exists():
                snapshot_path = cwd_candidate
            else:
                base_dir = source_path.parent if source_path is not None else Path.cwd()
                snapshot_path = (base_dir / snapshot_path).resolve()
        resolved_payload = _load_json(snapshot_path)
        return _extract_snapshot_rows(resolved_payload, source_path=snapshot_path, profile_label=None)

    top_level_rows = {
        str(label): dict(row)
        for label, row in payload.items()
        if _is_horizon_label(str(label)) and isinstance(row, Mapping)
    }
    if top_level_rows:
        return top_level_rows

    raise ValueError("Could not locate horizon rows in snapshot payload")


def _sort_key(label: str) -> tuple[float, str]:
    multiplier = 1.0
    if label.endswith("m"):
        multiplier = 1.0 / 60.0
    return (float(label[:-1]) * multiplier, label)


def replay_execution_tiers(
    snapshot_payload: Mapping[str, Any],
    *,
    config_path: Path,
    snapshot_path: Path | None = None,
    profile_label: str | None = None,
) -> Dict[str, Any]:
    policy = _load_execution_policy(config_path)
    rows = _extract_snapshot_rows(snapshot_payload, source_path=snapshot_path, profile_label=profile_label)
    horizons: Dict[str, Any] = {}
    cleared_labels: list[str] = []

    for label in sorted(rows.keys(), key=_sort_key):
        row = rows[label]
        execution_plan = row.get("execution_plan") if isinstance(row.get("execution_plan"), Mapping) else {}
        bias_direction = str(execution_plan.get("bias_direction") or "neutral")
        execution_alignment_ratio = float(execution_plan.get("execution_alignment_ratio") or 0.0)
        replayed_tier = _classify_execution_tier(
            row,
            bias_direction=bias_direction,
            execution_alignment_ratio=execution_alignment_ratio,
            policy=policy,
        )
        stored_tier = execution_plan.get("confluence_tier")
        stored_reason = execution_plan.get("reason")
        low_execution_confluence_cleared = bool(stored_reason == "low_execution_confluence" and replayed_tier != "low")
        if low_execution_confluence_cleared:
            cleared_labels.append(label)
        horizons[label] = {
            "stored_confluence_tier": stored_tier,
            "replayed_confluence_tier": replayed_tier,
            "stored_execution_status": execution_plan.get("status"),
            "stored_execution_reason": stored_reason,
            "bias_direction": bias_direction,
            "execution_alignment_ratio": execution_alignment_ratio,
            "support_ratio": row.get("confluence_support_ratio"),
            "mid_term_ratio": row.get("confluence_mid_term_ratio"),
            "low_execution_confluence_cleared": low_execution_confluence_cleared,
        }

    return {
        "config_path": str(config_path),
        "profile_label": profile_label,
        "policy_summary": {
            "short_term_strict_horizons": policy.get("short_term_strict_horizons", []),
            "short_term_min_support_ratio": policy.get("short_term_min_support_ratio"),
            "short_term_min_support_ratio_by_horizon": policy.get("short_term_min_support_ratio_by_horizon", {}),
            "pullback_entry_min_support_ratio": policy.get("pullback_entry_min_support_ratio"),
            "pullback_entry_min_mid_ratio": policy.get("pullback_entry_min_mid_ratio"),
            "medium_execution_alignment_ratio": policy.get("medium_execution_alignment_ratio"),
        },
        "overall_summary": {
            "horizons_replayed": list(horizons.keys()),
            "low_execution_confluence_cleared_horizons": cleared_labels,
            "cleared_low_execution_confluence_count": len(cleared_labels),
        },
        "per_horizon": horizons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay archived execution-tier classification from a prediction snapshot.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile-label", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_json(args.snapshot)
    replay = replay_execution_tiers(
        payload,
        config_path=args.config,
        snapshot_path=args.snapshot,
        profile_label=args.profile_label,
    )
    rendered = json.dumps(replay, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()