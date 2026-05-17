from __future__ import annotations

from pathlib import Path

from src.scripts.replay_execution_tier_snapshot import replay_execution_tiers


def test_replay_execution_tiers_reclassifies_old_low_execution_confluence_case(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
execution_policy:
  enabled: true
  short_term_strict_horizons: []
  short_term_min_support_ratio: 0.8
  short_term_min_mid_ratio: 0.8
  immediate_entry_min_support_ratio: 0.8
  immediate_entry_min_mid_ratio: 1.0
  pullback_entry_min_support_ratio: 0.5
  pullback_entry_min_mid_ratio: 0.66
  high_execution_alignment_ratio: 0.5
  medium_execution_alignment_ratio: 0.5
""".strip(),
        encoding="utf-8",
    )
    snapshot_payload = {
        "1h": {
            "horizon_hours": 1.0,
            "direction_next": "up",
            "confluence_support_ratio": 0.7777777777777778,
            "confluence_mid_term_ratio": 1.0,
            "execution_plan": {
                "bias_direction": "up",
                "execution_alignment_ratio": 1.0,
                "confluence_tier": "low",
                "status": "rejected",
                "reason": "low_execution_confluence",
            },
        }
    }

    replay = replay_execution_tiers(snapshot_payload, config_path=config_path)

    assert replay["policy_summary"]["short_term_strict_horizons"] == []
    assert replay["per_horizon"]["1h"]["replayed_confluence_tier"] == "medium"
    assert replay["per_horizon"]["1h"]["low_execution_confluence_cleared"] is True
    assert replay["overall_summary"]["low_execution_confluence_cleared_horizons"] == ["1h"]


def test_replay_execution_tiers_can_extract_profile_from_comparison_snapshot(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
execution_policy:
  enabled: true
  short_term_strict_horizons: []
  pullback_entry_min_support_ratio: 0.5
  pullback_entry_min_mid_ratio: 0.66
  medium_execution_alignment_ratio: 0.5
""".strip(),
        encoding="utf-8",
    )
    raw_snapshot_path = tmp_path / "shadow_snapshot.json"
    raw_snapshot_path.write_text(
        """
{
  "1h": {
    "horizon_hours": 1.0,
    "direction_next": "up",
    "confluence_support_ratio": 0.75,
    "confluence_mid_term_ratio": 1.0,
    "execution_plan": {
      "bias_direction": "up",
      "execution_alignment_ratio": 1.0,
      "confluence_tier": "low",
      "status": "rejected",
      "reason": "low_execution_confluence"
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    comparison_payload = {
        "shadow": {
            "snapshot": str(raw_snapshot_path),
        },
        "per_horizon": {
            "1h": {
                "shadow": {
                    "execution_status": "rejected",
                    "execution_reason": "low_execution_confluence",
                }
            }
        },
    }

    replay = replay_execution_tiers(
        comparison_payload,
        config_path=config_path,
        snapshot_path=tmp_path / "comparison.json",
        profile_label="shadow",
    )

    assert replay["profile_label"] == "shadow"
    assert replay["per_horizon"]["1h"]["replayed_confluence_tier"] == "medium"
