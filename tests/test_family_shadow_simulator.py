from __future__ import annotations

import unittest

import pandas as pd

from src.runtime.family_shadow_simulator import (
    FamilyShadowPolicy,
    default_family_policy_variants,
    replay_snapshot_with_family_shadow,
    resolve_family_snapshot_state,
    run_state_order_flow_shadow_validation,
)


def _snapshot_template(
    trade_action: str = "long",
    confidence: float = 0.55,
    expected_value: float = 0.0004,
    regime: str = "neutral",
) -> dict:
    return {
        "generated_at": "2026-04-03T10:00:00Z",
        "prompt_ready_summary": {
            "market_outlook_strategy": {
                "preferred_horizon": "4h",
                "selected_direction": "Long",
                "execution_state": "ready",
                "pending_trade_action": "long",
                "tradeable": True,
            }
        },
        "predictions": {
            "4h": {
                "trade_action": trade_action,
                "execution_score": 1.4,
                "execution_plan": {"status": "ready", "reason": "pass"},
                "confidence_score": confidence,
                "expected_value": expected_value,
                "regime_state": regime,
            },
            "12h": {
                "trade_action": trade_action,
                "execution_score": 1.2,
                "execution_plan": {"status": "ready", "reason": "pass"},
                "confidence_score": confidence,
                "expected_value": expected_value,
                "regime_state": regime,
            },
        },
    }


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-04-03T09:00:00Z"], utc=True),
            "trend_path_efficiency_4h": [0.8],
            "trend_path_efficiency_8h": [0.7],
            "trend_directional_persistence_4h": [0.9],
            "trend_directional_persistence_8h": [0.85],
            "range_compression_ratio_4h_24h": [0.7],
            "range_compression_transition_8h": [-0.4],
            "price_distance_to_high_atr_24h": [4.0],
            "price_distance_to_low_atr_24h": [1.0],
            "price_distance_to_high_pct_rank_24h": [0.9],
            "price_distance_to_low_pct_rank_24h": [0.2],
            "volume_regime_zscore_24h": [0.3],
            "cvd_ratio_6h": [-0.5],
            "cvd_zscore_6h": [-1.2],
            "trades_taker_imbalance_acceleration_3h": [-0.3],
            "trades_taker_imbalance_persistence_6h": [0.8],
            "interaction_imbalance_trend_6h": [-0.3],
            "interaction_breakout_volume_8h": [0.1],
            "vwap_deviation_8h": [-0.02],
        }
    )


class FamilyShadowSimulatorTests(unittest.TestCase):
    def test_state_weak_veto_blocks_conflict(self) -> None:
        snapshot = _snapshot_template(trade_action="long", confidence=0.5, expected_value=0.0002)
        state = resolve_family_snapshot_state(
            snapshot_ts=pd.Timestamp("2026-04-03T10:00:00Z"),
            feature_frame=_feature_frame(),
            max_staleness_hours=6.0,
        )
        policy = FamilyShadowPolicy(name="x", description="x", enforcement_mode="weak_signal_veto")
        result = replay_snapshot_with_family_shadow(snapshot, state=state, family="state_engineering", policy=policy)

        self.assertIn("4h", result.changed_horizons)
        self.assertEqual(result.shadow_predictions["4h"]["trade_action"], "hold")

    def test_mid_band_policy_respects_confidence_scope(self) -> None:
        snapshot = _snapshot_template(trade_action="long", confidence=0.8, expected_value=0.0002)
        state = resolve_family_snapshot_state(
            snapshot_ts=pd.Timestamp("2026-04-03T10:00:00Z"),
            feature_frame=_feature_frame(),
            max_staleness_hours=6.0,
        )
        policy = FamilyShadowPolicy(
            name="mid",
            description="mid",
            enforcement_mode="weak_signal_veto",
            confidence_band_min=0.45,
            confidence_band_max=0.65,
        )
        result = replay_snapshot_with_family_shadow(snapshot, state=state, family="state_engineering", policy=policy)

        self.assertEqual(result.changed_horizons, [])
        self.assertGreater(result.diagnostics["scope_confidence_out"], 0)

    def test_confluence_relief_avoids_block_on_disagreement(self) -> None:
        snapshot = _snapshot_template(trade_action="long", confidence=0.5, expected_value=0.0002)
        frame = _feature_frame()
        frame["cvd_ratio_6h"] = 0.6
        frame["cvd_zscore_6h"] = 1.5
        frame["trades_taker_imbalance_acceleration_3h"] = 0.35
        frame["interaction_imbalance_trend_6h"] = 0.3
        frame["vwap_deviation_8h"] = 0.04
        state = resolve_family_snapshot_state(
            snapshot_ts=pd.Timestamp("2026-04-03T10:00:00Z"),
            feature_frame=frame,
            max_staleness_hours=6.0,
        )
        policy = FamilyShadowPolicy(name="relief", description="relief", enforcement_mode="confluence_relief_disagreement")
        result = replay_snapshot_with_family_shadow(snapshot, state=state, family="state_engineering", policy=policy)

        self.assertEqual(result.changed_horizons, [])
        self.assertGreater(result.diagnostics["confluence_disagreement"], 0)

    def test_sweep_returns_both_families(self) -> None:
        snapshots = [_snapshot_template() for _ in range(3)]
        for idx, snap in enumerate(snapshots):
            snap["generated_at"] = f"2026-04-03T1{idx}:00:00Z"
        sweep = run_state_order_flow_shadow_validation(
            snapshots=snapshots,
            feature_frame=_feature_frame(),
            policies=default_family_policy_variants(),
            max_staleness_hours=6.0,
        )

        self.assertIn("state_engineering", sweep["families"])
        self.assertIn("order_flow", sweep["families"])
        self.assertIn("overall_recommendation", sweep)


if __name__ == "__main__":
    unittest.main()
