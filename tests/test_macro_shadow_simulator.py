from __future__ import annotations

import unittest

import pandas as pd

from src.runtime.macro_shadow_simulator import (
    MacroShadowPolicy,
    default_policy_variants,
    replay_snapshot_with_macro_shadow,
    resolve_macro_state,
    run_policy_sweep,
    summarize_replay_results,
)


def _snapshot_template(trade_action: str = "long", confidence: float = 0.55, expected_value: float = 0.0005) -> dict:
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
            "15m": {
                "trade_action": "long",
                "execution_score": 1.2,
                "execution_plan": {"status": "ready", "reason": "pass"},
                "confidence_score": confidence,
                "expected_value": expected_value,
            },
            "4h": {
                "trade_action": trade_action,
                "execution_score": 1.4,
                "execution_plan": {"status": "ready", "reason": "pass"},
                "confidence_score": confidence,
                "expected_value": expected_value,
                "regime_state": "neutral",
            },
        },
    }


class MacroShadowSimulatorTests(unittest.TestCase):
    def test_resolve_macro_state_reports_stale(self) -> None:
        frame = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-04-01T00:00:00Z"], utc=True),
                "macro_dollar_proxy_change_1d": [-0.01],
                "macro_us10y_change_1d": [-0.02],
                "macro_eurusd_change_1d": [0.01],
            }
        )
        policy = MacroShadowPolicy(max_staleness_hours=6.0)
        state = resolve_macro_state(
            snapshot_ts=pd.Timestamp("2026-04-03T00:00:00Z"),
            macro_frame=frame,
            policy=policy,
        )

        self.assertEqual(state.status, "stale")
        self.assertEqual(state.macro_bias, "long")

    def test_replay_blocks_conflicting_weak_trade(self) -> None:
        snapshot = _snapshot_template(trade_action="long", confidence=0.5, expected_value=0.0002)
        macro_state = resolve_macro_state(
            snapshot_ts=pd.Timestamp("2026-04-03T10:00:00Z"),
            macro_frame=pd.DataFrame(
                {
                    "ts": pd.to_datetime(["2026-04-03T09:00:00Z"], utc=True),
                    "macro_dollar_proxy_change_1d": [0.02],
                    "macro_us10y_change_1d": [0.01],
                    "macro_eurusd_change_1d": [-0.01],
                }
            ),
            policy=MacroShadowPolicy(max_staleness_hours=24.0),
        )

        result = replay_snapshot_with_macro_shadow(
            snapshot,
            macro_state=macro_state,
            policy=MacroShadowPolicy(),
        )

        self.assertIn("4h", result.changed_horizons)
        self.assertEqual(result.shadow_predictions["4h"]["trade_action"], "hold")
        self.assertEqual(result.shadow_predictions["4h"]["execution_plan"]["reason"], "macro_shadow_conflict_block")

    def test_summary_classifies_neutral_when_small_changes(self) -> None:
        snapshot = _snapshot_template(trade_action="long", confidence=0.5, expected_value=0.0002)
        macro_state = resolve_macro_state(
            snapshot_ts=pd.Timestamp("2026-04-03T10:00:00Z"),
            macro_frame=pd.DataFrame(
                {
                    "ts": pd.to_datetime(["2026-04-03T09:00:00Z"], utc=True),
                    "macro_dollar_proxy_change_1d": [0.02],
                    "macro_us10y_change_1d": [0.01],
                    "macro_eurusd_change_1d": [-0.01],
                }
            ),
            policy=MacroShadowPolicy(max_staleness_hours=24.0),
        )
        result = replay_snapshot_with_macro_shadow(
            snapshot,
            macro_state=macro_state,
            policy=MacroShadowPolicy(),
        )

        summary = summarize_replay_results(
            snapshots=[snapshot],
            replay_results=[result],
        )
        self.assertEqual(summary["assessment"], "neutral")
        self.assertEqual(summary["snapshot_count"], 1)

    def test_horizon_scope_blocks_only_configured_horizons(self) -> None:
        snapshot = _snapshot_template(trade_action="long", confidence=0.4, expected_value=0.0001)
        snapshot["predictions"]["1h"] = {
            "trade_action": "long",
            "execution_score": 1.3,
            "execution_plan": {"status": "ready", "reason": "pass"},
            "confidence_score": 0.4,
            "expected_value": 0.0001,
            "regime_state": "neutral",
        }
        macro_state = resolve_macro_state(
            snapshot_ts=pd.Timestamp("2026-04-03T10:00:00Z"),
            macro_frame=pd.DataFrame(
                {
                    "ts": pd.to_datetime(["2026-04-03T09:00:00Z"], utc=True),
                    "macro_dollar_proxy_change_1d": [0.02],
                    "macro_us10y_change_1d": [0.01],
                    "macro_eurusd_change_1d": [-0.01],
                }
            ),
            policy=MacroShadowPolicy(max_staleness_hours=24.0),
        )

        policy = MacroShadowPolicy(enabled_horizons=(4.0,), enforcement_mode="strict_veto_conflict")
        result = replay_snapshot_with_macro_shadow(snapshot, macro_state=macro_state, policy=policy)
        self.assertIn("4h", result.changed_horizons)
        self.assertNotIn("1h", result.changed_horizons)

    def test_policy_sweep_returns_rankings(self) -> None:
        snapshots = [_snapshot_template(trade_action="long", confidence=0.4, expected_value=0.0001) for _ in range(4)]
        for idx, snap in enumerate(snapshots):
            snap["generated_at"] = f"2026-04-03T1{idx}:00:00Z"
        macro_frame = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-04-03T00:00:00Z"], utc=True),
                "macro_dollar_proxy_change_1d": [0.02],
                "macro_us10y_change_1d": [0.01],
                "macro_eurusd_change_1d": [-0.01],
            }
        )
        sweep = run_policy_sweep(
            snapshots=snapshots,
            macro_frame=macro_frame,
            policies=default_policy_variants(max_staleness_hours=24.0),
        )

        self.assertGreaterEqual(sweep["variant_count"], 3)
        self.assertTrue(sweep["variant_rankings"])
        self.assertIn("first_pass_diagnosis", sweep)


if __name__ == "__main__":
    unittest.main()
