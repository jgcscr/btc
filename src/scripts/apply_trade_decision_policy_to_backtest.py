from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.scripts import run_refresh_and_predict as rrp


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if pd.isna(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply trade-decision policy offline to backtest rows and emit decision-aligned candidate artifact.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meta-output", type=Path, default=None)
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=None,
        help="Optional JSON output path for detailed decision-policy diagnostics.",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Run diagnostics and meta generation without writing aligned CSV output.",
    )
    parser.add_argument(
        "--feature-source",
        action="append",
        default=[],
        help="Optional CSV path(s) used to backfill live-equivalent decision features by timestamp.",
    )
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--replace-threshold-rule", type=int, default=1)
    parser.add_argument("--require-direction-ret-alignment", type=int, default=1)
    parser.add_argument("--use-oof-expected-value", type=int, default=1)
    parser.add_argument("--oof-expected-value-mode", type=str, default="max_with_raw_calibrated")
    parser.add_argument("--enforce-positive-oof-envelope", type=int, default=1)
    parser.add_argument("--block-when-no-positive-oof-bin", type=int, default=1)
    parser.add_argument("--positive-oof-min-samples", type=int, default=4)
    parser.add_argument("--allow-raw-ev-fallback-when-no-positive-oof-bin", type=int, default=1)
    parser.add_argument("--raw-ev-fallback-quantile", type=float, default=0.9)
    parser.add_argument("--raw-ev-fallback-min-edge-over-fee", type=float, default=0.0)
    parser.add_argument("--min-expected-net", type=float, default=0.0)
    parser.add_argument("--min-edge-over-fee", type=float, default=0.0)
    parser.add_argument("--weak-band-candidate-only-veto", type=int, default=0)
    parser.add_argument("--weak-band-pup-low", type=float, default=0.55)
    parser.add_argument("--weak-band-pup-high", type=float, default=0.60)
    parser.add_argument("--weak-band-high-inclusive", type=int, default=0)
    parser.add_argument("--weak-band-incumbent-reference", type=Path, default=None)
    parser.add_argument("--weak-band-incumbent-signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--refined-candidate-only-veto", type=int, default=0)
    parser.add_argument("--refined-pup-low", type=float, default=0.55)
    parser.add_argument("--refined-pup-high", type=float, default=0.60)
    parser.add_argument("--refined-high-inclusive", type=int, default=0)
    parser.add_argument("--refined-min-abs-ret-pred", type=float, default=0.001)
    parser.add_argument("--refined-max-abs-ret-pred", type=float, default=None)
    parser.add_argument("--refined-incumbent-reference", type=Path, default=None)
    parser.add_argument("--refined-incumbent-signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--midband-candidate-only-veto", type=int, default=0)
    parser.add_argument("--midband-pup-low", type=float, default=0.55)
    parser.add_argument("--midband-pup-high", type=float, default=0.60)
    parser.add_argument("--midband-high-inclusive", type=int, default=0)
    parser.add_argument("--midband-min-abs-ret-pred", type=float, default=0.0005)
    parser.add_argument("--midband-max-abs-ret-pred", type=float, default=0.001)
    parser.add_argument("--midband-incumbent-reference", type=Path, default=None)
    parser.add_argument("--midband-incumbent-signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--raw-ev-sign-candidate-only-veto", type=int, default=0)
    parser.add_argument("--raw-ev-sign-pup-low", type=float, default=0.55)
    parser.add_argument("--raw-ev-sign-pup-high", type=float, default=0.60)
    parser.add_argument("--raw-ev-sign-high-inclusive", type=int, default=0)
    parser.add_argument("--raw-ev-sign-max", type=float, default=0.0)
    parser.add_argument("--raw-ev-sign-incumbent-reference", type=Path, default=None)
    parser.add_argument("--raw-ev-sign-incumbent-signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--direction-align-candidate-only-veto", type=int, default=0)
    parser.add_argument("--direction-align-pup-low", type=float, default=0.55)
    parser.add_argument("--direction-align-pup-high", type=float, default=0.60)
    parser.add_argument("--direction-align-high-inclusive", type=int, default=0)
    parser.add_argument("--direction-align-require-aligned", type=int, default=0)
    parser.add_argument("--direction-align-use-midband-slice", type=int, default=1)
    parser.add_argument("--direction-align-min-abs-ret-pred", type=float, default=0.0005)
    parser.add_argument("--direction-align-max-abs-ret-pred", type=float, default=0.001)
    parser.add_argument("--direction-align-incumbent-reference", type=Path, default=None)
    parser.add_argument("--direction-align-incumbent-signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--joint-direction-midband-candidate-only-veto", type=int, default=0)
    parser.add_argument("--joint-direction-midband-pup-low", type=float, default=0.55)
    parser.add_argument("--joint-direction-midband-pup-high", type=float, default=0.60)
    parser.add_argument("--joint-direction-midband-high-inclusive", type=int, default=0)
    parser.add_argument("--joint-direction-midband-require-aligned", type=int, default=0)
    parser.add_argument("--joint-direction-midband-min-abs-ret-pred", type=float, default=0.0005)
    parser.add_argument("--joint-direction-midband-max-abs-ret-pred", type=float, default=0.001)
    parser.add_argument("--joint-direction-midband-incumbent-reference", type=Path, default=None)
    parser.add_argument("--joint-direction-midband-incumbent-signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--regime-state-candidate-only-veto", type=int, default=0)
    parser.add_argument("--regime-state-pup-low", type=float, default=0.55)
    parser.add_argument("--regime-state-pup-high", type=float, default=0.60)
    parser.add_argument("--regime-state-high-inclusive", type=int, default=0)
    parser.add_argument("--regime-state-regimes", type=str, default="")
    parser.add_argument("--regime-state-use-midband-slice", type=int, default=0)
    parser.add_argument("--regime-state-min-abs-ret-pred", type=float, default=0.0005)
    parser.add_argument("--regime-state-max-abs-ret-pred", type=float, default=0.001)
    parser.add_argument("--regime-state-incumbent-reference", type=Path, default=None)
    parser.add_argument("--regime-state-incumbent-signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--chop-high-vol-candidate-only-veto", type=int, default=0)
    parser.add_argument("--chop-high-vol-pup-low", type=float, default=0.55)
    parser.add_argument("--chop-high-vol-pup-high", type=float, default=0.60)
    parser.add_argument("--chop-high-vol-high-inclusive", type=int, default=0)
    parser.add_argument("--chop-high-vol-regime-state", type=str, default="chop")
    parser.add_argument("--chop-high-vol-volatility-col", type=str, default="volatility_realized_24h")
    parser.add_argument("--chop-high-vol-min-volatility", type=float, default=None)
    parser.add_argument("--chop-high-vol-use-midband-slice", type=int, default=0)
    parser.add_argument("--chop-high-vol-min-abs-ret-pred", type=float, default=0.0005)
    parser.add_argument("--chop-high-vol-max-abs-ret-pred", type=float, default=0.001)
    parser.add_argument("--chop-high-vol-incumbent-reference", type=Path, default=None)
    parser.add_argument("--chop-high-vol-incumbent-signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--volatility-only-candidate-only-veto", type=int, default=0)
    parser.add_argument("--volatility-only-pup-low", type=float, default=0.55)
    parser.add_argument("--volatility-only-pup-high", type=float, default=0.60)
    parser.add_argument("--volatility-only-high-inclusive", type=int, default=0)
    parser.add_argument("--volatility-only-volatility-col", type=str, default="volatility_realized_24h")
    parser.add_argument("--volatility-only-min-volatility", type=float, default=None)
    parser.add_argument("--volatility-only-use-midband-slice", type=int, default=0)
    parser.add_argument("--volatility-only-min-abs-ret-pred", type=float, default=0.0005)
    parser.add_argument("--volatility-only-max-abs-ret-pred", type=float, default=0.001)
    parser.add_argument("--volatility-only-incumbent-reference", type=Path, default=None)
    parser.add_argument("--volatility-only-incumbent-signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--triggered-regime-volatility-veto", type=int, default=0)
    parser.add_argument("--triggered-regime-volatility-regimes", type=str, default="")
    parser.add_argument("--triggered-regime-volatility-regime-col", type=str, default="regime_state")
    parser.add_argument("--triggered-regime-volatility-volatility-col", type=str, default="volatility_realized_24h")
    parser.add_argument("--triggered-regime-volatility-min-volatility", type=float, default=None)
    parser.add_argument("--triggered-regime-volatility-max-volatility", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.model.exists():
        raise FileNotFoundError(args.model)

    df = pd.read_csv(args.input)

    # Backfill decision-critical fields from run-local sources keyed by timestamp.
    feature_cols = [
        "ret_pred",
        "signal_dir_only",
        "regime_state",
        "volatility_realized_24h",
        "volatility_ewm_24h",
        "volatility_garch_like",
        "expected_value",
    ]
    missing_before = {
        col: int(pd.to_numeric(df[col], errors="coerce").isna().sum()) if col in df.columns else int(len(df))
        for col in feature_cols
    }
    missing_after = dict(missing_before)
    backfill_by_column = {col: 0 for col in feature_cols}
    backfill_by_source: Dict[str, Dict[str, int]] = {}
    if "ts" in df.columns and args.feature_source:
        base = df.copy()
        base["_ts_norm"] = pd.to_datetime(base["ts"], utc=True, errors="coerce").dt.floor("h")
        for src_path in args.feature_source:
            src = Path(str(src_path))
            if not src.exists():
                continue
            src_df = pd.read_csv(src)
            if "ts" not in src_df.columns:
                continue
            src_df = src_df.copy()
            src_df["_ts_norm"] = pd.to_datetime(src_df["ts"], utc=True, errors="coerce").dt.floor("h")
            keep_cols = ["_ts_norm", *[c for c in feature_cols if c in src_df.columns]]
            src_df = src_df.loc[:, keep_cols].dropna(subset=["_ts_norm"]).drop_duplicates(subset=["_ts_norm"], keep="last")
            per_source_counts = {col: 0 for col in feature_cols}
            for col in feature_cols:
                if col not in src_df.columns:
                    continue
                mapped = src_df.set_index("_ts_norm")[col]
                fill_values = base["_ts_norm"].map(mapped)
                if col not in base.columns:
                    filled = fill_values.notna()
                    base[col] = fill_values
                else:
                    current = base[col]
                    filled = current.isna() & fill_values.notna()
                    base[col] = current.where(current.notna(), fill_values)
                filled_count = int(filled.sum())
                backfill_by_column[col] += filled_count
                per_source_counts[col] += filled_count
            backfill_by_source[str(src)] = per_source_counts
        df = base.drop(columns=["_ts_norm"], errors="ignore")
    missing_after = {
        col: int(pd.to_numeric(df[col], errors="coerce").isna().sum()) if col in df.columns else int(len(df))
        for col in feature_cols
    }

    policy_cfg: Dict[str, Any] = {
        "enabled": True,
        "model_path": str(args.model),
        "threshold": float(args.threshold),
        "replace_threshold_rule": bool(int(args.replace_threshold_rule)),
        "require_direction_ret_alignment": bool(int(args.require_direction_ret_alignment)),
        "use_oof_expected_value": bool(int(args.use_oof_expected_value)),
        "oof_expected_value_mode": str(args.oof_expected_value_mode),
        "enforce_positive_oof_envelope": bool(int(args.enforce_positive_oof_envelope)),
        "block_when_no_positive_oof_bin": bool(int(args.block_when_no_positive_oof_bin)),
        "positive_oof_min_samples": int(args.positive_oof_min_samples),
        "allow_raw_ev_fallback_when_no_positive_oof_bin": bool(
            int(args.allow_raw_ev_fallback_when_no_positive_oof_bin),
        ),
        "raw_ev_fallback_quantile": float(args.raw_ev_fallback_quantile),
        "raw_ev_fallback_min_edge_over_fee": float(args.raw_ev_fallback_min_edge_over_fee),
        "min_expected_net": float(args.min_expected_net),
        "min_edge_over_fee": float(args.min_edge_over_fee),
    }
    policy = rrp._resolve_trade_decision_policy(policy_cfg)
    if not bool(policy.get("enabled", False)):
        raise RuntimeError("Resolved trade decision policy is disabled; refusing to emit aligned candidate artifact.")

    fee_cost = (float(args.fee_bps) + float(args.slippage_bps)) / 10_000.0
    triggered_count = 0
    threshold = max(0.0, min(1.0, float(policy.get("threshold", args.threshold))))
    min_expected_net = float(policy.get("min_expected_net", 0.0))
    min_edge_over_fee = float(policy.get("min_edge_over_fee", 0.0))
    require_alignment = bool(policy.get("require_direction_ret_alignment", True))
    enforce_envelope = bool(policy.get("enforce_positive_oof_envelope", False))
    block_no_positive_bin = bool(policy.get("block_when_no_positive_oof_bin", True))
    allow_raw_fallback = bool(policy.get("allow_raw_ev_fallback_when_no_positive_oof_bin", False))

    diagnostics = {
        "rows": int(len(df)),
        "triggered_rows": 0,
        "blocked_rows": 0,
        "rules": {
            "threshold_miss": 0,
            "expected_net_miss": 0,
            "edge_over_fee_miss": 0,
            "direction_mismatch": 0,
            "positive_envelope_out_of_bin": 0,
            "positive_envelope_no_positive_bin": 0,
            "raw_ev_fallback_failed": 0,
            "raw_ev_fallback_passed": 0,
            "envelope_unavailable": 0,
        },
        "missing_features": {
            "before": missing_before,
            "after": missing_after,
            "backfill_by_column": backfill_by_column,
            "backfill_by_source": backfill_by_source,
        },
        "policy_flags": {
            "threshold": threshold,
            "min_expected_net": min_expected_net,
            "min_edge_over_fee": min_edge_over_fee,
            "require_direction_ret_alignment": require_alignment,
            "enforce_positive_oof_envelope": enforce_envelope,
            "block_when_no_positive_oof_bin": block_no_positive_bin,
            "allow_raw_ev_fallback_when_no_positive_oof_bin": allow_raw_fallback,
        },
    }

    trade_decision_payloads = []
    out_rows = []
    for _, row in df.iterrows():
        p_up = _as_float(row.get("p_up", row.get("p_up_meta", 0.5)), 0.5)
        ret_pred = _as_float(row.get("ret_pred", 0.0), 0.0)
        ret_1h = _as_float(row.get("ret_1h", 0.0), 0.0)

        inferred_dir = 1 if _as_float(row.get("ret_pred", 0.0), 0.0) >= 0.0 else 0
        signal_dir_only = _as_int(row.get("signal_dir_only", inferred_dir), inferred_dir)
        regime_state = str(row.get("regime_state", rrp.REGIME_NEUTRAL)).lower()

        result = {
            "p_up": p_up,
            "ret_pred": ret_pred,
            "expected_value": _as_float(row.get("expected_value", p_up * ret_pred), p_up * ret_pred),
            "signal_dir_only": signal_dir_only,
            "signal_ensemble": _as_int(row.get("signal_ensemble", 0), 0),
            "trade_action": str(row.get("trade_action", "hold")),
            "volatility": {
                "snapshot": {
                    "volatility_realized_24h": _as_float(row.get("volatility_realized_24h", 0.0), 0.0),
                    "volatility_ewm_24h": _as_float(row.get("volatility_ewm_24h", 0.0), 0.0),
                    "volatility_garch_like": _as_float(row.get("volatility_garch_like", 0.0), 0.0),
                },
            },
        }
        payload = rrp._apply_trade_decision_model(
            result=result,
            regime_state=regime_state,
            residual_std=0.0,
            policy=policy,
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )

        trade_prob = _as_float(payload.get("trade_probability", 0.0), 0.0)
        expected_net = _as_float(payload.get("expected_net", 0.0), 0.0)
        edge_over_fee = _as_float(payload.get("edge_over_fee", 0.0), 0.0)
        direction_ret_aligned = bool(payload.get("direction_ret_aligned", True))
        envelope_obj = payload.get("positive_oof_envelope", {})
        envelope = envelope_obj if isinstance(envelope_obj, dict) else {}
        envelope_available = bool(envelope.get("available", False))
        has_positive_bin = bool(envelope.get("has_positive_bin", False))
        in_positive_bin = bool(envelope.get("in_positive_bin", False))
        raw_fallback_pass = bool(payload.get("raw_ev_fallback_pass", False))
        triggered = bool(payload.get("triggered", False))

        if trade_prob < threshold:
            diagnostics["rules"]["threshold_miss"] += 1
        if expected_net < min_expected_net:
            diagnostics["rules"]["expected_net_miss"] += 1
        if edge_over_fee < min_edge_over_fee:
            diagnostics["rules"]["edge_over_fee_miss"] += 1
        if require_alignment and (not direction_ret_aligned):
            diagnostics["rules"]["direction_mismatch"] += 1
        if enforce_envelope:
            if not envelope_available:
                diagnostics["rules"]["envelope_unavailable"] += 1
            elif has_positive_bin and (not in_positive_bin):
                diagnostics["rules"]["positive_envelope_out_of_bin"] += 1
            elif (not has_positive_bin) and block_no_positive_bin:
                diagnostics["rules"]["positive_envelope_no_positive_bin"] += 1
                if allow_raw_fallback:
                    if raw_fallback_pass:
                        diagnostics["rules"]["raw_ev_fallback_passed"] += 1
                    else:
                        diagnostics["rules"]["raw_ev_fallback_failed"] += 1

        if triggered:
            diagnostics["triggered_rows"] += 1
        else:
            diagnostics["blocked_rows"] += 1

        signal = int(result.get("signal_ensemble", 0) or 0)
        direction = int(result.get("signal_dir_only", signal_dir_only) or 0)
        gross = float(ret_1h if direction == 1 else -ret_1h)
        net = float(gross - fee_cost) if signal == 1 else 0.0
        gross_trade = float(gross) if signal == 1 else 0.0

        out = row.to_dict()
        out["p_up"] = p_up
        out["signal_ensemble"] = signal
        out["ret_ensemble_gross"] = gross_trade
        out["ret_ensemble_net"] = net
        out["trade_action"] = result.get("trade_action", out.get("trade_action", "hold"))
        out["trade_decision_triggered"] = bool(payload.get("triggered", False))
        out["trade_decision_direction_ret_aligned"] = bool(payload.get("direction_ret_aligned", False))
        out["trade_decision_expected_net_raw"] = _as_float(payload.get("expected_net_raw", result.get("expected_value", 0.0)), 0.0)
        out_rows.append(out)
        trade_decision_payloads.append(payload)
        if signal == 1:
            triggered_count += 1

    out_df = pd.DataFrame(out_rows)

    def _apply_candidate_only_veto(
        *,
        enabled: bool,
        reference_path: Path | None,
        reference_signal_col: str,
        pup_low: float,
        pup_high: float,
        high_inclusive: bool,
        min_abs_ret_pred: float | None,
        max_abs_ret_pred: float | None,
    ) -> tuple[int, bool, str]:
        if not enabled:
            return 0, False, "disabled"
        if reference_path is None:
            return 0, False, "missing_incumbent_reference"
        if not reference_path.exists():
            return 0, False, "incumbent_reference_not_found"
        if "ts" not in out_df.columns:
            return 0, False, "candidate_missing_ts"

        ref_df = pd.read_csv(reference_path)
        if "ts" not in ref_df.columns or reference_signal_col not in ref_df.columns:
            return 0, False, "incumbent_reference_missing_required_columns"

        cand_ts = pd.to_datetime(out_df["ts"], utc=True, errors="coerce").dt.floor("h")
        ref_ts = pd.to_datetime(ref_df["ts"], utc=True, errors="coerce").dt.floor("h")
        ref_sig = pd.to_numeric(ref_df[reference_signal_col], errors="coerce").fillna(0.0)
        ref = (
            pd.DataFrame({"_ts_norm": ref_ts, "_inc_sig": ref_sig})
            .dropna(subset=["_ts_norm"])
            .drop_duplicates(subset=["_ts_norm"], keep="last")
        )
        ref_map = ref.set_index("_ts_norm")["_inc_sig"]
        inc_active = cand_ts.map(ref_map).fillna(0.0).astype(float) != 0.0
        cand_active = pd.to_numeric(out_df.get("signal_ensemble", 0), errors="coerce").fillna(0.0).astype(float) != 0.0
        p_up = pd.to_numeric(out_df.get("p_up", out_df.get("p_up_meta", np.nan)), errors="coerce")

        if high_inclusive:
            in_band = (p_up >= float(pup_low)) & (p_up <= float(pup_high))
        else:
            in_band = (p_up >= float(pup_low)) & (p_up < float(pup_high))

        veto_mask = cand_active & (~inc_active) & in_band.fillna(False)
        if min_abs_ret_pred is not None or max_abs_ret_pred is not None:
            abs_ret_pred = pd.to_numeric(out_df.get("ret_pred", np.nan), errors="coerce").abs()
            if min_abs_ret_pred is not None:
                veto_mask = veto_mask & (abs_ret_pred >= float(min_abs_ret_pred)).fillna(False)
            if max_abs_ret_pred is not None:
                veto_mask = veto_mask & (abs_ret_pred < float(max_abs_ret_pred)).fillna(False)

        veto_rows = int(veto_mask.sum())
        if veto_rows <= 0:
            return 0, True, "enabled_no_matching_rows"

        out_df.loc[veto_mask, "signal_ensemble"] = 0
        if "ret_ensemble_net" in out_df.columns:
            out_df.loc[veto_mask, "ret_ensemble_net"] = 0.0
        if "ret_ensemble_gross" in out_df.columns:
            out_df.loc[veto_mask, "ret_ensemble_gross"] = 0.0
        if "trade_action" in out_df.columns:
            out_df.loc[veto_mask, "trade_action"] = "hold"
        if "trade_decision_triggered" in out_df.columns:
            out_df.loc[veto_mask, "trade_decision_triggered"] = False
        return veto_rows, True, "veto_applied"

    weak_veto_rows, weak_veto_reference_found, weak_veto_reason = _apply_candidate_only_veto(
        enabled=bool(int(args.weak_band_candidate_only_veto)),
        reference_path=args.weak_band_incumbent_reference,
        reference_signal_col=str(args.weak_band_incumbent_signal_col),
        pup_low=float(args.weak_band_pup_low),
        pup_high=float(args.weak_band_pup_high),
        high_inclusive=bool(int(args.weak_band_high_inclusive)),
        min_abs_ret_pred=None,
        max_abs_ret_pred=None,
    )

    refined_veto_rows, refined_veto_reference_found, refined_veto_reason = _apply_candidate_only_veto(
        enabled=bool(int(args.refined_candidate_only_veto)),
        reference_path=args.refined_incumbent_reference,
        reference_signal_col=str(args.refined_incumbent_signal_col),
        pup_low=float(args.refined_pup_low),
        pup_high=float(args.refined_pup_high),
        high_inclusive=bool(int(args.refined_high_inclusive)),
        min_abs_ret_pred=float(args.refined_min_abs_ret_pred),
        max_abs_ret_pred=(None if args.refined_max_abs_ret_pred is None else float(args.refined_max_abs_ret_pred)),
    )

    midband_veto_rows, midband_veto_reference_found, midband_veto_reason = _apply_candidate_only_veto(
        enabled=bool(int(args.midband_candidate_only_veto)),
        reference_path=args.midband_incumbent_reference,
        reference_signal_col=str(args.midband_incumbent_signal_col),
        pup_low=float(args.midband_pup_low),
        pup_high=float(args.midband_pup_high),
        high_inclusive=bool(int(args.midband_high_inclusive)),
        min_abs_ret_pred=float(args.midband_min_abs_ret_pred),
        max_abs_ret_pred=(None if args.midband_max_abs_ret_pred is None else float(args.midband_max_abs_ret_pred)),
    )

    raw_ev_veto_rows = 0
    raw_ev_veto_reference_found = False
    raw_ev_veto_reason = "disabled"
    if bool(int(args.raw_ev_sign_candidate_only_veto)):
        if args.raw_ev_sign_incumbent_reference is None:
            raw_ev_veto_reason = "missing_incumbent_reference"
        elif not args.raw_ev_sign_incumbent_reference.exists():
            raw_ev_veto_reason = "incumbent_reference_not_found"
        elif "ts" not in out_df.columns:
            raw_ev_veto_reason = "candidate_missing_ts"
        else:
            ref_df = pd.read_csv(args.raw_ev_sign_incumbent_reference)
            if "ts" not in ref_df.columns or str(args.raw_ev_sign_incumbent_signal_col) not in ref_df.columns:
                raw_ev_veto_reason = "incumbent_reference_missing_required_columns"
            else:
                raw_ev_veto_reference_found = True
                cand_ts = pd.to_datetime(out_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_ts = pd.to_datetime(ref_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_sig = pd.to_numeric(ref_df[str(args.raw_ev_sign_incumbent_signal_col)], errors="coerce").fillna(0.0)
                ref = (
                    pd.DataFrame({"_ts_norm": ref_ts, "_inc_sig": ref_sig})
                    .dropna(subset=["_ts_norm"])
                    .drop_duplicates(subset=["_ts_norm"], keep="last")
                )
                ref_map = ref.set_index("_ts_norm")["_inc_sig"]
                inc_active = cand_ts.map(ref_map).fillna(0.0).astype(float) != 0.0
                cand_active = pd.to_numeric(out_df.get("signal_ensemble", 0), errors="coerce").fillna(0.0).astype(float) != 0.0
                p_up = pd.to_numeric(out_df.get("p_up", out_df.get("p_up_meta", np.nan)), errors="coerce")
                if bool(int(args.raw_ev_sign_high_inclusive)):
                    in_band = (p_up >= float(args.raw_ev_sign_pup_low)) & (p_up <= float(args.raw_ev_sign_pup_high))
                else:
                    in_band = (p_up >= float(args.raw_ev_sign_pup_low)) & (p_up < float(args.raw_ev_sign_pup_high))
                raw_ev_col = pd.to_numeric(out_df.get("trade_decision_expected_net_raw", np.nan), errors="coerce")
                to_veto = cand_active & (~inc_active) & in_band.fillna(False) & (raw_ev_col <= float(args.raw_ev_sign_max)).fillna(False)
                raw_ev_veto_rows = int(to_veto.sum())
                if raw_ev_veto_rows > 0:
                    out_df.loc[to_veto, "signal_ensemble"] = 0
                    if "ret_ensemble_net" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_net"] = 0.0
                    if "ret_ensemble_gross" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_gross"] = 0.0
                    if "trade_action" in out_df.columns:
                        out_df.loc[to_veto, "trade_action"] = "hold"
                    if "trade_decision_triggered" in out_df.columns:
                        out_df.loc[to_veto, "trade_decision_triggered"] = False
                    raw_ev_veto_reason = "veto_applied"
                else:
                    raw_ev_veto_reason = "enabled_no_matching_rows"

    triggered_count = int(
        (pd.to_numeric(out_df.get("signal_ensemble", 0), errors="coerce").fillna(0.0).astype(float) != 0.0).sum()
    )

    if bool(int(args.weak_band_candidate_only_veto)) and weak_veto_rows > 0:
        diagnostics["rules"]["weak_band_candidate_only_veto"] = int(weak_veto_rows)
    if bool(int(args.refined_candidate_only_veto)) and refined_veto_rows > 0:
        diagnostics["rules"]["refined_candidate_only_veto"] = int(refined_veto_rows)
    if bool(int(args.midband_candidate_only_veto)) and midband_veto_rows > 0:
        diagnostics["rules"]["midband_candidate_only_veto"] = int(midband_veto_rows)
    if bool(int(args.raw_ev_sign_candidate_only_veto)) and raw_ev_veto_rows > 0:
        diagnostics["rules"]["raw_ev_sign_candidate_only_veto"] = int(raw_ev_veto_rows)

    def _resolve_alignment_series() -> pd.Series:
        aligned_col = out_df.get("trade_decision_direction_ret_aligned")
        if aligned_col is None:
            signal_dir_only = pd.to_numeric(out_df.get("signal_dir_only", np.nan), errors="coerce")
            ret_pred = pd.to_numeric(out_df.get("ret_pred", np.nan), errors="coerce")
            return (
                ((signal_dir_only == 1) & (ret_pred > 0.0))
                | ((signal_dir_only == 0) & (ret_pred < 0.0))
            ).fillna(False)
        return pd.Series(aligned_col, index=out_df.index).fillna(False).astype(bool)

    direction_align_veto_rows = 0
    direction_align_veto_reference_found = False
    direction_align_veto_reason = "disabled"
    if bool(int(args.direction_align_candidate_only_veto)):
        if args.direction_align_incumbent_reference is None:
            direction_align_veto_reason = "missing_incumbent_reference"
        elif not args.direction_align_incumbent_reference.exists():
            direction_align_veto_reason = "incumbent_reference_not_found"
        elif "ts" not in out_df.columns:
            direction_align_veto_reason = "candidate_missing_ts"
        else:
            ref_df = pd.read_csv(args.direction_align_incumbent_reference)
            if "ts" not in ref_df.columns or str(args.direction_align_incumbent_signal_col) not in ref_df.columns:
                direction_align_veto_reason = "incumbent_reference_missing_required_columns"
            else:
                direction_align_veto_reference_found = True
                cand_ts = pd.to_datetime(out_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_ts = pd.to_datetime(ref_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_sig = pd.to_numeric(ref_df[str(args.direction_align_incumbent_signal_col)], errors="coerce").fillna(0.0)
                ref = (
                    pd.DataFrame({"_ts_norm": ref_ts, "_inc_sig": ref_sig})
                    .dropna(subset=["_ts_norm"])
                    .drop_duplicates(subset=["_ts_norm"], keep="last")
                )
                ref_map = ref.set_index("_ts_norm")["_inc_sig"]
                inc_active = cand_ts.map(ref_map).fillna(0.0).astype(float) != 0.0
                cand_active = pd.to_numeric(out_df.get("signal_ensemble", 0), errors="coerce").fillna(0.0).astype(float) != 0.0
                p_up = pd.to_numeric(out_df.get("p_up", out_df.get("p_up_meta", np.nan)), errors="coerce")
                if bool(int(args.direction_align_high_inclusive)):
                    in_band = (p_up >= float(args.direction_align_pup_low)) & (p_up <= float(args.direction_align_pup_high))
                else:
                    in_band = (p_up >= float(args.direction_align_pup_low)) & (p_up < float(args.direction_align_pup_high))

                aligned_series = _resolve_alignment_series()

                required_alignment = bool(int(args.direction_align_require_aligned))
                alignment_match = aligned_series == required_alignment

                to_veto = cand_active & (~inc_active) & in_band.fillna(False) & alignment_match.fillna(False)
                if bool(int(args.direction_align_use_midband_slice)):
                    abs_ret_pred = pd.to_numeric(out_df.get("ret_pred", np.nan), errors="coerce").abs()
                    to_veto = to_veto & (abs_ret_pred >= float(args.direction_align_min_abs_ret_pred)).fillna(False)
                    to_veto = to_veto & (abs_ret_pred < float(args.direction_align_max_abs_ret_pred)).fillna(False)

                direction_align_veto_rows = int(to_veto.sum())
                if direction_align_veto_rows > 0:
                    out_df.loc[to_veto, "signal_ensemble"] = 0
                    if "ret_ensemble_net" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_net"] = 0.0
                    if "ret_ensemble_gross" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_gross"] = 0.0
                    if "trade_action" in out_df.columns:
                        out_df.loc[to_veto, "trade_action"] = "hold"
                    if "trade_decision_triggered" in out_df.columns:
                        out_df.loc[to_veto, "trade_decision_triggered"] = False
                    direction_align_veto_reason = "veto_applied"
                else:
                    direction_align_veto_reason = "enabled_no_matching_rows"

    if bool(int(args.direction_align_candidate_only_veto)) and direction_align_veto_rows > 0:
        diagnostics["rules"]["direction_align_candidate_only_veto"] = int(direction_align_veto_rows)

    joint_direction_midband_veto_rows = 0
    joint_direction_midband_veto_reference_found = False
    joint_direction_midband_veto_reason = "disabled"
    if bool(int(args.joint_direction_midband_candidate_only_veto)):
        if args.joint_direction_midband_incumbent_reference is None:
            joint_direction_midband_veto_reason = "missing_incumbent_reference"
        elif not args.joint_direction_midband_incumbent_reference.exists():
            joint_direction_midband_veto_reason = "incumbent_reference_not_found"
        elif "ts" not in out_df.columns:
            joint_direction_midband_veto_reason = "candidate_missing_ts"
        else:
            ref_df = pd.read_csv(args.joint_direction_midband_incumbent_reference)
            if "ts" not in ref_df.columns or str(args.joint_direction_midband_incumbent_signal_col) not in ref_df.columns:
                joint_direction_midband_veto_reason = "incumbent_reference_missing_required_columns"
            else:
                joint_direction_midband_veto_reference_found = True
                cand_ts = pd.to_datetime(out_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_ts = pd.to_datetime(ref_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_sig = pd.to_numeric(ref_df[str(args.joint_direction_midband_incumbent_signal_col)], errors="coerce").fillna(0.0)
                ref = (
                    pd.DataFrame({"_ts_norm": ref_ts, "_inc_sig": ref_sig})
                    .dropna(subset=["_ts_norm"])
                    .drop_duplicates(subset=["_ts_norm"], keep="last")
                )
                ref_map = ref.set_index("_ts_norm")["_inc_sig"]
                inc_active = cand_ts.map(ref_map).fillna(0.0).astype(float) != 0.0
                cand_active = pd.to_numeric(out_df.get("signal_ensemble", 0), errors="coerce").fillna(0.0).astype(float) != 0.0
                p_up = pd.to_numeric(out_df.get("p_up", out_df.get("p_up_meta", np.nan)), errors="coerce")
                if bool(int(args.joint_direction_midband_high_inclusive)):
                    in_band = (p_up >= float(args.joint_direction_midband_pup_low)) & (p_up <= float(args.joint_direction_midband_pup_high))
                else:
                    in_band = (p_up >= float(args.joint_direction_midband_pup_low)) & (p_up < float(args.joint_direction_midband_pup_high))

                aligned_series = _resolve_alignment_series()
                required_alignment = bool(int(args.joint_direction_midband_require_aligned))
                alignment_match = aligned_series == required_alignment
                abs_ret_pred = pd.to_numeric(out_df.get("ret_pred", np.nan), errors="coerce").abs()
                in_midband = (
                    (abs_ret_pred >= float(args.joint_direction_midband_min_abs_ret_pred)).fillna(False)
                    & (abs_ret_pred < float(args.joint_direction_midband_max_abs_ret_pred)).fillna(False)
                )

                to_veto = cand_active & (~inc_active) & in_band.fillna(False) & in_midband & alignment_match.fillna(False)
                joint_direction_midband_veto_rows = int(to_veto.sum())
                if joint_direction_midband_veto_rows > 0:
                    out_df.loc[to_veto, "signal_ensemble"] = 0
                    if "ret_ensemble_net" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_net"] = 0.0
                    if "ret_ensemble_gross" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_gross"] = 0.0
                    if "trade_action" in out_df.columns:
                        out_df.loc[to_veto, "trade_action"] = "hold"
                    if "trade_decision_triggered" in out_df.columns:
                        out_df.loc[to_veto, "trade_decision_triggered"] = False
                    joint_direction_midband_veto_reason = "veto_applied"
                else:
                    joint_direction_midband_veto_reason = "enabled_no_matching_rows"

    if bool(int(args.joint_direction_midband_candidate_only_veto)) and joint_direction_midband_veto_rows > 0:
        diagnostics["rules"]["joint_direction_midband_candidate_only_veto"] = int(joint_direction_midband_veto_rows)

    regime_state_veto_rows = 0
    regime_state_veto_reference_found = False
    regime_state_veto_reason = "disabled"
    regime_state_filters = [
        value.strip().lower()
        for value in str(args.regime_state_regimes).split(",")
        if value.strip()
    ]
    if bool(int(args.regime_state_candidate_only_veto)):
        if not regime_state_filters:
            regime_state_veto_reason = "no_regime_state_filters_configured"
        elif args.regime_state_incumbent_reference is None:
            regime_state_veto_reason = "missing_incumbent_reference"
        elif not args.regime_state_incumbent_reference.exists():
            regime_state_veto_reason = "incumbent_reference_not_found"
        elif "ts" not in out_df.columns:
            regime_state_veto_reason = "candidate_missing_ts"
        else:
            ref_df = pd.read_csv(args.regime_state_incumbent_reference)
            if "ts" not in ref_df.columns or str(args.regime_state_incumbent_signal_col) not in ref_df.columns:
                regime_state_veto_reason = "incumbent_reference_missing_required_columns"
            else:
                regime_state_veto_reference_found = True
                cand_ts = pd.to_datetime(out_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_ts = pd.to_datetime(ref_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_sig = pd.to_numeric(ref_df[str(args.regime_state_incumbent_signal_col)], errors="coerce").fillna(0.0)
                ref = (
                    pd.DataFrame({"_ts_norm": ref_ts, "_inc_sig": ref_sig})
                    .dropna(subset=["_ts_norm"])
                    .drop_duplicates(subset=["_ts_norm"], keep="last")
                )
                ref_map = ref.set_index("_ts_norm")["_inc_sig"]
                inc_active = cand_ts.map(ref_map).fillna(0.0).astype(float) != 0.0
                cand_active = pd.to_numeric(out_df.get("signal_ensemble", 0), errors="coerce").fillna(0.0).astype(float) != 0.0
                p_up = pd.to_numeric(out_df.get("p_up", out_df.get("p_up_meta", np.nan)), errors="coerce")
                if bool(int(args.regime_state_high_inclusive)):
                    in_band = (p_up >= float(args.regime_state_pup_low)) & (p_up <= float(args.regime_state_pup_high))
                else:
                    in_band = (p_up >= float(args.regime_state_pup_low)) & (p_up < float(args.regime_state_pup_high))
                regime_state = out_df.get("regime_state", pd.Series(index=out_df.index, dtype=object)).map(
                    lambda value: str(value).strip().lower() if pd.notna(value) else ""
                )
                in_regime = regime_state.isin(regime_state_filters)
                to_veto = cand_active & (~inc_active) & in_band.fillna(False) & in_regime.fillna(False)
                if bool(int(args.regime_state_use_midband_slice)):
                    abs_ret_pred = pd.to_numeric(out_df.get("ret_pred", np.nan), errors="coerce").abs()
                    to_veto = to_veto & (abs_ret_pred >= float(args.regime_state_min_abs_ret_pred)).fillna(False)
                    to_veto = to_veto & (abs_ret_pred < float(args.regime_state_max_abs_ret_pred)).fillna(False)

                regime_state_veto_rows = int(to_veto.sum())
                if regime_state_veto_rows > 0:
                    out_df.loc[to_veto, "signal_ensemble"] = 0
                    if "ret_ensemble_net" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_net"] = 0.0
                    if "ret_ensemble_gross" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_gross"] = 0.0
                    if "trade_action" in out_df.columns:
                        out_df.loc[to_veto, "trade_action"] = "hold"
                    if "trade_decision_triggered" in out_df.columns:
                        out_df.loc[to_veto, "trade_decision_triggered"] = False
                    regime_state_veto_reason = "veto_applied"
                else:
                    regime_state_veto_reason = "enabled_no_matching_rows"

    if bool(int(args.regime_state_candidate_only_veto)) and regime_state_veto_rows > 0:
        diagnostics["rules"]["regime_state_candidate_only_veto"] = int(regime_state_veto_rows)

    chop_high_vol_veto_rows = 0
    chop_high_vol_veto_reference_found = False
    chop_high_vol_veto_reason = "disabled"
    if bool(int(args.chop_high_vol_candidate_only_veto)):
        if args.chop_high_vol_min_volatility is None:
            chop_high_vol_veto_reason = "missing_min_volatility"
        elif args.chop_high_vol_incumbent_reference is None:
            chop_high_vol_veto_reason = "missing_incumbent_reference"
        elif not args.chop_high_vol_incumbent_reference.exists():
            chop_high_vol_veto_reason = "incumbent_reference_not_found"
        elif "ts" not in out_df.columns:
            chop_high_vol_veto_reason = "candidate_missing_ts"
        else:
            ref_df = pd.read_csv(args.chop_high_vol_incumbent_reference)
            if "ts" not in ref_df.columns or str(args.chop_high_vol_incumbent_signal_col) not in ref_df.columns:
                chop_high_vol_veto_reason = "incumbent_reference_missing_required_columns"
            else:
                chop_high_vol_veto_reference_found = True
                cand_ts = pd.to_datetime(out_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_ts = pd.to_datetime(ref_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_sig = pd.to_numeric(ref_df[str(args.chop_high_vol_incumbent_signal_col)], errors="coerce").fillna(0.0)
                ref = (
                    pd.DataFrame({"_ts_norm": ref_ts, "_inc_sig": ref_sig})
                    .dropna(subset=["_ts_norm"])
                    .drop_duplicates(subset=["_ts_norm"], keep="last")
                )
                ref_map = ref.set_index("_ts_norm")["_inc_sig"]
                inc_active = cand_ts.map(ref_map).fillna(0.0).astype(float) != 0.0
                cand_active = pd.to_numeric(out_df.get("signal_ensemble", 0), errors="coerce").fillna(0.0).astype(float) != 0.0
                p_up = pd.to_numeric(out_df.get("p_up", out_df.get("p_up_meta", np.nan)), errors="coerce")
                if bool(int(args.chop_high_vol_high_inclusive)):
                    in_band = (p_up >= float(args.chop_high_vol_pup_low)) & (p_up <= float(args.chop_high_vol_pup_high))
                else:
                    in_band = (p_up >= float(args.chop_high_vol_pup_low)) & (p_up < float(args.chop_high_vol_pup_high))
                regime_state = out_df.get("regime_state", pd.Series(index=out_df.index, dtype=object)).map(
                    lambda value: str(value).strip().lower() if pd.notna(value) else ""
                )
                volatility = pd.to_numeric(
                    out_df.get(str(args.chop_high_vol_volatility_col), np.nan),
                    errors="coerce",
                )
                in_regime = regime_state == str(args.chop_high_vol_regime_state).strip().lower()
                high_vol = (volatility >= float(args.chop_high_vol_min_volatility)).fillna(False)
                to_veto = cand_active & (~inc_active) & in_band.fillna(False) & in_regime.fillna(False) & high_vol
                if bool(int(args.chop_high_vol_use_midband_slice)):
                    abs_ret_pred = pd.to_numeric(out_df.get("ret_pred", np.nan), errors="coerce").abs()
                    to_veto = to_veto & (abs_ret_pred >= float(args.chop_high_vol_min_abs_ret_pred)).fillna(False)
                    to_veto = to_veto & (abs_ret_pred < float(args.chop_high_vol_max_abs_ret_pred)).fillna(False)

                chop_high_vol_veto_rows = int(to_veto.sum())
                if chop_high_vol_veto_rows > 0:
                    out_df.loc[to_veto, "signal_ensemble"] = 0
                    if "ret_ensemble_net" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_net"] = 0.0
                    if "ret_ensemble_gross" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_gross"] = 0.0
                    if "trade_action" in out_df.columns:
                        out_df.loc[to_veto, "trade_action"] = "hold"
                    if "trade_decision_triggered" in out_df.columns:
                        out_df.loc[to_veto, "trade_decision_triggered"] = False
                    chop_high_vol_veto_reason = "veto_applied"
                else:
                    chop_high_vol_veto_reason = "enabled_no_matching_rows"

    if bool(int(args.chop_high_vol_candidate_only_veto)) and chop_high_vol_veto_rows > 0:
        diagnostics["rules"]["chop_high_vol_candidate_only_veto"] = int(chop_high_vol_veto_rows)

    volatility_only_veto_rows = 0
    volatility_only_veto_reference_found = False
    volatility_only_veto_reason = "disabled"
    if bool(int(args.volatility_only_candidate_only_veto)):
        if args.volatility_only_min_volatility is None:
            volatility_only_veto_reason = "missing_min_volatility"
        elif args.volatility_only_incumbent_reference is None:
            volatility_only_veto_reason = "missing_incumbent_reference"
        elif not args.volatility_only_incumbent_reference.exists():
            volatility_only_veto_reason = "incumbent_reference_not_found"
        elif "ts" not in out_df.columns:
            volatility_only_veto_reason = "candidate_missing_ts"
        else:
            ref_df = pd.read_csv(args.volatility_only_incumbent_reference)
            if "ts" not in ref_df.columns or str(args.volatility_only_incumbent_signal_col) not in ref_df.columns:
                volatility_only_veto_reason = "incumbent_reference_missing_required_columns"
            else:
                volatility_only_veto_reference_found = True
                cand_ts = pd.to_datetime(out_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_ts = pd.to_datetime(ref_df["ts"], utc=True, errors="coerce").dt.floor("h")
                ref_sig = pd.to_numeric(ref_df[str(args.volatility_only_incumbent_signal_col)], errors="coerce").fillna(0.0)
                ref = (
                    pd.DataFrame({"_ts_norm": ref_ts, "_inc_sig": ref_sig})
                    .dropna(subset=["_ts_norm"])
                    .drop_duplicates(subset=["_ts_norm"], keep="last")
                )
                ref_map = ref.set_index("_ts_norm")["_inc_sig"]
                inc_active = cand_ts.map(ref_map).fillna(0.0).astype(float) != 0.0
                cand_active = pd.to_numeric(out_df.get("signal_ensemble", 0), errors="coerce").fillna(0.0).astype(float) != 0.0
                p_up = pd.to_numeric(out_df.get("p_up", out_df.get("p_up_meta", np.nan)), errors="coerce")
                if bool(int(args.volatility_only_high_inclusive)):
                    in_band = (p_up >= float(args.volatility_only_pup_low)) & (p_up <= float(args.volatility_only_pup_high))
                else:
                    in_band = (p_up >= float(args.volatility_only_pup_low)) & (p_up < float(args.volatility_only_pup_high))
                volatility = pd.to_numeric(
                    out_df.get(str(args.volatility_only_volatility_col), np.nan),
                    errors="coerce",
                )
                high_vol = (volatility >= float(args.volatility_only_min_volatility)).fillna(False)
                to_veto = cand_active & (~inc_active) & in_band.fillna(False) & high_vol
                if bool(int(args.volatility_only_use_midband_slice)):
                    abs_ret_pred = pd.to_numeric(out_df.get("ret_pred", np.nan), errors="coerce").abs()
                    to_veto = to_veto & (abs_ret_pred >= float(args.volatility_only_min_abs_ret_pred)).fillna(False)
                    to_veto = to_veto & (abs_ret_pred < float(args.volatility_only_max_abs_ret_pred)).fillna(False)

                volatility_only_veto_rows = int(to_veto.sum())
                if volatility_only_veto_rows > 0:
                    out_df.loc[to_veto, "signal_ensemble"] = 0
                    if "ret_ensemble_net" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_net"] = 0.0
                    if "ret_ensemble_gross" in out_df.columns:
                        out_df.loc[to_veto, "ret_ensemble_gross"] = 0.0
                    if "trade_action" in out_df.columns:
                        out_df.loc[to_veto, "trade_action"] = "hold"
                    if "trade_decision_triggered" in out_df.columns:
                        out_df.loc[to_veto, "trade_decision_triggered"] = False
                    volatility_only_veto_reason = "veto_applied"
                else:
                    volatility_only_veto_reason = "enabled_no_matching_rows"

    if bool(int(args.volatility_only_candidate_only_veto)) and volatility_only_veto_rows > 0:
        diagnostics["rules"]["volatility_only_candidate_only_veto"] = int(volatility_only_veto_rows)

    triggered_regime_volatility_veto_rows = 0
    triggered_regime_volatility_veto_reason = "disabled"
    triggered_regime_volatility_filters = [
        value.strip().lower()
        for value in str(args.triggered_regime_volatility_regimes).split(",")
        if value.strip()
    ]
    if bool(int(args.triggered_regime_volatility_veto)):
        signal_series = pd.to_numeric(out_df.get("signal_ensemble", 0), errors="coerce").fillna(0.0).astype(float)
        regime_series = out_df.get(
            str(args.triggered_regime_volatility_regime_col),
            pd.Series(index=out_df.index, dtype=object),
        ).map(lambda value: str(value).strip().lower() if pd.notna(value) else "unknown")
        volatility_series = pd.to_numeric(
            out_df.get(str(args.triggered_regime_volatility_volatility_col), np.nan),
            errors="coerce",
        )
        to_veto = signal_series != 0.0
        if triggered_regime_volatility_filters:
            to_veto = to_veto & regime_series.isin(triggered_regime_volatility_filters)
        if args.triggered_regime_volatility_min_volatility is not None:
            to_veto = to_veto & volatility_series.ge(float(args.triggered_regime_volatility_min_volatility)).fillna(False)
        if args.triggered_regime_volatility_max_volatility is not None:
            to_veto = to_veto & volatility_series.lt(float(args.triggered_regime_volatility_max_volatility)).fillna(False)
        triggered_regime_volatility_veto_rows = int(to_veto.sum())
        if triggered_regime_volatility_veto_rows > 0:
            out_df.loc[to_veto, "signal_ensemble"] = 0
            if "ret_ensemble_net" in out_df.columns:
                out_df.loc[to_veto, "ret_ensemble_net"] = 0.0
            if "ret_ensemble_gross" in out_df.columns:
                out_df.loc[to_veto, "ret_ensemble_gross"] = 0.0
            if "trade_action" in out_df.columns:
                out_df.loc[to_veto, "trade_action"] = "hold"
            if "trade_decision_triggered" in out_df.columns:
                out_df.loc[to_veto, "trade_decision_triggered"] = False
            triggered_regime_volatility_veto_reason = "veto_applied"
        else:
            triggered_regime_volatility_veto_reason = "enabled_no_matching_rows"

    if bool(int(args.triggered_regime_volatility_veto)) and triggered_regime_volatility_veto_rows > 0:
        diagnostics["rules"]["triggered_regime_volatility_veto"] = int(triggered_regime_volatility_veto_rows)
    if not args.diagnostics_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output, index=False)

    meta = {
        "input": str(args.input),
        "model": str(args.model),
        "output": str(args.output),
        "rows": int(len(out_df)),
        "trade_count": int(triggered_count),
        "net_return_total": float(pd.to_numeric(out_df.get("ret_ensemble_net", 0.0), errors="coerce").fillna(0.0).sum()),
        "policy": {
            "threshold": float(args.threshold),
            "replace_threshold_rule": bool(int(args.replace_threshold_rule)),
            "require_direction_ret_alignment": bool(int(args.require_direction_ret_alignment)),
            "use_oof_expected_value": bool(int(args.use_oof_expected_value)),
            "oof_expected_value_mode": str(args.oof_expected_value_mode),
            "enforce_positive_oof_envelope": bool(int(args.enforce_positive_oof_envelope)),
            "block_when_no_positive_oof_bin": bool(int(args.block_when_no_positive_oof_bin)),
            "positive_oof_min_samples": int(args.positive_oof_min_samples),
            "allow_raw_ev_fallback_when_no_positive_oof_bin": bool(
                int(args.allow_raw_ev_fallback_when_no_positive_oof_bin),
            ),
            "raw_ev_fallback_quantile": float(args.raw_ev_fallback_quantile),
            "raw_ev_fallback_min_edge_over_fee": float(args.raw_ev_fallback_min_edge_over_fee),
            "min_expected_net": float(args.min_expected_net),
            "min_edge_over_fee": float(args.min_edge_over_fee),
        },
        "diagnostics": diagnostics,
        "weak_band_candidate_only_veto": {
            "enabled": bool(int(args.weak_band_candidate_only_veto)),
            "p_up_low": float(args.weak_band_pup_low),
            "p_up_high": float(args.weak_band_pup_high),
            "high_inclusive": bool(int(args.weak_band_high_inclusive)),
            "incumbent_reference": (
                str(args.weak_band_incumbent_reference)
                if args.weak_band_incumbent_reference is not None
                else None
            ),
            "incumbent_signal_col": str(args.weak_band_incumbent_signal_col),
            "incumbent_reference_found": bool(weak_veto_reference_found),
            "vetoed_rows": int(weak_veto_rows),
            "status": str(weak_veto_reason),
        },
        "refined_candidate_only_veto": {
            "enabled": bool(int(args.refined_candidate_only_veto)),
            "p_up_low": float(args.refined_pup_low),
            "p_up_high": float(args.refined_pup_high),
            "high_inclusive": bool(int(args.refined_high_inclusive)),
            "min_abs_ret_pred": float(args.refined_min_abs_ret_pred),
            "max_abs_ret_pred": (None if args.refined_max_abs_ret_pred is None else float(args.refined_max_abs_ret_pred)),
            "incumbent_reference": (
                str(args.refined_incumbent_reference)
                if args.refined_incumbent_reference is not None
                else None
            ),
            "incumbent_signal_col": str(args.refined_incumbent_signal_col),
            "incumbent_reference_found": bool(refined_veto_reference_found),
            "vetoed_rows": int(refined_veto_rows),
            "status": str(refined_veto_reason),
        },
        "midband_candidate_only_veto": {
            "enabled": bool(int(args.midband_candidate_only_veto)),
            "p_up_low": float(args.midband_pup_low),
            "p_up_high": float(args.midband_pup_high),
            "high_inclusive": bool(int(args.midband_high_inclusive)),
            "min_abs_ret_pred": float(args.midband_min_abs_ret_pred),
            "max_abs_ret_pred": (None if args.midband_max_abs_ret_pred is None else float(args.midband_max_abs_ret_pred)),
            "incumbent_reference": (
                str(args.midband_incumbent_reference)
                if args.midband_incumbent_reference is not None
                else None
            ),
            "incumbent_signal_col": str(args.midband_incumbent_signal_col),
            "incumbent_reference_found": bool(midband_veto_reference_found),
            "vetoed_rows": int(midband_veto_rows),
            "status": str(midband_veto_reason),
        },
        "raw_ev_sign_candidate_only_veto": {
            "enabled": bool(int(args.raw_ev_sign_candidate_only_veto)),
            "p_up_low": float(args.raw_ev_sign_pup_low),
            "p_up_high": float(args.raw_ev_sign_pup_high),
            "high_inclusive": bool(int(args.raw_ev_sign_high_inclusive)),
            "raw_ev_sign_max": float(args.raw_ev_sign_max),
            "raw_ev_field": "trade_decision_expected_net_raw",
            "incumbent_reference": (
                str(args.raw_ev_sign_incumbent_reference)
                if args.raw_ev_sign_incumbent_reference is not None
                else None
            ),
            "incumbent_signal_col": str(args.raw_ev_sign_incumbent_signal_col),
            "incumbent_reference_found": bool(raw_ev_veto_reference_found),
            "vetoed_rows": int(raw_ev_veto_rows),
            "status": str(raw_ev_veto_reason),
        },
        "direction_align_candidate_only_veto": {
            "enabled": bool(int(args.direction_align_candidate_only_veto)),
            "p_up_low": float(args.direction_align_pup_low),
            "p_up_high": float(args.direction_align_pup_high),
            "high_inclusive": bool(int(args.direction_align_high_inclusive)),
            "require_aligned": bool(int(args.direction_align_require_aligned)),
            "use_midband_slice": bool(int(args.direction_align_use_midband_slice)),
            "min_abs_ret_pred": float(args.direction_align_min_abs_ret_pred),
            "max_abs_ret_pred": float(args.direction_align_max_abs_ret_pred),
            "alignment_field": "trade_decision_direction_ret_aligned",
            "incumbent_reference": (
                str(args.direction_align_incumbent_reference)
                if args.direction_align_incumbent_reference is not None
                else None
            ),
            "incumbent_signal_col": str(args.direction_align_incumbent_signal_col),
            "incumbent_reference_found": bool(direction_align_veto_reference_found),
            "vetoed_rows": int(direction_align_veto_rows),
            "status": str(direction_align_veto_reason),
        },
        "joint_direction_midband_candidate_only_veto": {
            "enabled": bool(int(args.joint_direction_midband_candidate_only_veto)),
            "p_up_low": float(args.joint_direction_midband_pup_low),
            "p_up_high": float(args.joint_direction_midband_pup_high),
            "high_inclusive": bool(int(args.joint_direction_midband_high_inclusive)),
            "require_aligned": bool(int(args.joint_direction_midband_require_aligned)),
            "min_abs_ret_pred": float(args.joint_direction_midband_min_abs_ret_pred),
            "max_abs_ret_pred": float(args.joint_direction_midband_max_abs_ret_pred),
            "alignment_field": "trade_decision_direction_ret_aligned",
            "incumbent_reference": (
                str(args.joint_direction_midband_incumbent_reference)
                if args.joint_direction_midband_incumbent_reference is not None
                else None
            ),
            "incumbent_signal_col": str(args.joint_direction_midband_incumbent_signal_col),
            "incumbent_reference_found": bool(joint_direction_midband_veto_reference_found),
            "vetoed_rows": int(joint_direction_midband_veto_rows),
            "status": str(joint_direction_midband_veto_reason),
        },
        "regime_state_candidate_only_veto": {
            "enabled": bool(int(args.regime_state_candidate_only_veto)),
            "p_up_low": float(args.regime_state_pup_low),
            "p_up_high": float(args.regime_state_pup_high),
            "high_inclusive": bool(int(args.regime_state_high_inclusive)),
            "regime_states": regime_state_filters,
            "use_midband_slice": bool(int(args.regime_state_use_midband_slice)),
            "min_abs_ret_pred": float(args.regime_state_min_abs_ret_pred),
            "max_abs_ret_pred": float(args.regime_state_max_abs_ret_pred),
            "regime_field": "regime_state",
            "incumbent_reference": (
                str(args.regime_state_incumbent_reference)
                if args.regime_state_incumbent_reference is not None
                else None
            ),
            "incumbent_signal_col": str(args.regime_state_incumbent_signal_col),
            "incumbent_reference_found": bool(regime_state_veto_reference_found),
            "vetoed_rows": int(regime_state_veto_rows),
            "status": str(regime_state_veto_reason),
        },
        "chop_high_vol_candidate_only_veto": {
            "enabled": bool(int(args.chop_high_vol_candidate_only_veto)),
            "p_up_low": float(args.chop_high_vol_pup_low),
            "p_up_high": float(args.chop_high_vol_pup_high),
            "high_inclusive": bool(int(args.chop_high_vol_high_inclusive)),
            "regime_state": str(args.chop_high_vol_regime_state).strip().lower(),
            "volatility_col": str(args.chop_high_vol_volatility_col),
            "min_volatility": (
                None if args.chop_high_vol_min_volatility is None else float(args.chop_high_vol_min_volatility)
            ),
            "use_midband_slice": bool(int(args.chop_high_vol_use_midband_slice)),
            "min_abs_ret_pred": float(args.chop_high_vol_min_abs_ret_pred),
            "max_abs_ret_pred": float(args.chop_high_vol_max_abs_ret_pred),
            "incumbent_reference": (
                str(args.chop_high_vol_incumbent_reference)
                if args.chop_high_vol_incumbent_reference is not None
                else None
            ),
            "incumbent_signal_col": str(args.chop_high_vol_incumbent_signal_col),
            "incumbent_reference_found": bool(chop_high_vol_veto_reference_found),
            "vetoed_rows": int(chop_high_vol_veto_rows),
            "status": str(chop_high_vol_veto_reason),
        },
        "volatility_only_candidate_only_veto": {
            "enabled": bool(int(args.volatility_only_candidate_only_veto)),
            "p_up_low": float(args.volatility_only_pup_low),
            "p_up_high": float(args.volatility_only_pup_high),
            "high_inclusive": bool(int(args.volatility_only_high_inclusive)),
            "volatility_col": str(args.volatility_only_volatility_col),
            "min_volatility": (
                None if args.volatility_only_min_volatility is None else float(args.volatility_only_min_volatility)
            ),
            "use_midband_slice": bool(int(args.volatility_only_use_midband_slice)),
            "min_abs_ret_pred": float(args.volatility_only_min_abs_ret_pred),
            "max_abs_ret_pred": float(args.volatility_only_max_abs_ret_pred),
            "incumbent_reference": (
                str(args.volatility_only_incumbent_reference)
                if args.volatility_only_incumbent_reference is not None
                else None
            ),
            "incumbent_signal_col": str(args.volatility_only_incumbent_signal_col),
            "incumbent_reference_found": bool(volatility_only_veto_reference_found),
            "vetoed_rows": int(volatility_only_veto_rows),
            "status": str(volatility_only_veto_reason),
        },
        "triggered_regime_volatility_veto": {
            "enabled": bool(int(args.triggered_regime_volatility_veto)),
            "regime_states": triggered_regime_volatility_filters,
            "regime_col": str(args.triggered_regime_volatility_regime_col),
            "volatility_col": str(args.triggered_regime_volatility_volatility_col),
            "min_volatility": (
                float(args.triggered_regime_volatility_min_volatility)
                if args.triggered_regime_volatility_min_volatility is not None
                else None
            ),
            "max_volatility": (
                float(args.triggered_regime_volatility_max_volatility)
                if args.triggered_regime_volatility_max_volatility is not None
                else None
            ),
            "vetoed_rows": int(triggered_regime_volatility_veto_rows),
            "status": str(triggered_regime_volatility_veto_reason),
        },
    }

    if args.meta_output is not None:
        args.meta_output.parent.mkdir(parents=True, exist_ok=True)
        args.meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.diagnostics_output is not None:
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_output.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
