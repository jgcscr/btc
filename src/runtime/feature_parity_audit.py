from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.data.source_parity import load_source_family_artifacts


LEAKAGE_SAFE_REPORT_PATH = Path("artifacts/analysis/featurelift_20260331_rerun/comparison_report.md")
RELIABILITY_PATH = Path("artifacts/analysis/feature_reliability_15m_1h_slice_20260331.json")


def classify_feature_family(feature_name: str) -> str:
    name = str(feature_name).strip().lower()
    if not name:
        return "other"

    if name.startswith("macro_"):
        return "macro"
    if name.startswith("onchain_"):
        return "onchain"
    if name.startswith("fut_") or name in {"funding_rate", "funding_rate_annualized", "open_interest", "funding_rate_zscore_24h"}:
        return "derivatives"
    if name.startswith("intrabar_"):
        return "intrabar"
    if name.startswith("volatility_") or name.startswith("vol_"):
        return "volatility"
    if name.startswith("liquidity_") or name.startswith("vwap_") or name.startswith("distance_from_session_"):
        return "liquidity_structure"
    if name.startswith("range_") or name.startswith("price_distance_"):
        return "state_engineering"
    if name.startswith("trend_") or name.startswith("momentum_") or name.startswith("interaction_"):
        return "state_engineering"
    if name.startswith("cvd_") or name.startswith("trades_") or "imbalance" in name:
        return "order_flow"
    if name.startswith("ret_"):
        return "forward_return_proxy"
    if name in {
        "open",
        "high",
        "low",
        "close",
        "close_delta_1h",
        "close_pct_change_1h",
        "ma_close_7h",
        "ma_close_24h",
        "ma_ratio_7_24",
    }:
        return "price_core"
    if name in {
        "volume",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "volume_delta_1h",
        "volume_pct_change_1h",
    }:
        return "volume_core"
    return "other"


def classify_source_family(source_name: str) -> str:
    source = str(source_name).strip().lower()
    if source in {"macro", "macroeconomic"}:
        return "macro"
    if source in {"onchain", "on_chain"}:
        return "onchain"
    if source in {"funding", "futures", "derivatives"}:
        return "derivatives"
    if source in {"intrabar", "intraday"}:
        return "intrabar"
    return classify_feature_family(source)


def family_counts(feature_names: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name in feature_names:
        family = classify_feature_family(name)
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _feature_to_family_map(feature_names: Iterable[str]) -> Dict[str, str]:
    return {str(name): classify_feature_family(str(name)) for name in feature_names}


def _load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
def _read_reliability_payload(path: Path) -> Dict[str, Any]:
    payload = _load_json(path) or {}
    accepted = payload.get("accepted_features")
    scores = payload.get("feature_scores")
    return {
        "accepted_features": [str(v) for v in accepted] if isinstance(accepted, list) else [],
        "feature_scores": scores if isinstance(scores, dict) else {},
    }


def _extract_feature_names_from_metadata(path: Path) -> List[str]:
    payload = _load_json(path)
    if not payload:
        return []
    values = payload.get("feature_names")
    if not isinstance(values, list):
        return []
    return [str(v) for v in values]


def _collect_horizon_feature_sets(models_root: Path, horizons: Sequence[float]) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for horizon in horizons:
        suffix = f"{int(round(horizon))}h" if horizon >= 1.0 else f"{int(round(horizon * 60))}m"
        dir_candidates = sorted(models_root.glob(f"xgb_dir{suffix}_v*/model_metadata_direction.json"))
        ret_candidates = sorted(models_root.glob(f"xgb_ret{suffix}_v*/model_metadata.json"))

        direction_features = _extract_feature_names_from_metadata(dir_candidates[-1]) if dir_candidates else []
        regression_features = _extract_feature_names_from_metadata(ret_candidates[-1]) if ret_candidates else []
        out[suffix] = {
            "direction_features": direction_features,
            "regression_features": regression_features,
            "training_union": sorted(set(direction_features) | set(regression_features)),
        }
    return out


def _parse_featurelift_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "report_found": False,
            "leakage_safe_rerun": False,
            "multi_horizon_leakage_warning": "missing_report",
        }

    text = path.read_text(encoding="utf-8")
    leakage_safe = "supersedes" in text.lower() and "leakage" in text.lower()
    degraded_multi = "degradations" in text.lower() and "leaked edge" in text.lower()
    return {
        "report_found": True,
        "leakage_safe_rerun": leakage_safe,
        "multi_horizon_leakage_warning": "present" if degraded_multi else "not_detected",
    }


@dataclass(frozen=True)
class CandidateScore:
    family: str
    expected_value: float
    implementation_risk: float
    parity_gain: float
    evidence: str
    recommendation: str


def rank_candidates(
    *,
    available_families: Iterable[str],
    live_enforced_families: Iterable[str],
    ignored_families: Iterable[str],
    reliability_payload: Mapping[str, Any],
    leakage_payload: Mapping[str, Any],
) -> List[CandidateScore]:
    available = set(available_families)
    live = set(live_enforced_families)
    ignored = set(ignored_families)

    accepted_features = reliability_payload.get("accepted_features", [])
    score_map = reliability_payload.get("feature_scores", {})
    accepted_family_counts = family_counts(accepted_features)

    mean_scores: Dict[str, float] = {}
    family_score_counts: Dict[str, int] = {}
    if isinstance(score_map, Mapping):
        for name, payload in score_map.items():
            if not isinstance(payload, Mapping):
                continue
            raw_score = payload.get("score")
            if not isinstance(raw_score, (float, int)):
                continue
            family = classify_feature_family(str(name))
            mean_scores[family] = mean_scores.get(family, 0.0) + float(raw_score)
            family_score_counts[family] = family_score_counts.get(family, 0) + 1
    for family, total in list(mean_scores.items()):
        count = max(family_score_counts.get(family, 1), 1)
        mean_scores[family] = total / count

    candidates: List[CandidateScore] = []
    for family in sorted(available - live):
        accepted_hits = accepted_family_counts.get(family, 0)
        reliability_mean = mean_scores.get(family, 0.5)

        expected = min(1.0, 0.45 + 0.1 * min(accepted_hits, 4) + 0.45 * reliability_mean)
        risk = 0.55
        evidence_bits: List[str] = []

        if family in ignored:
            evidence_bits.append("ignored_by_live_policy")
            risk += 0.05
        if accepted_hits > 0:
            evidence_bits.append(f"accepted_features={accepted_hits}")
            risk -= 0.08
        if reliability_mean < 0.55:
            evidence_bits.append(f"low_reliability={reliability_mean:.2f}")
            risk += 0.15

        if family in {"derivatives", "onchain"}:
            risk += 0.12
        if family == "macro":
            risk -= 0.05
        if family == "forward_return_proxy":
            risk = 0.95
            expected = min(expected, 0.2)
            evidence_bits.append("potential_leakage_proxy")

        if leakage_payload.get("multi_horizon_leakage_warning") == "present" and family in {
            "forward_return_proxy",
            "state_engineering",
        }:
            evidence_bits.append("leakage_rerun_requires_revalidation")
            risk += 0.08

        parity_gain = min(1.0, 0.6 + 0.4 * (1.0 if family in ignored else 0.5))
        risk = max(0.0, min(1.0, risk))

        recommendation = "defer"
        if expected >= 0.72 and risk <= 0.62:
            recommendation = "pilot_shadow"
        elif expected >= 0.6 and risk <= 0.75:
            recommendation = "instrument_first"

        candidates.append(
            CandidateScore(
                family=family,
                expected_value=round(expected, 3),
                implementation_risk=round(risk, 3),
                parity_gain=round(parity_gain, 3),
                evidence=",".join(evidence_bits) if evidence_bits else "limited_evidence",
                recommendation=recommendation,
            )
        )

    candidates.sort(
        key=lambda c: (
            -(0.55 * c.expected_value + 0.30 * c.parity_gain - 0.45 * c.implementation_risk),
            c.family,
        )
    )
    return candidates


def build_parity_audit(
    *,
    horizons: Sequence[float],
    models_root: Path,
    ignored_columns: Sequence[str],
    ignored_sources: Sequence[str],
    max_source_lag_hours: float,
    reliability_path: Path = RELIABILITY_PATH,
    featurelift_report_path: Path = LEAKAGE_SAFE_REPORT_PATH,
) -> Dict[str, Any]:
    horizon_sets = _collect_horizon_feature_sets(models_root, horizons)
    all_training_features = sorted(
        {
            name
            for payload in horizon_sets.values()
            for name in payload.get("training_union", [])
        }
    )

    # Feature engineering sources available in code, independent of model selection.
    available_families = {
        "price_core",
        "volume_core",
        "volatility",
        "order_flow",
        "liquidity_structure",
        "state_engineering",
        "intrabar",
        "macro",
        "onchain",
        "derivatives",
        "forward_return_proxy",
    }

    training_family_counts = family_counts(all_training_features)
    live_feature_names = all_training_features
    live_family_counts = family_counts(live_feature_names)

    ignored_feature_families = {
        classify_feature_family(name)
        for name in ignored_columns
        if str(name).strip()
    }
    stale_tolerated_families = {
        classify_source_family(source)
        for source in ignored_sources
        if str(source).strip()
    }
    zero_imputed_families = set(ignored_feature_families)

    live_enforced_families = {
        family
        for family in live_family_counts
        if family not in ignored_feature_families and family not in stale_tolerated_families
    }

    reliability_payload = _read_reliability_payload(reliability_path)
    leakage_payload = _parse_featurelift_report(featurelift_report_path)

    candidates = rank_candidates(
        available_families=available_families,
        live_enforced_families=live_enforced_families,
        ignored_families=ignored_feature_families | stale_tolerated_families,
        reliability_payload=reliability_payload,
        leakage_payload=leakage_payload,
    )

    by_horizon: Dict[str, Any] = {}
    for horizon_label, payload in horizon_sets.items():
        features = payload.get("training_union", [])
        by_horizon[horizon_label] = {
            "training_feature_count": len(features),
            "training_family_counts": family_counts(features),
            "ignored_feature_count": sum(1 for name in features if classify_feature_family(name) in ignored_feature_families),
            "stale_tolerated_feature_count": sum(
                1 for name in features if classify_feature_family(name) in stale_tolerated_families
            ),
            "ignored_features": sorted(
                [name for name in features if classify_feature_family(name) in ignored_feature_families]
            )[:80],
            "stale_tolerated_features": sorted(
                [name for name in features if classify_feature_family(name) in stale_tolerated_families]
            )[:80],
        }

    return {
        "horizons": [float(v) for v in horizons],
        "available_feature_families": sorted(available_families),
        "training_family_counts": training_family_counts,
        "live_family_counts": live_family_counts,
        "live_enforced_families": sorted(live_enforced_families),
        "ignored_families": sorted(ignored_feature_families),
        "stale_tolerated_families": sorted(stale_tolerated_families),
        "zero_imputed_families": sorted(zero_imputed_families),
        "feature_coverage_policy": {
            "ignored_sources": sorted({str(v).strip().lower() for v in ignored_sources if str(v).strip()}),
            "ignored_columns": sorted({str(v).strip() for v in ignored_columns if str(v).strip()}),
            "max_source_lag_hours": float(max_source_lag_hours),
        },
        "leakage_safe_evidence": leakage_payload,
        "reliability_evidence": {
            "accepted_feature_count": len(reliability_payload.get("accepted_features", [])),
            "accepted_family_counts": family_counts(reliability_payload.get("accepted_features", [])),
        },
        "source_family_artifacts": load_source_family_artifacts(["macro", "onchain"]),
        "by_horizon": by_horizon,
        "likely_untapped_candidates": [
            {
                "rank": idx + 1,
                "family": item.family,
                "expected_value": item.expected_value,
                "implementation_risk": item.implementation_risk,
                "parity_gain": item.parity_gain,
                "recommendation": item.recommendation,
                "evidence": item.evidence,
            }
            for idx, item in enumerate(candidates)
        ],
        "notes": [
            "Audit is visibility-only; no live policy or threshold changes are applied.",
            "Multi-horizon leakage-safe rerun indicates prior apparent edge in 4h/8h/12h was likely leakage-driven.",
            "Families listed as stale_tolerated or ignored are currently exempt from live feature coverage blocking.",
        ],
    }


def render_markdown_report(payload: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Train/Live Feature Parity Audit")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Incremental visibility artifact only; no live trading behavior changes.")
    lines.append("- Ground truth includes leakage-safe rerun artifacts and reliability snapshot evidence.")
    lines.append("")

    lines.append("## Family Inventory")
    lines.append(f"- Available families in code: {', '.join(payload.get('available_feature_families', []))}")
    lines.append(f"- Live-enforced families: {', '.join(payload.get('live_enforced_families', []))}")
    lines.append(f"- Ignored families: {', '.join(payload.get('ignored_families', []))}")
    lines.append(f"- Stale-tolerated families: {', '.join(payload.get('stale_tolerated_families', []))}")
    lines.append(f"- Zero-imputed families: {', '.join(payload.get('zero_imputed_families', []))}")
    lines.append("")

    leakage = payload.get("leakage_safe_evidence", {}) if isinstance(payload.get("leakage_safe_evidence"), Mapping) else {}
    lines.append("## Leakage-Safe Ground Truth")
    lines.append(f"- Leakage-safe rerun report found: {bool(leakage.get('report_found'))}")
    lines.append(f"- Multi-horizon leakage warning: {leakage.get('multi_horizon_leakage_warning', 'unknown')}")
    lines.append("- Interpretation: treat multi-horizon prior lift as non-robust until revalidated under leakage-safe constraints.")
    lines.append("")

    lines.append("## Top Opportunities")
    lines.append("| Rank | Family | Expected Value | Risk | Recommendation | Evidence |")
    lines.append("| --- | --- | ---: | ---: | --- | --- |")
    candidates = payload.get("likely_untapped_candidates", [])
    if isinstance(candidates, list):
        for row in candidates[:8]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {rank} | {family} | {value:.3f} | {risk:.3f} | {rec} | {evidence} |".format(
                    rank=int(row.get("rank", 0)),
                    family=str(row.get("family", "unknown")),
                    value=float(row.get("expected_value", 0.0)),
                    risk=float(row.get("implementation_risk", 0.0)),
                    rec=str(row.get("recommendation", "n/a")),
                    evidence=str(row.get("evidence", "n/a")),
                )
            )
    lines.append("")

    lines.append("## Recommended Order")
    lines.append("1. Instrument and validate macro family enforcement path in shadow mode (no live gating change yet).")
    lines.append("2. Promote high-coverage state-engineering signals from wired-but-ignored to measurable shadow enforcement.")
    lines.append("3. Re-test derivatives/on-chain families only after coverage freshness and missingness constraints are tightened.")
    lines.append("")

    return "\n".join(lines) + "\n"
