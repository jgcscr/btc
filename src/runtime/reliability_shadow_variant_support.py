from __future__ import annotations

from pathlib import Path


REFERENCE_FEATURE_ABLATION_VARIANT = "reference_feature_ablation"
REFERENCE_FEATURE_ABLATION_THRESHOLD_VARIANT_PREFIX = f"{REFERENCE_FEATURE_ABLATION_VARIANT}_threshold_"


def format_threshold_variant_name(threshold: float) -> str:
    normalized = f"{float(threshold):.4f}".rstrip("0").rstrip(".")
    return f"threshold_{normalized.replace('.', 'p')}"


def format_reference_feature_ablation_threshold_variant_name(threshold: float) -> str:
    return f"{REFERENCE_FEATURE_ABLATION_VARIANT}_{format_threshold_variant_name(threshold)}"


def format_reference_feature_ablation_selection_guard_variant_name(threshold: float) -> str:
    return f"{format_reference_feature_ablation_threshold_variant_name(threshold)}_selection_calibration_guard"


def format_reference_feature_ablation_abs_ret_pred_variant_name(threshold: float, floor: float) -> str:
    normalized_floor = f"{float(floor):.5f}".rstrip("0").rstrip(".")
    return (
        f"{format_reference_feature_ablation_threshold_variant_name(threshold)}"
        f"_neutral_abs_ret_pred_floor_{normalized_floor.replace('.', 'p')}"
    )


def format_reference_feature_ablation_neutral_p_up_cap_variant_name(threshold: float, max_p_up: float) -> str:
    normalized_cap = f"{float(max_p_up):.5f}".rstrip("0").rstrip(".")
    return (
        f"{format_reference_feature_ablation_threshold_variant_name(threshold)}"
        f"_neutral_p_up_cap_{normalized_cap.replace('.', 'p')}"
    )


def shadow_variant_uses_reference_feature_ablation_model(official_shadow_variant: str) -> bool:
    normalized = str(official_shadow_variant or "none").strip().lower()
    return normalized == REFERENCE_FEATURE_ABLATION_VARIANT or normalized.startswith(
        REFERENCE_FEATURE_ABLATION_THRESHOLD_VARIANT_PREFIX
    )


def is_supported_official_shadow_variant(official_shadow_variant: str) -> bool:
    normalized = str(official_shadow_variant or "none").strip().lower()
    if normalized in {
        "auto",
        "none",
        REFERENCE_FEATURE_ABLATION_VARIANT,
        "selection_calibration_guard",
        "weak_band",
        "refined",
        "midband",
        "raw_ev_sign",
        "direction_alignment",
        "joint_direction_midband",
        "regime_state",
        "chop_high_volatility",
        "volatility_only",
        "triggered_regime_volatility",
    }:
        return True
    if normalized.startswith("threshold_"):
        return True
    return shadow_variant_uses_reference_feature_ablation_model(normalized)


def official_shadow_overlap_triggered_trade_diag_path(summary_dir: Path, official_shadow_variant: str) -> Path:
    normalized = str(official_shadow_variant or "none").strip().lower()
    if normalized == "none":
        return summary_dir / "overlap_triggered_trade_diagnostics.json"
    return summary_dir / f"overlap_triggered_trade_diagnostics_shadow_{normalized}.json"