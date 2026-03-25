# Direction Enhancement Closeout (2026-03-25)

## Scope
This note closes the direction-enhancement iteration performed on 2026-03-25, including:
- policy/profile variants
- model-candidate training variants
- replay and shadow-cadence validation

## Final Operational Baseline
- Active shadow lhs profile: `configs/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop.yaml`
- Shadow cadence wiring: `scripts/run_cadence.sh` uses `shadow_direction_enhanced_relaxed_chop` vs `shadow_chop_suppression`

## Final Shadow Cadence Check
- Command: `bash scripts/run_cadence.sh shadow`
- Latest run id: `20260325T224348Z`
- Manifest: `artifacts/predictions/comparisons/shadow_profile_comparison_manifest_20260325T224348Z.json`
- Comparison: `artifacts/predictions/comparisons/shadow_direction_enhanced_relaxed_chop_vs_shadow_chop_suppression_20260325T224348Z.json`

### Outcome Summary
- `profiles_differ`: true
- `operationally_meaningful_difference`: true
- `operational_diff_horizons`: 15m, 1h, 4h, 8h
- `either_profile_actionable`: true
- `lhs_actionable_horizons`: 15m
- `rhs_actionable_horizons`: none

Interpretation:
- The enhanced relaxed-chop lhs still differs materially from chop-suppression.
- Current snapshot still contains blocking reasons across major horizons (coherence, confluence, execution).
- This does not support additional live-promotion changes beyond the current shadow baseline at this time.

## Labeling Of Experimental Assets
These are experimental and should be treated as non-promoted test assets:

### Experimental Policy Profiles
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_coherence_relief.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_gate_relief.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_relaxed_neutral.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_relaxed_broad.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop_unblock.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_relaxed_chop_aggressive.yaml`

### Experimental Model-Candidate Profiles
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_model_candidate.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_model_candidate_threshold_aligned.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_model_candidate_probscale.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_model_candidateonly.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_model_candidateonly_threshold047.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_model_candidate165.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_candidate165_probe.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_candidate165_allcal_probe.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_enhanced_candidate165_relaxedbins_probe.yaml`
- `configs/archive/direction_enhancement_20260325/run_refresh_and_predict.shadow_direction_model165_relaxedbins_th_0p003.yaml`
- `configs/archive/direction_enhancement_20260325/tmp_model165_threshold_0p004.yaml`

### Experimental Model Artifacts
- `artifacts/models/trade_decision_model_direction_candidate_20260325.json`
- `artifacts/models/trade_decision_model_direction_candidateonly_20260325.json`
- `artifacts/models/trade_decision_model_direction_candidate165_20260325.json`
- `artifacts/models/trade_decision_model_direction_candidate165_allcal_20260325.json`
- `artifacts/models/trade_decision_model_direction_candidate165_relaxedbins_20260325.json`

### Experimental Replay Outputs
- `artifacts/tmp_validation/shadow_direction_model_candidate_replay_20260325`
- `artifacts/tmp_validation/shadow_direction_model_candidate_threshold_aligned_replay_20260325`
- `artifacts/tmp_validation/shadow_direction_model_candidate_probscale_replay_20260325`
- `artifacts/tmp_validation/shadow_direction_model_candidateonly_replay_20260325`
- `artifacts/tmp_validation/shadow_direction_model_candidateonly_threshold047_replay_20260325`
- `artifacts/tmp_validation/shadow_direction_model_candidate165_probe_replay_20260325`
- `artifacts/tmp_validation/shadow_direction_model_candidate165_threshold_sweep_0p004`
- `artifacts/tmp_validation/model165_threshold_search_0p006_20260325`
- `artifacts/tmp_validation/model165_threshold_search_0p008_20260325`
- `artifacts/tmp_validation/model165_threshold_search_0p01_20260325`
- `artifacts/tmp_validation/model165_relaxedbins_threshold_search_0p003_20260325`
- `artifacts/tmp_validation/candidate165_probe_20260325`
- `artifacts/tmp_validation/candidate165_allcal_probe_20260325`
- `artifacts/tmp_validation/candidate165_relaxedbins_probe_20260325`

## Final Verdict
- Keep current shadow baseline as-is.
- Do not promote experimental model-candidate profiles from this batch.
- Treat this iteration as complete and archived for traceability.
