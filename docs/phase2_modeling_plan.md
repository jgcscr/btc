# Phase 2 Modeling Execution Plan

## A. Objectives
- **Refreshed XGBoost ensembles:** Rebuild 1h/4h regression + direction models using enriched feature sets, recalibrated thresholds, and updated guardrails.
- **Sequence models:** Evaluate LSTM and transformer architectures leveraging expanded deltas/z-scores to capture regime changes and longer-term dependencies.
- **Hybrid ensemble:** Assess stacking/blending strategies combining tree-based and sequence learners for improved recall without excessive drawdown.

## B. Dataset & Feature Requirements
- **Base datasets:** artifacts/datasets/btc_features_multi_horizon_splits.npz and artifacts/datasets/btc_features_1h_direction_splits.npz refreshed with augmented features.
- **Sequence length:** 32–64 timesteps (hourly) for sequence models; retain 7/24-hour windows for tree-based baselines.
- **Normalization:** StandardScaler on training splits for tree models; per-feature z-score/LayerNorm for sequence inputs; ensure consistent scaling via metadata.
- **Feature set:** Include price/futures deltas, pct-change, z-scores, macro pct-change, funding deltas, CryptoQuant derived signals; drop constant columns via `_drop_constant_features`.
- **Label hygiene:** Ensure ret targets align with truncated rows post-merge; maintain alignment for 1h/4h horizons.

## C. Optuna Search Strategy
- **Search space:**
  - XGBoost: max_depth (3–9), learning_rate (0.01–0.2), subsample (0.6–1.0), colsample_bytree (0.5–1.0), min_child_weight (1–10).
  - LSTM: hidden_dim (64–256), num_layers (1–3), dropout (0.0–0.3), sequence_len (32–64).
  - Transformer: d_model (64–256), num_heads (2–8), num_layers (2–4), feedforward_dim (128–512), dropout (0.0–0.3).
- **Budget:** 150 trials per family (XGB/LSTM/Transformer) with pruning via median/Hyperband; checkpoint best artifact.
- **Metrics:** Optimize F1/up recall with drawdown penalty (custom objective), log hit rate and cumulative return for monitoring.
- **Hardware:** GPU-enabled runners (A10/3090) for sequence models; CPU pool for XGB; ensure reproducible seeds logged in study metadata.

## D. Validation Approach
- **Walk-forward splits:** Rolling 70/15/15 windows per quarter for 2023–2025 to evaluate temporal generalization.
- **Backtests:** Regenerate 1h, 4h, and 1h+4h confirmation equity curves with new models; compare net metrics vs. Phase 1 baselines.
- **Guardrails:** Enforce minimum trades (>=200), hit rate >=0.52, max drawdown <=-0.35 (4h) / <=-0.10 (1h) before promotion.
- **Stress scenarios:** Re-run extreme volatility periods (Nov-2022 FTX, Mar-2023 banking shock) to ensure stability.
- **Documentation:** Capture Optuna study summaries, parameter importances, and final thresholds in artifacts/.

## E. Timeline & Milestones
| Milestone | Owner | Target Date |
| --- | --- | --- |
| Dataset refresh w/ augmented features & QA | Data Eng | 2026-01-07 |
| XGBoost Phase 2 Optuna study & backtest review | Quant Lead | 2026-01-17 |
| LSTM prototype training + evaluation report | ML Engineer | 2026-01-24 |
| Transformer experiment & ensemble comparison | ML Engineer | 2026-01-31 |
| Guardrail codification & automation integration | Platform Eng | 2026-02-05 |
| Phase 2 sign-off & automation handoff | Program Manager | 2026-02-10 |
