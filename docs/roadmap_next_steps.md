# BTC Prediction System Roadmap

## Executive Summary
The current BTC trading stack reliably produces hourly ensemble signals augmented with optional 4h confirmation, supported by curated BigQuery datasets, paper-trade logs, and reproducible backtests. Recent analysis highlights headroom in three areas: richer features (on-chain, macro, order-book detail), modern modeling (LSTM/transformer hybrids, volatility-aware ensembles), and tighter production guardrails (automated validation, monitoring, deployment automation). This roadmap sequences the work so each phase compounds toward a production-grade, research-driven pipeline.

## Prioritized Backlog

### Phase 1 – Data Foundations
- **Objective:** Expand and harden the feature layer.
- **Key Actions:**
  - Add on-chain, macro, and funding-rate loaders; schedule ingestion.
  - Standardize feature versioning and data quality checks.
  - Produce refreshed NPZ artifacts (1h, multi-horizon, new domains).
- **Outputs:** New `data/` loaders, updated SQL/feature specs, data validation reports.
- **Dependencies & Risks:** External API stability, cost of historical backfill, need for schema evolution in BigQuery.

### Phase 2 – Modeling Enhancements
- **Objective:** Upgrade predictive models beyond gradient boosted trees.
- **Key Actions:**
  - Implement LSTM/GRU training scripts with sequence datasets.
  - Add Optuna-driven hyperparameter search orchestration.
  - Prototype hybrid ensembles (tree + deep) with stacking meta-learner.
- **Outputs:** New training scripts, tracked experiment configs, serialized model artifacts, Optuna study logs.
- **Dependencies & Risks:** GPU availability, training time budgeting, risk of overfitting without walk-forward controls.

### Phase 3 – Evaluation & Governance
- **Objective:** Institutionalize repeatable validation and guardrails.
- **Key Actions:**
  - Extend walk-forward schedule (monthly + quarterly windows).
  - Build regression test suite comparing new models vs baseline metrics.
  - Automate metric dashboards and anomaly alerts (e.g., Airflow/Great Expectations).
- **Outputs:** CI jobs, regression test harness, dashboards, alerting playbooks.
- **Dependencies & Risks:** Requires finalized data schemas, integration with monitoring stack.

### Phase 4 – Deployment & Operations
- **Objective:** Operationalize live trading with resilience.
- **Key Actions:**
  - Containerize realtime service, add blue/green deploy scripts.
  - Implement model registry with version pinning and rollback hooks.
  - Add live monitoring (latency, fill rates) and PagerDuty-style alerts.
- **Outputs:** Docker images, deployment runbooks, monitoring dashboards, incident response SOPs.
- **Dependencies & Risks:** Production exchange limits, credentials management, incident response staffing.

## Near-Term Task Queue (Next 4–6 Weeks)
1. Build on-chain data loader (Glassnode/DefiLlama) and integrate into nightly ingestion.
2. Extend dataset builder to include volatility and funding-rate features; regenerate NPZ splits.
3. Add LSTM sequence training script using existing multi-horizon dataset; log experiments via MLflow.
4. Stand up Optuna hyperparameter sweep for XGBoost + LSTM thresholds with budgeted compute.
5. Implement ensemble orchestration script that blends tree + neural predictions with configurable weights.
6. Enhance monitoring scripts with automated alert routing (Slack/webhook) and retry logic.

## Medium-Term Initiatives (1–3 Quarters)
- Transformer-based sequence encoder (Temporal Fusion Transformer or Informer) for multi-horizon forecasting.
- Hybrid CNN-LSTM model capturing local order-book microstructure features.
- Volatility regime classifier integrated into signal gating (switch thresholds dynamically).
- Feature store abstraction with point-in-time correctness validation.
- Realistic transaction cost model including slippage curves by liquidity regime.

## Long-Term Vision (12+ Months)
- Multi-asset portfolio optimization engine sharing core feature + model pipeline.
- Reinforcement learning layer for position sizing under risk constraints.
- Adaptive ensemble governance with automated A/B tests and Bayesian model averaging.
- Fully automated disaster-recovery deployment (multi-region + exchange failover).

## Testing & Validation Milestones
- Monthly walk-forward backtest refresh comparing new models vs baseline KPIs (hit rate, net return, drawdown).
- Automated regression suite triggered on every model artifact update (pytest + custom metrics diff).
- Live monitoring checkpoints: hourly sanity checks (feature drift, signal distribution), daily reconciliation with exchange fills, weekly postmortem review of deviations.
- Quarterly chaos drills simulating data outages and latency spikes to verify failover procedures.
