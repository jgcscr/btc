# Automation Readiness Checklist

## 1. Current Pipeline Status
- **Data ingest:** Hourly spot/futures/macro feeds refreshing via curated ETL; backfill gaps identified with ~1.2k non-hourly intervals pending smoothing.
- **Feature engineering:** Expanded price deltas/z-scores applied across dataset builders and live prep; legacy features padded for inference.
- **Models:** Latest XGBoost 1h/4h regression + direction bundles deployed with cached metadata; LSTM/transformer optional.
- **Thresholds:** Horizons 1–12 stored in artifacts/predictions/manual/thresholds.json with 4h `p_up_min=0.30` adopted across manual/backtest flows.
- **Backtests:** 1h baseline, 4h standalone, and combined 1h+4h confirmation runs stored under artifacts/analysis/; metrics reviewed for net-of-costs performance.

## 2. Gaps Before Automation
- Complete Phase 2 model refresh (multi-horizon ensemble re-training, calibration audit).
- Add regression/backtest guardrails (min trades, drawdown caps, hit-rate thresholds) with automated test enforcement.
- Implement data-feed health checks and rerun tolerances for curated feature gaps.
- Build alerting coverage (Slack/email) for ingestion failures, model drift, and trading anomalies.
- Formalize thresholds governance (versioning, rollback, approval workflow).
- Harden manual CLI outputs (schema validation, failure retries) for bot consumption.

## 3. Proposed Paper-Trading Architecture
- **Scheduler:** Orchestrate hourly runs (e.g., APScheduler/Cron) triggering feature refresh + signal evaluation; maintains state of last successful cycle.
- **Signal service:** Python microservice wrapping prepare_data_for_signals and compute_signal_for_index, exposing REST/CLI for horizons and ensembles.
- **Broker adapter:** Swappable interface for paper vs. live (start with Binance/Alpaca sandbox) handling order sizing, fee modeling, and order status polling.
- **Portfolio/risk engine:** Applies trade guardrails, position limits, and confirmation rules (1h with 4h gate) before routing to broker.
- **Monitoring/observability:** Logging, metrics (Prometheus/Grafana), run artifacts, and automated alert hooks.
- **Persistence layer:** Store executed trades, signal snapshots, and equity curves (Postgres/BigQuery) for reconciliation.

## 4. Immediate Action Items
| Item | Owner | Target Date |
| --- | --- | --- |
| Draft Phase 2 retraining plan & dataset validation checklist | Quant Lead | 2026-01-10 |
| Implement threshold loader guardrails + regression tests | Platform Eng | 2025-12-22 |
| Prototype scheduler + signal service wrapper (dry run) | Platform Eng | 2026-01-05 |
| Define broker adapter requirements & sandbox credentials | Trading Ops | 2025-12-29 |
| Design monitoring/alerting stack (dashboards + incident playbook) | SRE | 2026-01-08 |
| Document go/no-go criteria for automation pilot | Quant Lead | 2026-01-12 |
