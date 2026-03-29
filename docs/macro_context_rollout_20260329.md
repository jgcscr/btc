# Macro Context Rollout 2026-03-29

This note closes out the free macro-context rollout for dollar strength, US10Y, and EUR/USD.

## 1. Source Recommendation

Primary recommendation:

| Series | Primary source | Exact code | Why this source |
| --- | --- | --- | --- |
| Dollar strength | FRED | `DTWEXBGS` | Stable free API-style CSV, deep history, clean automation. Used as the operational proxy instead of exact DXY. |
| US 10Y yield | FRED | `DGS10` | Stable free Treasury series with deep history. |
| EUR/USD | Frankfurter backed by ECB reference rates | `EUR/USD` | Free daily FX API with simple automation and no auth. |

Fallbacks:

- Dollar strength: Yahoo Finance `DX-Y.NYB`, Stooq market-data downloads.
- US10Y: Yahoo Finance `^TNX`, U.S. Treasury daily curve download.
- EUR/USD: ECB reference-rate download, Yahoo Finance `EURUSD=X`.

Operational decision:

- Exact DXY was not pursued in the implementation.
- The repo now uses `DTWEXBGS` as the free dollar-strength proxy because it is materially easier to automate and maintain without paid market-data dependencies.

## 2. Implementation

New code:

- `src/data/macro_loader.py`
- `src/scripts/refresh_macro_features.py`

Updated integration points:

- `src/scripts/build_training_dataset.py`
- `src/scripts/build_training_dataset_direction.py`
- `src/scripts/build_training_dataset_multi_horizon.py`
- `src/scripts/build_training_dataset_15m.py`
- `src/scripts/run_refresh_and_predict.py`
- `README.md`

Stored outputs:

- `data/processed/macro/daily_features.parquet`
- `data/processed/macro/source_manifest.json`

The macro parquet contains:

- raw levels: `macro_dollar_proxy`, `macro_us10y`, `macro_eurusd`
- one-day change features for each series
- 30-day rolling z-score features for each series
- 5-day trend features for each series

## 3. Timing And Leakage Policy

Daily macro rows are not merged at the source date directly.

Instead:

- `macro_source_date` records the original business date from FRED or Frankfurter.
- `ts` is set to `macro_source_date + 1 day` at `00:00:00 UTC`.

That rule is intentionally conservative. It avoids treating a daily macro observation as usable for intraday BTC decisions on the same date when publication timing may differ by source.

## 4. Pipeline Integration

The repo now treats macro as contextual features, not hard trading overrides.

Integration path:

1. `src.scripts.refresh_macro_features` builds the normalized macro parquet.
2. The processed macro parquet is included in the shared `PROCESSED_PATHS` used by hourly, multi-horizon, and 15m dataset builders.
3. Macro features are preserved through the external-source pruning step and appear in dataset `feature_names`.
4. `run_refresh_and_predict` also accepts a `--macro-path` local override and now treats it as a real merge input rather than metadata-only.

Verified artifact result:

- `artifacts/datasets/btc_features_1h_splits.npz` includes 12 macro feature columns.
- `artifacts/datasets/btc_features_multi_horizon_splits.npz` includes the same 12 macro feature columns.
- `artifacts/datasets/btc_features_15m_splits.npz` rebuilt successfully against the final code state.

## 5. Lightweight Validation

Executed artifact:

- `artifacts/analysis/macro_ablation_20260329.json`

Method used:

- A lightweight out-of-sample sanity study on `artifacts/datasets/btc_features_multi_horizon_splits.npz`.
- Logistic-regression classifiers were fit on train+validation splits and evaluated on the held-out test split.
- Variants compared: baseline without macro columns, baseline with macro columns, and macro-only.

Observed result:

- Macro-only performance was near random across horizons, which argues strongly against using these macro inputs as standalone directional rules.
- Baseline with macro was almost identical to baseline without macro.
- The delta was too small and inconsistent to justify promotion from this quick study.

Important limitation:

- The baseline scores from this quick study were unrealistically strong overall, which means this specific diagnostic should not be treated as a production-grade promotion result.
- In other words, the quick ablation is useful for relative direction only: macro-only looks weak, and adding macro did not produce a compelling incremental lift.

Raw redundancy check from the same study:

- `macro_dollar_proxy` and `macro_eurusd` were strongly negatively correlated at roughly `-0.95` on the fitted sample.
- That is consistent with the expected overlap and supports the view that dollar proxy plus EUR/USD can be partially redundant.

## 6. Recommendation

Current recommendation:

- Keep the new macro bundle as an experimental contextual source.
- Do not promote any macro-driven policy rule such as “DXY down implies BTC bullish.”
- Do not assign discretionary fixed weights to DXY proxy, US10Y, and EUR/USD inside the automated stack.
- For the current Binance-only live runtime, treat macro columns as optional context rather than required feature-coverage inputs.
- Restore macro as a required live dependency only after every approved runtime profile merges a maintained macro parquet or equivalent live source bundle on every refresh and the feature-coverage gate is intentionally re-tightened.
- If macro is evaluated further, use the repository’s existing reliability and walk-forward workflow rather than the lightweight sanity study above.

## 7. Commands Used

Build or refresh the macro bundle:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.refresh_macro_features --full-refresh
```

Rebuild the main datasets with macro integrated:

```bash
/workspaces/btc/.venv/bin/python -m src.scripts.build_training_dataset --output-dir artifacts/datasets
/workspaces/btc/.venv/bin/python -m src.scripts.build_training_dataset_multi_horizon --output-dir artifacts/datasets
/workspaces/btc/.venv/bin/python -m src.scripts.build_training_dataset_15m --output-dir artifacts/datasets
```

Run the focused macro tests:

```bash
/workspaces/btc/.venv/bin/python -m unittest tests.test_macro_loader_and_integration
```
