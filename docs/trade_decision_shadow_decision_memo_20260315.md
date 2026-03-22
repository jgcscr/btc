## Trade-Decision Shadow Decision Memo

Historical shadow-selection memo.

This file records why the shadow decision path looked the way it did on 2026-03-15 and 2026-03-16. It is not the current source of truth for active deployment, current workflow routing, or present-day operator action.

Use these current sources instead:

- `artifacts/monitoring/reliability_promotion_deploy_manifest.json`
- `docs/operations_runbook.md`
- `docs/agent_system_handoff_20260320.md`

Date: 2026-03-15
Runs reviewed: `20260315T200516Z`, `20260315T204343Z`, `20260315T210636Z`, `20260316T004726Z`

Decision:

- Keep the base policy as the operational default.
- Keep the reference-feature ablation branch in shadow only.
- Tune ablation conservatively through boundary shadow thresholds before considering any broader policy change.

Why this is the current best-practice choice:

- The base candidate is weak, but materially less bad than the raw ablation branch on the same run.
- The ablation branch recovered coverage mainly by admitting many more low-confidence chop trades.
- Those extra trades degraded both economics and recent active-trade calibration, which is exactly the failure mode the promotion gates are supposed to reject.

Evidence from the reviewed run:

- Base candidate: `13` trades, `net_return_total = -0.0151`, rolling delta versus incumbent `+0.0061`, companion gate not significant.
- Ablation candidate: `101` trades, `net_return_total = -0.0923`, rolling delta versus incumbent `-0.0711`, companion gate not significant.
- Ablation recent active-trade calibration: `61` rows, `auc = 0.4513`, `ece_10 = 0.0663`, `ece_drift = 0.0441`.
- Ablation regime concentration: most triggered rows shifted into `chop`, and chop trades were the main source of losses.

Approved tuning direction:

- Do not relax promotion gates.
- Do not replace the live default with ablation.
- Add ablation-specific threshold variants in official shadow selection so the workflow can test whether stricter thresholds recover calibration and economics without changing the deployed baseline.

Fresh tuning result from runs `20260315T204343Z` and `20260315T210636Z`:

- Official replay with ablation thresholds `0.57`, `0.58`, `0.60`, and `0.62` showed those settings were too coarse; each variant collapsed to `0` trades and was useful only as an upper bound.
- A manual boundary evaluation at `0.555` was materially better:
	- companion gate `promote = true`
	- rolling windows `candidate_wins = 5`, `baseline_wins = 0`
	- recent active-trade `auc = 0.5667`
	- recent active-trade `rows = 16`
	- recent active-trade `ece_10 = 0.0971`
	- recent active-trade `ece_drift = 0.0770`
- That means the immediate blocker shifted from economics to calibration. The branch became directionally promising, but it is still not deployable.
- A manual boundary evaluation at `0.56` effectively collapsed the branch again, with `0` recent active rows and no usable rolling comparison.

Finalized replay result from run `20260315T210636Z`:

- The official runtime replay now includes the finalized ablation sweep `0.555` and `0.56` directly from config.
- `official_shadow_variant` remained `none`.
- `reference_feature_ablation_threshold_0p555` became the strongest ablation-derived candidate on economics and significance:
	- `trade_count = 42`
	- `net_return_total = 0.0830`
	- companion `promote = true`, `pvalue_one_sided = 0.0375`
	- rolling delta versus incumbent `+0.1042`
	- recent active-trade `auc = 0.5667`
- It still failed selection because calibration remained outside policy bounds:
	- recent active-trade `ece_10 = 0.0971`
	- `ece_drift = 0.0770`
- `reference_feature_ablation_threshold_0p56` again collapsed to `0` trades and is only useful as an upper diagnostic bound.

Calibration-guard follow-up on run `20260315T210636Z`:

- A diagnostic neutral-only selection guard on top of `reference_feature_ablation_threshold_0p555` with `min_p_up = 0.46` retained positive economics while materially improving calibration error.
- Diagnostic guard result:
	- `trade_count = 34`
	- `net_return_total = 0.0637`
	- rolling delta versus incumbent `+0.0849`
	- recent active-trade `ece_10 = 0.0258`
	- `ece_drift = -0.0251`
- That diagnostic guard still failed the formal promotion bar because recent active-trade support thinned too far:
	- `recent_selection_rows = 11`
	- effective minimum rows `= 14`
	- recent `auc = 0.4`
- This confirms the right diagnosis: the remaining problem is not gross economics, but calibration support and ranking quality inside the retained neutral slice.

Finalized workflow result from run `20260316T004726Z`:

- The runtime workflow now builds the neutral-guard branch as a first-class diagnostic shadow variant: `reference_feature_ablation_threshold_0p555_selection_calibration_guard`.
- `official_shadow_variant` still remained `none`.
- The new diagnostic guard branch became the workflow's `best_ineligible_variant`, which is the correct ranking outcome.
- Final diagnostic guard metrics:
	- `trade_count = 34`
	- `net_return_total = 0.0637`
	- companion `promote = false`, `pvalue_one_sided = 0.07`
	- rolling delta versus incumbent `+0.0849`
	- recent active-trade `ece_10 = 0.0258`
	- `ece_drift = -0.0251`
	- recent active-trade `rows = 11`
	- recent `auc = 0.4`
- This locks in the final conclusion for items 1 and 2: threshold tuning fixed the ablation economics problem, and the calibration guard shows calibration can be fixed, but not yet without losing too much support and ranking quality.

Current operational recommendation:

- Keep `0.555` in the ablation threshold sweep.
- Keep `0.56` as the upper diagnostic bound.
- Keep the neutral `min_p_up = 0.46` guard as a diagnostic shadow variant only; it is informative, but it does not justify relaxing row-count or AUC requirements.
- Do not spend more time on higher ablation thresholds until calibration on the `0.555` boundary is fixed.
- The next tuning target should be calibration and slice quality, especially in the retained neutral-active trades, not further threshold escalation.

Exit criteria for any ablation-derived variant:

- Companion significance must pass.
- Recent active-trade calibration must pass.
- Recent and rolling deltas must be nonnegative.
- The variant must no longer be dominated by incumbent performance.