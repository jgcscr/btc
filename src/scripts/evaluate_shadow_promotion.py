from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate model promotion using incumbent vs candidate quality metrics.")
    parser.add_argument("--incumbent", type=Path, required=True, help="Incumbent model_quality.json")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate model_quality.json")
    parser.add_argument("--min-auc-delta", type=float, default=0.002)
    parser.add_argument("--max-brier-increase", type=float, default=0.0)
    parser.add_argument("--max-ece-increase", type=float, default=0.01)
    parser.add_argument(
        "--min-trade-count",
        type=int,
        default=10,
        help="Minimum candidate trade_count required for promotion.",
    )
    parser.add_argument(
        "--trade-count-key",
        type=str,
        default="trade_count",
        help="Metric key in quality JSON used as candidate trade count.",
    )
    parser.add_argument(
        "--min-net-return",
        type=float,
        default=0.0,
        help="Minimum candidate net return required for promotion.",
    )
    parser.add_argument(
        "--net-return-key",
        type=str,
        default="net_return_total",
        help="Metric key in quality JSON used as candidate net return.",
    )
    parser.add_argument(
        "--champion-gate",
        type=Path,
        default=None,
        help="Optional champion-challenger gate JSON; when provided it must have promote=true.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/promotion_gate.json"))
    args = parser.parse_args()

    inc = _load_metrics(args.incumbent)
    cand = _load_metrics(args.candidate)

    auc_delta = float(cand.get("auc", float("nan"))) - float(inc.get("auc", float("nan")))
    brier_delta = float(cand.get("brier", float("nan"))) - float(inc.get("brier", float("nan")))
    ece_delta = float(cand.get("ece_10", float("nan"))) - float(inc.get("ece_10", float("nan")))
    candidate_trade_count = pd.to_numeric(cand.get(args.trade_count_key, float("nan")), errors="coerce")
    candidate_net_return = pd.to_numeric(cand.get(args.net_return_key, float("nan")), errors="coerce")
    champion_promote = None
    if args.champion_gate:
        champion_payload = _load_metrics(args.champion_gate)
        champion_promote = bool(champion_payload.get("promote", False))

    checks = {
        "has_auc_delta": bool(pd.notna(auc_delta)),
        "has_brier_delta": bool(pd.notna(brier_delta)),
        "has_ece_delta": bool(pd.notna(ece_delta)),
        "auc_delta_ok": bool(pd.notna(auc_delta) and auc_delta >= args.min_auc_delta),
        "brier_delta_ok": bool(pd.notna(brier_delta) and brier_delta <= args.max_brier_increase),
        "ece_delta_ok": bool(pd.notna(ece_delta) and ece_delta <= args.max_ece_increase),
        "trade_count_ok": bool(pd.notna(candidate_trade_count) and int(candidate_trade_count) >= int(args.min_trade_count)),
        "net_return_ok": bool(pd.notna(candidate_net_return) and float(candidate_net_return) >= float(args.min_net_return)),
    }
    if champion_promote is not None:
        checks["champion_significance_ok"] = bool(champion_promote)

    failed_checks = [name for name, ok in checks.items() if not ok]
    warnings = []
    if pd.notna(candidate_trade_count) and int(candidate_trade_count) == 0:
        warnings.append("no_ensemble_trades_in_evaluation_window")

    promote = len(failed_checks) == 0

    payload = {
        "promote": bool(promote),
        "auc_delta": float(auc_delta),
        "brier_delta": float(brier_delta),
        "ece_delta": float(ece_delta),
        "candidate_trade_count": None if pd.isna(candidate_trade_count) else int(candidate_trade_count),
        "candidate_net_return": None if pd.isna(candidate_net_return) else float(candidate_net_return),
        "champion_promote": champion_promote,
        "failed_checks": failed_checks,
        "warnings": warnings,
        "thresholds": {
            "min_auc_delta": args.min_auc_delta,
            "max_brier_increase": args.max_brier_increase,
            "max_ece_increase": args.max_ece_increase,
            "min_trade_count": int(args.min_trade_count),
            "trade_count_key": args.trade_count_key,
            "min_net_return": float(args.min_net_return),
            "net_return_key": args.net_return_key,
            "champion_gate": str(args.champion_gate) if args.champion_gate else None,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if not promote:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
