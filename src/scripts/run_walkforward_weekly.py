import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from src.utils import cloud_io

GOVERNANCE_OUTPUT_URI = os.environ.get("GOVERNANCE_OUTPUT_URI", "artifacts/analysis/governance")


def today_str() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d")


def extract_kpis(report_path: Path) -> Optional[Dict[str, float]]:
    if not report_path.exists():
        return None

    header = None
    last_values = None
    with open(report_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("fold"):
                header = line.split("\t")
            elif header and line[0].isdigit():
                last_values = line.split("\t")
    if not header or not last_values:
        return None

    header_index = {name: idx for idx, name in enumerate(header)}

    def _value(key: str) -> Optional[float]:
        idx = header_index.get(key)
        if idx is None or idx >= len(last_values):
            return None
        try:
            return float(last_values[idx])
        except (TypeError, ValueError):
            return None

    return {
        "equity_ensemble_net": _value("Ens_cum"),
        "drawdown": _value("Ens_dd"),
        "sharpe_like": _value("Ens_sharpe"),
        "hit_rate": _value("Ens_hit"),
    }


def run_walkforward_eval(out_dir: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "src/scripts/walk_forward_eval.py"]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", env.get("PYTHONPATH", os.getcwd()))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=os.getcwd())

    (out_dir / "walkforward.log").write_text(proc.stdout + "\n" + proc.stderr)
    (out_dir / "walkforward.json").write_text(proc.stdout)
    return proc


def load_baseline_kpis(current_date: str) -> Optional[Dict[str, float]]:
    dates = [d for d in cloud_io.list_subdirectories(GOVERNANCE_OUTPUT_URI) if d < current_date]
    for candidate in sorted(dates, reverse=True):
        candidate_uri = cloud_io.join_uri(GOVERNANCE_OUTPUT_URI, f"{candidate}/walkforward.json")
        try:
            local_path, cleanup = cloud_io.resolve_to_local(candidate_uri)
        except (FileNotFoundError, ValueError):
            continue
        try:
            kpis = extract_kpis(Path(local_path))
        finally:
            if cleanup:
                cleanup()
        if kpis:
            return kpis
    return None


def main() -> None:
    current_date = today_str()
    with cloud_io.local_directory(GOVERNANCE_OUTPUT_URI, current_date) as (out_dir, remote_uri):
        proc = run_walkforward_eval(out_dir)
        if proc.returncode != 0:
            print("walk_forward_eval.py exited with non-zero status", file=sys.stderr)

        report_path = out_dir / "walkforward.json"
        if not report_path.exists():
            print(f"walkforward.json not found in {out_dir}; aborting.", file=sys.stderr)
            sys.exit(1)

        kpis = extract_kpis(report_path)
        if not kpis:
            print("Unable to parse KPIs from walkforward output.", file=sys.stderr)
            sys.exit(1)

        baseline = load_baseline_kpis(current_date)
        thresholds = {
            "equity_ensemble_net": 0.01,
            "hit_rate": 0.05,
            "drawdown": 0.05,
            "sharpe_like": 0.1,
        }
        diff: Dict[str, Dict[str, float]] = {}
        if baseline:
            for key, current_value in kpis.items():
                baseline_value = baseline.get(key)
                if current_value is None or baseline_value is None:
                    continue
                delta = current_value - baseline_value
                diff[key] = {
                    "current": current_value,
                    "baseline": baseline_value,
                    "delta": delta,
                    "flag": abs(delta) > thresholds.get(key, 0.05),
                }

        (out_dir / "walkforward_diff.json").write_text(json.dumps(diff, indent=2))

        summary_lines = [f"# Walk-Forward Validation Summary ({current_date})", "", "## Latest KPIs"]
        for key, value in kpis.items():
            summary_lines.append(f"- **{key}**: {value}")
        if baseline and diff:
            summary_lines.extend(["", "## Baseline Comparison"])
            for key, info in diff.items():
                flag = " [ALERT]" if info.get("flag") else ""
                summary_lines.append(
                    f"- {key}: {info['current']} (prev {info['baseline']}, Δ={info['delta']:.4f}){flag}"
                )
        else:
            summary_lines.extend(["", "_No previous baseline found._"])
        (out_dir / "summary.md").write_text("\n".join(summary_lines))

        print(f"Walk-forward validation complete. Results synced to {remote_uri}")


if __name__ == "__main__":
    main()
