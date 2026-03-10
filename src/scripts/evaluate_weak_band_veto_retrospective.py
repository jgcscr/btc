from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _read_csv_or_parquet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _pair_frames(
    default_df: pd.DataFrame,
    mode_frames: Dict[str, pd.DataFrame],
    incumbent_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, str]:
    has_ts = ("ts" in default_df.columns) and ("ts" in incumbent_df.columns) and all(
        "ts" in frame.columns for frame in mode_frames.values()
    )
    if has_ts:
        d = default_df.copy()
        i = incumbent_df.copy()
        d["_ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce").dt.floor("h")
        i["_ts"] = pd.to_datetime(i["ts"], utc=True, errors="coerce").dt.floor("h")
        d = d.dropna(subset=["_ts"]).drop_duplicates(subset=["_ts"], keep="last")
        i = i.dropna(subset=["_ts"]).drop_duplicates(subset=["_ts"], keep="last")
        merged = d.copy()
        default_cols = [c for c in merged.columns if c != "_ts"]
        merged = merged.rename(columns={c: f"{c}_default" for c in default_cols})
        for mode_name, frame in mode_frames.items():
            s = frame.copy()
            s["_ts"] = pd.to_datetime(s["ts"], utc=True, errors="coerce").dt.floor("h")
            s = s.dropna(subset=["_ts"]).drop_duplicates(subset=["_ts"], keep="last")
            mode_cols = [c for c in s.columns if c != "_ts"]
            s = s.rename(columns={c: f"{c}_{mode_name}" for c in mode_cols})
            merged = merged.merge(s, on="_ts", how="inner")
        merged = merged.merge(i, on="_ts", how="inner")
        return merged, "timestamp_hour"

    n = int(min([len(default_df), len(incumbent_df), *[len(frame) for frame in mode_frames.values()]]))
    d = default_df.tail(n).reset_index(drop=True)
    i = incumbent_df.tail(n).reset_index(drop=True)
    merged = pd.DataFrame(index=np.arange(n))
    for col in d.columns:
        merged[f"{col}_default"] = d[col].to_numpy()
    for mode_name, frame in mode_frames.items():
        s = frame.tail(n).reset_index(drop=True)
        for col in s.columns:
            merged[f"{col}_{mode_name}"] = s[col].to_numpy()
    for col in i.columns:
        merged[col] = i[col].to_numpy()
    return merged, "tail_index"


def _bootstrap_stats(candidate: np.ndarray, incumbent: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    n = int(min(candidate.size, incumbent.size))
    if n <= 5:
        return {
            "mean_diff": float("nan"),
            "pvalue_one_sided": float("nan"),
            "nonzero_paired_rows": int(0),
            "std_diff": float("nan"),
            "n_pairs": int(n),
        }

    c = candidate[-n:]
    i = incumbent[-n:]
    diff = c - i
    nonzero = int(np.count_nonzero(np.abs(diff) > 0.0))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else float("nan")

    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        sample_idx = rng.integers(0, n, size=n)
        samples[idx] = float(np.mean(diff[sample_idx]))

    mean_diff = float(np.mean(diff))
    pvalue = float(np.mean(samples <= 0.0)) if int(n_boot) > 0 else float("nan")
    return {
        "mean_diff": mean_diff,
        "pvalue_one_sided": pvalue,
        "nonzero_paired_rows": int(nonzero),
        "std_diff": std_diff,
        "n_pairs": int(n),
    }


def _window_metrics(
    *,
    candidate_ret: np.ndarray,
    candidate_signal: np.ndarray,
    incumbent_ret: np.ndarray,
    incumbent_signal: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    stats = _bootstrap_stats(candidate_ret, incumbent_ret, n_boot=n_boot, seed=seed)
    return {
        "candidate_trade_count": int(np.count_nonzero(candidate_signal != 0.0)),
        "incumbent_trade_count": int(np.count_nonzero(incumbent_signal != 0.0)),
        "candidate_net_return_total": float(np.sum(candidate_ret)),
        "incumbent_net_return_total": float(np.sum(incumbent_ret)),
        "mean_diff": float(stats["mean_diff"]),
        "pvalue_one_sided": float(stats["pvalue_one_sided"]),
        "nonzero_paired_rows": int(stats["nonzero_paired_rows"]),
        "std_diff": float(stats["std_diff"]),
        "n_pairs": int(stats["n_pairs"]),
    }


def _build_windows(n_rows: int, window_size: int, step_size: int, min_rows: int) -> List[Tuple[int, int]]:
    if n_rows <= 0:
        return []
    if n_rows < max(int(window_size), int(min_rows)):
        return [(0, n_rows)] if n_rows >= int(min_rows) else []

    windows: List[Tuple[int, int]] = []
    start = 0
    while start < n_rows:
        end = min(start + int(window_size), n_rows)
        if end - start >= int(min_rows):
            windows.append((start, end))
        if end == n_rows:
            break
        start += int(step_size)

    return windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrospective non-gating comparison of default vs broad/refined shadow veto policy outputs.",
    )
    parser.add_argument("--default-candidate", type=Path, required=True)
    parser.add_argument("--broad-shadow-candidate", type=Path, required=True)
    parser.add_argument("--refined-shadow-candidate", type=Path, required=True)
    parser.add_argument("--midband-shadow-candidate", type=Path, required=True)
    parser.add_argument("--raw-ev-shadow-candidate", type=Path, required=True)
    parser.add_argument("--direction-align-shadow-candidate", type=Path, required=True)
    parser.add_argument("--joint-direction-midband-shadow-candidate", type=Path, required=True)
    parser.add_argument("--regime-state-shadow-candidate", type=Path, required=True)
    parser.add_argument("--chop-high-vol-shadow-candidate", type=Path, required=True)
    parser.add_argument("--volatility-only-shadow-candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--incumbent-col", type=str, default="ret_ensemble_net")
    parser.add_argument("--signal-col", type=str, default="signal_ensemble")
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--step-size", type=int, default=24)
    parser.add_argument("--min-rows", type=int, default=80)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--broad-shadow-meta", type=Path, default=None)
    parser.add_argument("--refined-shadow-meta", type=Path, default=None)
    parser.add_argument("--midband-shadow-meta", type=Path, default=None)
    parser.add_argument("--raw-ev-shadow-meta", type=Path, default=None)
    parser.add_argument("--direction-align-shadow-meta", type=Path, default=None)
    parser.add_argument("--joint-direction-midband-shadow-meta", type=Path, default=None)
    parser.add_argument("--regime-state-shadow-meta", type=Path, default=None)
    parser.add_argument("--chop-high-vol-shadow-meta", type=Path, default=None)
    parser.add_argument("--volatility-only-shadow-meta", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.default_candidate,
        args.broad_shadow_candidate,
        args.refined_shadow_candidate,
        args.midband_shadow_candidate,
        args.raw_ev_shadow_candidate,
        args.direction_align_shadow_candidate,
        args.joint_direction_midband_shadow_candidate,
        args.regime_state_shadow_candidate,
        args.chop_high_vol_shadow_candidate,
        args.volatility_only_shadow_candidate,
        args.incumbent,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    default_df = _read_csv_or_parquet(args.default_candidate)
    broad_df = _read_csv_or_parquet(args.broad_shadow_candidate)
    refined_df = _read_csv_or_parquet(args.refined_shadow_candidate)
    midband_df = _read_csv_or_parquet(args.midband_shadow_candidate)
    raw_ev_df = _read_csv_or_parquet(args.raw_ev_shadow_candidate)
    direction_align_df = _read_csv_or_parquet(args.direction_align_shadow_candidate)
    joint_direction_midband_df = _read_csv_or_parquet(args.joint_direction_midband_shadow_candidate)
    regime_state_df = _read_csv_or_parquet(args.regime_state_shadow_candidate)
    chop_high_vol_df = _read_csv_or_parquet(args.chop_high_vol_shadow_candidate)
    volatility_only_df = _read_csv_or_parquet(args.volatility_only_shadow_candidate)
    incumbent_df = _read_csv_or_parquet(args.incumbent)

    mode_frames = {
        "broad_shadow": broad_df,
        "refined_shadow": refined_df,
        "midband_shadow": midband_df,
        "raw_ev_shadow": raw_ev_df,
        "direction_align_shadow": direction_align_df,
        "joint_direction_midband_shadow": joint_direction_midband_df,
        "regime_state_shadow": regime_state_df,
        "chop_high_vol_shadow": chop_high_vol_df,
        "volatility_only_shadow": volatility_only_df,
    }
    paired, pairing = _pair_frames(default_df, mode_frames, incumbent_df)
    if paired.empty:
        raise RuntimeError("No paired rows available for weak-band veto comparison")

    d_ret = pd.to_numeric(paired[f"{args.candidate_col}_default"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    i_ret = pd.to_numeric(paired[args.incumbent_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    d_sig = pd.to_numeric(paired[f"{args.signal_col}_default"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    i_sig = pd.to_numeric(paired[args.signal_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    mode_series: Dict[str, Dict[str, np.ndarray]] = {}
    for mode_name in mode_frames:
        mode_series[mode_name] = {
            "ret": pd.to_numeric(
                paired[f"{args.candidate_col}_{mode_name}"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float),
            "sig": pd.to_numeric(
                paired[f"{args.signal_col}_{mode_name}"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float),
        }

    full_default = _window_metrics(
        candidate_ret=d_ret,
        candidate_signal=d_sig,
        incumbent_ret=i_ret,
        incumbent_signal=i_sig,
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )

    full_modes: Dict[str, Dict[str, Any]] = {
        "default": {**full_default, "vetoed_rows": 0},
    }
    full_deltas_vs_default: Dict[str, Dict[str, Any]] = {}
    for mode_idx, mode_name in enumerate(mode_frames.keys()):
        mode_metrics = _window_metrics(
            candidate_ret=mode_series[mode_name]["ret"],
            candidate_signal=mode_series[mode_name]["sig"],
            incumbent_ret=i_ret,
            incumbent_signal=i_sig,
            n_boot=int(args.n_boot),
            seed=int(args.seed) + 100 + mode_idx,
        )
        mode_vetoed = int(np.count_nonzero((d_sig != 0.0) & (mode_series[mode_name]["sig"] == 0.0)))
        full_modes[mode_name] = {**mode_metrics, "vetoed_rows": mode_vetoed}
        full_deltas_vs_default[mode_name] = {
            "candidate_net_return_total": float(
                mode_metrics["candidate_net_return_total"] - full_default["candidate_net_return_total"]
            ),
            "mean_diff": float(mode_metrics["mean_diff"] - full_default["mean_diff"]),
            "candidate_trade_count": int(mode_metrics["candidate_trade_count"] - full_default["candidate_trade_count"]),
        }

    current_window = {
        "modes": full_modes,
        "deltas_vs_default": full_deltas_vs_default,
    }

    windows = _build_windows(
        n_rows=len(d_ret),
        window_size=int(args.window_size),
        step_size=int(args.step_size),
        min_rows=int(args.min_rows),
    )

    retrospective: List[Dict[str, Any]] = []
    for idx, (start, end) in enumerate(windows):
        wd_ret = d_ret[start:end]
        wi_ret = i_ret[start:end]

        wd_sig = d_sig[start:end]
        wi_sig = i_sig[start:end]

        wd = _window_metrics(
            candidate_ret=wd_ret,
            candidate_signal=wd_sig,
            incumbent_ret=wi_ret,
            incumbent_signal=wi_sig,
            n_boot=int(args.n_boot),
            seed=int(args.seed) + 10 + idx,
        )
        window_modes: Dict[str, Dict[str, Any]] = {"default": {**wd, "vetoed_rows": 0}}
        deltas_vs_default: Dict[str, Dict[str, Any]] = {}
        mode_flags: Dict[str, Dict[str, bool]] = {}

        for mode_idx, mode_name in enumerate(mode_frames.keys()):
            ws_ret = mode_series[mode_name]["ret"][start:end]
            ws_sig = mode_series[mode_name]["sig"][start:end]
            ws = _window_metrics(
                candidate_ret=ws_ret,
                candidate_signal=ws_sig,
                incumbent_ret=wi_ret,
                incumbent_signal=wi_sig,
                n_boot=int(args.n_boot),
                seed=int(args.seed) + 1000 + (100 * mode_idx) + idx,
            )
            delta_net = float(ws["candidate_net_return_total"] - wd["candidate_net_return_total"])
            delta_mean = float(ws["mean_diff"] - wd["mean_diff"])
            window_vetoed_rows = int(np.count_nonzero((wd_sig != 0.0) & (ws_sig == 0.0)))
            window_modes[mode_name] = {**ws, "vetoed_rows": window_vetoed_rows}
            deltas_vs_default[mode_name] = {
                "candidate_net_return_total": delta_net,
                "mean_diff": delta_mean,
                "candidate_trade_count": int(ws["candidate_trade_count"] - wd["candidate_trade_count"]),
            }
            mode_flags[mode_name] = {
                "improves_mean_diff": bool(np.isfinite(delta_mean) and delta_mean > 0.0),
                "improves_candidate_net_return_total": bool(np.isfinite(delta_net) and delta_net > 0.0),
                "clearly_harms": bool(np.isfinite(delta_net) and np.isfinite(delta_mean) and delta_net < 0.0 and delta_mean < 0.0),
            }

        retrospective.append(
            {
                "window_id": int(idx + 1),
                "start_row": int(start),
                "end_row_exclusive": int(end),
                "rows": int(end - start),
                "modes": window_modes,
                "deltas_vs_default": deltas_vs_default,
                "mode_flags_vs_default": mode_flags,
            }
        )

    aggregate: Dict[str, Any] = {
        "number_of_evaluated_windows": int(len(retrospective)),
        "modes": {},
    }
    for mode_name in mode_frames:
        mode_vetoed_rows_arr = np.asarray(
            [int(w["modes"][mode_name]["vetoed_rows"]) for w in retrospective],
            dtype=float,
        )
        mode_net_delta_arr = np.asarray(
            [float(w["deltas_vs_default"][mode_name]["candidate_net_return_total"]) for w in retrospective],
            dtype=float,
        )
        mode_mean_delta_arr = np.asarray(
            [float(w["deltas_vs_default"][mode_name]["mean_diff"]) for w in retrospective],
            dtype=float,
        )
        aggregate["modes"][mode_name] = {
            "windows_improved_mean_diff": int(
                sum(1 for w in retrospective if bool(w["mode_flags_vs_default"][mode_name]["improves_mean_diff"]))
            ),
            "windows_improved_candidate_net_return_total": int(
                sum(
                    1
                    for w in retrospective
                    if bool(w["mode_flags_vs_default"][mode_name]["improves_candidate_net_return_total"])
                )
            ),
            "aggregate_delta_candidate_net_return_total": float(np.nansum(mode_net_delta_arr))
            if len(mode_net_delta_arr)
            else float("nan"),
            "aggregate_delta_mean_diff": float(np.nansum(mode_mean_delta_arr)) if len(mode_mean_delta_arr) else float("nan"),
            "mean_delta_mean_diff": float(np.nanmean(mode_mean_delta_arr)) if len(mode_mean_delta_arr) else float("nan"),
            "median_vetoed_rows_per_window": float(np.nanmedian(mode_vetoed_rows_arr))
            if len(mode_vetoed_rows_arr)
            else float("nan"),
            "mean_vetoed_rows_per_window": float(np.nanmean(mode_vetoed_rows_arr)) if len(mode_vetoed_rows_arr) else float("nan"),
            "windows_clearly_harmed": [
                {
                    "window_id": int(w["window_id"]),
                    "delta_candidate_net_return_total": float(w["deltas_vs_default"][mode_name]["candidate_net_return_total"]),
                    "delta_mean_diff": float(w["deltas_vs_default"][mode_name]["mean_diff"]),
                    "vetoed_rows": int(w["modes"][mode_name]["vetoed_rows"]),
                }
                for w in retrospective
                if bool(w["mode_flags_vs_default"][mode_name]["clearly_harms"])
            ],
        }

    shadow_meta_vetoed: Dict[str, Any] = {
        "broad_shadow": None,
        "refined_shadow": None,
        "midband_shadow": None,
        "raw_ev_shadow": None,
        "direction_align_shadow": None,
        "joint_direction_midband_shadow": None,
        "regime_state_shadow": None,
        "chop_high_vol_shadow": None,
        "volatility_only_shadow": None,
    }
    meta_specs = [
        ("broad_shadow", args.broad_shadow_meta),
        ("refined_shadow", args.refined_shadow_meta),
        ("midband_shadow", args.midband_shadow_meta),
        ("raw_ev_shadow", args.raw_ev_shadow_meta),
        ("direction_align_shadow", args.direction_align_shadow_meta),
        ("joint_direction_midband_shadow", args.joint_direction_midband_shadow_meta),
        ("regime_state_shadow", args.regime_state_shadow_meta),
        ("chop_high_vol_shadow", args.chop_high_vol_shadow_meta),
        ("volatility_only_shadow", args.volatility_only_shadow_meta),
    ]
    for mode_name, meta_path in meta_specs:
        if meta_path is None or not meta_path.exists():
            continue
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if mode_name == "broad_shadow":
                weak = meta_payload.get("weak_band_candidate_only_veto", {})
                if isinstance(weak, dict):
                    shadow_meta_vetoed[mode_name] = int(weak.get("vetoed_rows", 0))
            elif mode_name == "refined_shadow":
                refined = meta_payload.get("refined_candidate_only_veto", {})
                if isinstance(refined, dict):
                    shadow_meta_vetoed[mode_name] = int(refined.get("vetoed_rows", 0))
            elif mode_name == "midband_shadow":
                midband = meta_payload.get("midband_candidate_only_veto", {})
                if isinstance(midband, dict):
                    shadow_meta_vetoed[mode_name] = int(midband.get("vetoed_rows", 0))
            elif mode_name == "raw_ev_shadow":
                raw_ev = meta_payload.get("raw_ev_sign_candidate_only_veto", {})
                if isinstance(raw_ev, dict):
                    shadow_meta_vetoed[mode_name] = int(raw_ev.get("vetoed_rows", 0))
            elif mode_name == "direction_align_shadow":
                direction_align = meta_payload.get("direction_align_candidate_only_veto", {})
                if isinstance(direction_align, dict):
                    shadow_meta_vetoed[mode_name] = int(direction_align.get("vetoed_rows", 0))
            elif mode_name == "regime_state_shadow":
                regime_state = meta_payload.get("regime_state_candidate_only_veto", {})
                if isinstance(regime_state, dict):
                    shadow_meta_vetoed[mode_name] = int(regime_state.get("vetoed_rows", 0))
            elif mode_name == "chop_high_vol_shadow":
                chop_high_vol = meta_payload.get("chop_high_vol_candidate_only_veto", {})
                if isinstance(chop_high_vol, dict):
                    shadow_meta_vetoed[mode_name] = int(chop_high_vol.get("vetoed_rows", 0))
            elif mode_name == "volatility_only_shadow":
                volatility_only = meta_payload.get("volatility_only_candidate_only_veto", {})
                if isinstance(volatility_only, dict):
                    shadow_meta_vetoed[mode_name] = int(volatility_only.get("vetoed_rows", 0))
            else:
                joint_direction_midband = meta_payload.get("joint_direction_midband_candidate_only_veto", {})
                if isinstance(joint_direction_midband, dict):
                    shadow_meta_vetoed[mode_name] = int(joint_direction_midband.get("vetoed_rows", 0))
        except Exception:
            shadow_meta_vetoed[mode_name] = None

    output_payload = {
        "default_candidate": str(args.default_candidate),
        "broad_shadow_candidate": str(args.broad_shadow_candidate),
        "refined_shadow_candidate": str(args.refined_shadow_candidate),
        "midband_shadow_candidate": str(args.midband_shadow_candidate),
        "raw_ev_shadow_candidate": str(args.raw_ev_shadow_candidate),
        "direction_align_shadow_candidate": str(args.direction_align_shadow_candidate),
        "joint_direction_midband_shadow_candidate": str(args.joint_direction_midband_shadow_candidate),
        "regime_state_shadow_candidate": str(args.regime_state_shadow_candidate),
        "chop_high_vol_shadow_candidate": str(args.chop_high_vol_shadow_candidate),
        "volatility_only_shadow_candidate": str(args.volatility_only_shadow_candidate),
        "incumbent": str(args.incumbent),
        "pairing": pairing,
        "windowing": {
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "min_rows": int(args.min_rows),
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
        },
        "current_window": current_window,
        "retrospective_windows": retrospective,
        "aggregate": aggregate,
        "shadow_meta_vetoed_rows": shadow_meta_vetoed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
