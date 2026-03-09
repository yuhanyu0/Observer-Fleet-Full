from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .metrics import make_returns, estimate_corr, eig_decomp, spectrum_metrics, affine_invariant_distance, theta_vec, axis_composition, rolling_slope, bootstrap_v1_stability
from .universe import groups_for_universe, CONSTRAINT_MAP
from .utils import ensure_dir, dump_json, pct_rank

@dataclass
class ObserverProtocol:
    observer_id: str
    universe_id: str
    grammar_id: str
    window: int
    estimator: str
    baseline: str = "median_first_year"
    use_log_returns: bool = True
    min_coverage: float = 0.7
    ridge: float = 1e-6
    K: int = 3
    slope_window: int = 20

def _build_ticker_maps(groups):
    tickers = []
    t2g = {}
    for g, xs in groups.items():
        for x in xs:
            if x not in t2g:
                tickers.append(x)
                t2g[x] = g
    groups_order = list(groups.keys())
    return tickers, t2g, groups_order

def run_observer(protocol: ObserverProtocol, prices: pd.DataFrame, outdir: Path, lookback_days: int = 252):
    groups = groups_for_universe(protocol.universe_id)
    tickers, t2g, groups_order = _build_ticker_maps(groups)
    px = prices.reindex(columns=tickers)
    cov = px.notna().mean(axis=0)
    keep = cov[cov >= protocol.min_coverage].index.tolist()
    px = px[keep].ffill()
    tickers = list(px.columns)
    t2g = {t: t2g[t] for t in tickers}
    rets = make_returns(px, use_log=protocol.use_log_returns).dropna(how="all")
    if len(rets) < protocol.window + 5:
        raise RuntimeError(f"{protocol.observer_id}: insufficient data after coverage filter")
    dates = rets.index
    rows = []
    prev_v1 = None
    prev_C = None

    # baseline pool for "median_first_year"
    baseline_C = None
    if protocol.baseline == "median_first_year":
        Cs = []
        first_n = min(len(rets), 252)
        for j in range(protocol.window, first_n):
            X = rets.iloc[j-protocol.window:j].dropna(axis=0, how="any").values
            if X.shape[0] >= max(20, min(40, X.shape[1] + 2)):
                try:
                    C = estimate_corr(X, estimator=protocol.estimator)
                    C = C + protocol.ridge * np.eye(C.shape[0])
                    Cs.append(C)
                except Exception:
                    pass
        if Cs:
            baseline_C = np.median(np.stack(Cs, axis=0), axis=0)

    for j in range(protocol.window, len(rets)):
        dt = dates[j]
        window_df = rets.iloc[j-protocol.window:j]
        missing_rate = float(1.0 - window_df.dropna(axis=0, how="any").shape[0] / max(1, window_df.shape[0]))
        Wdf = window_df.dropna(axis=0, how="any")
        X = Wdf.values
        eff_n = int(X.shape[0])
        if eff_n < max(20, min(40, X.shape[1] + 2)):
            continue
        try:
            C = estimate_corr(X, estimator=protocol.estimator)
            C = C + protocol.ridge * np.eye(C.shape[0])
            w, V = eig_decomp(C)
            mets = spectrum_metrics(w)
            v1 = V[:, 0]
            comp = axis_composition(v1, tickers, t2g, groups_order)
            constraint_comp = {}
            for g, sh in comp.items():
                c = CONSTRAINT_MAP.get(g, "Other")
                constraint_comp[c] = constraint_comp.get(c, 0.0) + float(sh)

            d_fr = np.nan
            if baseline_C is None:
                baseline_C = C.copy()
            if baseline_C is not None:
                d_fr = affine_invariant_distance(baseline_C, C)
            v_fr = np.nan
            theta = np.nan
            if prev_C is not None:
                v_fr = affine_invariant_distance(prev_C, C)
            if prev_v1 is not None:
                theta = theta_vec(prev_v1, v1)

            eigs = np.linalg.eigvalsh(C)
            min_eig = float(np.min(eigs))
            try:
                cond = float(np.linalg.cond(C))
            except Exception:
                cond = np.nan
            bstab = bootstrap_v1_stability(X, estimator=protocol.estimator)

            row = {
                "date": pd.Timestamp(dt),
                "eff_n": eff_n,
                "missing_rate": missing_rate,
                "min_eig": min_eig,
                "cond_number": cond,
                "bootstrap_stability": bstab,
                "s1": mets["s1"],
                "erank": mets["erank"],
                "pr": mets["pr"],
                "gap": mets["gap"],
                "theta_v1": theta,
                "d_fr": d_fr,
                "v_fr": v_fr,
            }
            for g in groups_order:
                row[f"axis_{g}"] = comp.get(g, np.nan)
            for c, sh in constraint_comp.items():
                row[f"constraint_{c}"] = sh
            rows.append(row)

            prev_v1 = v1.copy()
            prev_C = C.copy()
        except Exception:
            continue

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"{protocol.observer_id}: no valid rows")
    # add percentiles and synthetic scores
    for col in ["s1", "v_fr", "theta_v1"]:
        vals = df[col].astype(float).values
        pcts = []
        for x in vals:
            pcts.append(pct_rank(vals, x))
        df[f"{col}_pct"] = pcts
    # concentration proxy: high s1 and low erank/pr
    inv_erank = 1.0 / df["erank"].replace(0, np.nan)
    inv_pr = 1.0 / df["pr"].replace(0, np.nan)
    inv_erank_pct = [pct_rank(inv_erank.values, x) for x in inv_erank.values]
    inv_pr_pct = [pct_rank(inv_pr.values, x) for x in inv_pr.values]
    df["single_axis_score"] = np.nanmean(np.vstack([df["s1_pct"], inv_erank_pct, inv_pr_pct]), axis=0)
    df["hazard_i"] = np.nanmean(np.vstack([df["s1_pct"], df["v_fr_pct"], df["single_axis_score"]]), axis=0)
    # migration proxy: theta + total axis drift
    axis_cols = [c for c in df.columns if c.startswith("axis_")]
    tv = df[axis_cols].diff().abs().sum(axis=1)
    tv_pct = [pct_rank(tv.values, x) for x in tv.values]
    df["migration_i"] = np.nanmean(np.vstack([df["theta_v1_pct"], tv_pct]), axis=0)

    # slopes
    for c in axis_cols:
        df[f"slope_{c}"] = rolling_slope(df[c], protocol.slope_window)

    # save artifacts
    outdir = ensure_dir(outdir)
    df.to_csv(outdir / "metrics_timeseries.csv", index=False)

    latest = df.iloc[-1]
    # leaders/challengers from latest axis shares
    axis_now = {c.replace("axis_", ""): float(latest[c]) for c in axis_cols}
    leaders = sorted(axis_now.items(), key=lambda kv: kv[1], reverse=True)[:6]
    pd.DataFrame(leaders, columns=["group","share"]).to_csv(outdir/"leaders.csv", index=False)
    slopes = {c.replace("slope_axis_", "").replace("axis_", ""): float(latest[c]) for c in df.columns if c.startswith("slope_axis_")}
    migr = sorted(slopes.items(), key=lambda kv: kv[1], reverse=True)
    pd.DataFrame(migr, columns=["group","slope"]).to_csv(outdir/"migration.csv", index=False)

    # summary
    top3 = [g for g, _ in leaders[:3]]
    chall = [g for g, _ in leaders[3:6]]
    top_constraints = sorted(
        [(c.replace("constraint_",""), float(latest[c])) for c in df.columns if c.startswith("constraint_")],
        key=lambda kv: kv[1], reverse=True
    )
    summary = {
        "observer_id": protocol.observer_id,
        "protocol": protocol.__dict__,
        "as_of_date": str(pd.Timestamp(latest["date"]).date()),
        "quality": {
            "eff_n": int(latest["eff_n"]),
            "missing_rate": float(latest["missing_rate"]),
            "min_eig": float(latest["min_eig"]),
            "cond_number": None if np.isnan(latest["cond_number"]) else float(latest["cond_number"]),
            "bootstrap_stability": None if np.isnan(latest["bootstrap_stability"]) else float(latest["bootstrap_stability"])
        },
        "metrics": {
            "s1": float(latest["s1"]),
            "erank": float(latest["erank"]),
            "pr": float(latest["pr"]),
            "gap": float(latest["gap"]),
            "theta_v1": None if np.isnan(latest["theta_v1"]) else float(latest["theta_v1"]),
            "d_fr": None if np.isnan(latest["d_fr"]) else float(latest["d_fr"]),
            "v_fr": None if np.isnan(latest["v_fr"]) else float(latest["v_fr"]),
            "single_axis_score": float(latest["single_axis_score"]),
            "hazard_i": float(latest["hazard_i"]),
            "migration_i": float(latest["migration_i"])
        },
        "leaders": top3,
        "challengers": chall,
        "constraint_top": [c for c,_ in top_constraints[:3]],
        "axis_shares": axis_now,
        "constraint_shares": {c:s for c,s in top_constraints}
    }
    dump_json(outdir/"summary.json", summary)

    # diagnostic plot
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    axs[0,0].plot(df["date"], df["hazard_i"], label="hazard_i", color="crimson")
    axs[0,0].plot(df["date"], df["migration_i"], label="migration_i", color="royalblue")
    axs[0,0].legend(frameon=False, fontsize=8)
    axs[0,0].set_title("Observer hazard/migration")
    axs[0,1].plot(df["date"], df["s1"], label="s1")
    axs[0,1].plot(df["date"], df["erank"]/df["erank"].max(), label="erank_norm")
    axs[0,1].legend(frameon=False, fontsize=8)
    axs[0,1].set_title("Spectrum")
    ser = pd.Series(axis_now).sort_values(ascending=False).head(8)
    axs[1,0].barh(list(ser.index)[::-1], list(ser.values)[::-1], color="teal")
    axs[1,0].set_title("Latest axis shares")
    ser2 = pd.Series({k:v for k,v in slopes.items() if not np.isnan(v)}).sort_values(ascending=False).head(8)
    if not ser2.empty:
        axs[1,1].barh(list(ser2.index)[::-1], list(ser2.values)[::-1], color="orange")
    axs[1,1].set_title("Latest migration slopes")
    plt.tight_layout()
    fig.savefig(outdir/"diagnostic.png", dpi=140)
    plt.close(fig)

    return summary, df
