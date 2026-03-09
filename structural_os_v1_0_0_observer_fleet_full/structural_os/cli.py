from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime, timezone
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from .universe import groups_for_universe, flatten_groups
from .observer import ObserverProtocol, run_observer
from .aggregate import aggregate_ensemble, pick_events
from .sitegen import plot_phase, plot_visibility, plot_observer_heatmap, plot_constraints, generate_home, generate_brief, generate_observers_page, generate_events, make_archive
from .utils import ensure_dir, dump_json

def load_cfg(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def load_registry(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def build_universe_union(registry):
    tickers = []
    seen = set()
    for obs in registry["observers"]:
        groups = groups_for_universe(obs["universe_id"])
        for t in flatten_groups(groups):
            if t not in seen:
                tickers.append(t)
                seen.add(t)
    return tickers

def download_prices(tickers, start, end, freq):
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=freq,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    px = pd.DataFrame(index=data.index)
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = data[(t, "Close")] if (t, "Close") in data.columns else data[(t, "Adj Close")]
                px[t] = s
            except Exception:
                pass
    else:
        if "Close" in data.columns:
            px[tickers[0]] = data["Close"]
        elif "Adj Close" in data.columns:
            px[tickers[0]] = data["Adj Close"]
    px = px.sort_index()
    return px

def build_observation_fingerprint(cfg, registry):
    import hashlib, json
    obs_proto = {
        "registry_path": "docs/observers/registry.json",
        "observers": registry["observers"],
        "pipeline": cfg["pipeline"],
    }
    raw = json.dumps(obs_proto, sort_keys=True, separators=(",", ":")).encode()
    return {
        "protocol_hash": hashlib.sha256(raw).hexdigest(),
        "window_set": sorted({o["window"] for o in registry["observers"]}),
        "estimators": sorted({o["estimator"] for o in registry["observers"]}),
        "universes": sorted({o["universe_id"] for o in registry["observers"]}),
        "grammar_ids": sorted({o["grammar_id"] for o in registry["observers"]}),
        "producer": "observer_fleet_v1"
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs='?')
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    docs = Path("docs")
    ensure_dir(docs/"assets")
    ensure_dir(docs/"events")
    ensure_dir(docs/"briefs")
    ensure_dir(docs/"observers"/"runs")

    registry = load_registry(docs/"observers"/"registry.json")
    all_tickers = build_universe_union(registry)
    prices = download_prices(all_tickers, cfg["pipeline"]["start"], cfg["pipeline"]["end"], cfg["pipeline"]["freq"])

    summaries = []
    latest_date = None
    for ob in registry["observers"]:
        protocol = ObserverProtocol(
            observer_id=ob["observer_id"],
            universe_id=ob["universe_id"],
            grammar_id=ob["grammar_id"],
            window=int(ob["window"]),
            estimator=ob["estimator"],
            baseline=registry["default"]["baseline"],
            use_log_returns=registry["default"]["use_log_returns"],
            min_coverage=registry["default"]["min_coverage"],
            ridge=registry["default"]["ridge"],
            K=registry["default"]["K"],
            slope_window=cfg["pipeline"]["slope_window"],
        )
        outdir = docs/"observers"/"runs"/datetime.now(timezone.utc).date().isoformat()/protocol.observer_id
        summary, df = run_observer(protocol, prices, outdir, lookback_days=cfg["pipeline"]["lookback_days"])
        summaries.append(summary)
        latest_date = summary["as_of_date"]

    observation_fingerprint = build_observation_fingerprint(cfg, registry)
    status = aggregate_ensemble(latest_date, summaries, cfg["thresholds"], observation_fingerprint)
    status["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    status["producer_version"] = "1.0.0"
    status["window"] = {"lookback_days": cfg["pipeline"]["lookback_days"]}

    # build ensemble timeseries from latest rows across observers by reading each metrics_timeseries
    ens_rows = []
    # collect all observer dfs
    all_dfs = {}
    date_run = latest_date
    for s in summaries:
        p = docs/"observers"/"runs"/date_run/s["observer_id"]/ "metrics_timeseries.csv"
        all_dfs[s["observer_id"]] = pd.read_csv(p, parse_dates=["date"])

    # common dates union over recent lookback
    dates = sorted(set().union(*[set(df["date"].dt.date.astype(str)) for df in all_dfs.values()]))
    dates = dates[-cfg["pipeline"]["lookback_days"]:]
    for d in dates:
        dsums = []
        for s in summaries:
            df = all_dfs[s["observer_id"]]
            hit = df[df["date"].dt.date.astype(str) == d]
            if hit.empty:
                continue
            row = hit.iloc[-1]
            dsums.append({
                "observer_id": s["observer_id"],
                "protocol": s["protocol"],
                "quality": {
                    "eff_n": int(row["eff_n"]),
                    "missing_rate": float(row["missing_rate"]),
                    "min_eig": float(row["min_eig"]),
                    "cond_number": None if np.isnan(row["cond_number"]) else float(row["cond_number"]),
                    "bootstrap_stability": None if np.isnan(row["bootstrap_stability"]) else float(row["bootstrap_stability"]),
                },
                "metrics": {
                    "hazard_i": float(row["hazard_i"]),
                    "migration_i": float(row["migration_i"]),
                },
                "leaders": s["leaders"],
                "challengers": s["challengers"],
                "axis_shares": s["axis_shares"],
                "constraint_shares": s["constraint_shares"],
            })
        if len(dsums) < max(4, len(summaries)//2):
            continue
        st = aggregate_ensemble(d, dsums, cfg["thresholds"], observation_fingerprint)
        ens_rows.append({
            "date": d,
            "hazard": st["scores"]["hazard"],
            "migration": st["scores"]["migration"],
            "opportunity": st["scores"]["opportunity"],
            "visibility": st["ensemble"]["visibility"],
            "regime": st["state"]["regime"],
            "cp_score": 0.0
        })
    ens_df = pd.DataFrame(ens_rows)
    if not ens_df.empty:
        ens_df["cp_score"] = ens_df[["hazard","migration","visibility"]].diff().abs().sum(axis=1).fillna(0)

    # plots
    plot_phase(ens_df, docs/"assets"/"phase.png")
    plot_visibility(ens_df, docs/"assets"/"visibility.png")
    plot_observer_heatmap(summaries, docs/"assets"/"agreement.png")
    plot_constraints(summaries, docs/"assets"/"constraints.png")

    # events
    event_idx = pick_events(ens_df, n_events=cfg["pipeline"]["event_count"], min_sep=cfg["pipeline"]["event_min_sep_days"])
    events = generate_events(docs, ens_df, event_idx, status)

    # outputs
    dump_json(docs/"status.json", status)
    dump_json(docs/"events"/"events.json", events)
    generate_observers_page(docs, summaries, registry)
    brief_name = generate_brief(docs, status, ens_df)
    generate_home(docs, status, brief_name)
    if cfg["pipeline"].get("archive", True):
        make_archive(docs, status)

if __name__ == "__main__":
    main()
