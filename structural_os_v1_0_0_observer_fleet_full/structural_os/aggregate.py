from __future__ import annotations
import math, json
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import dump_json, ensure_dir, pct_rank, canonical_json_bytes, sha256_hex_bytes, keccak_hex_bytes

def cosine_sim(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return np.nan
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / max(1, len(sa | sb)))

def robust_quality(qblock):
    # qblock list of dicts
    vals = []
    for q in qblock:
        eff = min(1.0, q["eff_n"]/max(40, q["eff_n"]))
        miss = 1.0 - min(1.0, q["missing_rate"])
        mineig = 1.0 if q["min_eig"] is None else float(max(0.0, min(1.0, q["min_eig"]*1000)))
        cond = q["cond_number"] if q["cond_number"] is not None else 1e6
        condq = float(max(0.0, min(1.0, 1.0 / (1.0 + math.log10(max(cond, 1.0))))))
        bst = q.get("bootstrap_stability")
        bstq = 0.5 if bst is None or math.isnan(bst) else float(max(0.0, min(1.0, bst)))
        vals.append(np.mean([eff, miss, mineig, condq, bstq]))
    if not vals:
        return float("nan")
    return float(np.nanmedian(vals))

def compute_agreement(summaries):
    # structural agreement from axis share vectors
    ids = [s["observer_id"] for s in summaries]
    axis_keys = sorted({k for s in summaries for k in s["axis_shares"].keys()})
    V = []
    for s in summaries:
        V.append([s["axis_shares"].get(k, 0.0) for k in axis_keys])
    sims = []
    for i in range(len(V)):
        for j in range(i+1, len(V)):
            x = cosine_sim(V[i], V[j])
            if not np.isnan(x):
                sims.append((ids[i], ids[j], x))
    if sims:
        A = float(np.median([x for _,_,x in sims]))
    else:
        A = float("nan")
    # semantic consistency via leaders overlap
    sem = []
    for i in range(len(summaries)):
        for j in range(i+1, len(summaries)):
            sem.append(jaccard(summaries[i]["leaders"], summaries[j]["leaders"]))
    S = float(np.median(sem)) if sem else float("nan")
    return A, S, sims, axis_keys, V

def compute_gain(summaries):
    vals = []
    for s in summaries:
        m = s["metrics"]
        hazard_i = float(m["hazard_i"])
        mig_i = float(m["migration_i"])
        vals.append(np.nanmean([hazard_i, mig_i]))
    return float(np.nanmean(vals)) if vals else float("nan")

def classify_state(visibility, hazard, migration, thresholds):
    if visibility < thresholds["fog_visibility_low"]:
        return "fog"
    if hazard >= thresholds["hazard_brake"]:
        return "brake"
    if hazard >= thresholds["hazard_tightening"] and migration < thresholds["migration_high"]:
        return "tightening"
    if migration >= thresholds["migration_high"]:
        return "theme_migration"
    return "stable"

def make_action(regime, hazard, migration, opportunity, visibility, watch_constraints):
    # dual-channel + brake devices
    if regime == "fog":
        risk_mode, risk_budget, cooldown, review = "tighten", 0.15, 48, True
        explore_mode, explore_budget = "off", 0.0
    elif regime == "brake":
        risk_mode, risk_budget, cooldown, review = "brake", 0.05, 72, True
        explore_mode, explore_budget = "off", 0.0
    elif regime == "tightening":
        risk_mode, risk_budget, cooldown, review = "tighten", 0.30, 24, True
        explore_mode, explore_budget = ("probe", max(0.01, 0.1*opportunity)) if opportunity > 0.15 else ("off", 0.0)
    elif regime == "theme_migration":
        risk_mode, risk_budget, cooldown, review = "normal", max(0.35, 0.6*(1-hazard)), 12, False
        explore_mode, explore_budget = ("explore", min(0.15, 0.2*opportunity + 0.02))
    else:
        risk_mode, risk_budget, cooldown, review = "normal", max(0.45, 0.7*(1-hazard)), 12, False
        explore_mode, explore_budget = ("probe", min(0.05, 0.1*opportunity)) if opportunity > 0.10 else ("off", 0.0)

    brakes = {
        "rate": {
            "max_change_rate": float(max(0.0, min(1.0, risk_budget))),
            "cooldown_hours": int(cooldown),
            "rebalance_allowed": bool(regime not in ["fog", "brake"])
        },
        "cost": {
            "surcharge_bps": int(round((1.0-risk_budget)*100)),
            "max_gross": float(max(0.10, min(1.0, risk_budget + 0.2)))
        },
        "audit": {
            "review_required": bool(review),
            "replay_required": bool(regime in ["fog", "brake"]),
            "sample_rate": float(0.5 if regime in ["fog","brake"] else 0.1)
        }
    }

    action = {
        "action_code": f"{risk_mode[:3].upper()}+{explore_mode[:3].upper()}",
        "risk": {
            "mode": risk_mode,
            "budget": float(risk_budget),
            "cooldown_hours": int(cooldown),
            "human_review_required": bool(review)
        },
        "explore": {
            "mode": explore_mode,
            "budget": float(explore_budget)
        },
        "brakes": brakes,
        "watch_constraints": watch_constraints[:3],
        "rationale": [
            f"visibility={visibility:.2f}",
            f"hazard={hazard:.2f}",
            f"migration={migration:.2f}",
            f"opportunity={opportunity:.2f}",
            "flight-controller rule applied"
        ]
    }
    return action

def aggregate_ensemble(date, summaries, thresholds, observation_fingerprint):
    Q = robust_quality([s["quality"] for s in summaries])
    A, S, sims, axis_keys, vectors = compute_agreement(summaries)
    visibility = float(np.nanmin([Q, A, S]))
    gain = compute_gain(summaries)

    hazard = float(np.nanmean([s["metrics"]["hazard_i"] for s in summaries]))
    migration = float(np.nanmean([s["metrics"]["migration_i"] for s in summaries]))

    # emergent constraint: mean of shares
    constraint_keys = sorted({k for s in summaries for k in s["constraint_shares"].keys()})
    cmat = np.array([[s["constraint_shares"].get(k, 0.0) for k in constraint_keys] for s in summaries], dtype=float)
    cmean = np.nanmean(cmat, axis=0) if cmat.size else np.array([])
    top_idx = np.argsort(cmean)[::-1] if cmean.size else np.array([], dtype=int)
    dominant = constraint_keys[int(top_idx[0])] if len(top_idx) else None
    emergent = constraint_keys[int(top_idx[1])] if len(top_idx) > 1 else dominant

    opportunity = float(max(0.0, migration * visibility * (1.0 - hazard)))

    regime = classify_state(visibility, hazard, migration, thresholds)
    action = make_action(regime, hazard, migration, opportunity, visibility, [dominant, emergent] if dominant else [])

    observers = []
    for s in summaries:
        observers.append({
            "observer_id": s["observer_id"],
            "protocol": s["protocol"],
            "quality": s["quality"],
            "metrics": s["metrics"],
            "leaders": s["leaders"],
            "challengers": s["challengers"]
        })

    ensemble = {
        "Q": float(Q),
        "A": float(A),
        "S": float(S),
        "visibility": float(visibility),
        "gain": float(gain),
        "friction": float(max(0.0, 1.0 - gain)),
        "agreement_pairs": [{"left": a, "right": b, "similarity": x} for a,b,x in sims[:50]],
        "observer_count": len(summaries),
    }

    status = {
        "schema_version": "1.0.0",
        "producer_version": "1.0.0",
        "generated_at_utc": None,  # filled by caller
        "as_of_date": date,
        "window": {"lookback_days": None},
        "state": {"regime": regime},
        "scores": {"hazard": hazard, "migration": migration, "opportunity": opportunity},
        "constraints": {
            "dominant_constraint": dominant,
            "emergent_constraint": emergent,
            "constraint_novelty": float(np.nanstd(cmean)) if cmean.size else None
        },
        "observation_fingerprint": observation_fingerprint,
        "observers": observers,
        "ensemble": ensemble,
        "action": action
    }
    return status

def pick_events(ens_df: pd.DataFrame, n_events: int = 3, min_sep: int = 20):
    if ens_df.empty:
        return []
    score = ens_df["cp_score"].fillna(0).values
    idx = np.argsort(score)[::-1]
    chosen = []
    for j in idx:
        if len(chosen) >= n_events:
            break
        if all(abs(j - c) >= min_sep for c in chosen):
            chosen.append(int(j))
    chosen.sort()
    return chosen
