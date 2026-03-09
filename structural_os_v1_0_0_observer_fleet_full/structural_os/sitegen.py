from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import dump_json, ensure_dir, canonical_json_bytes, sha256_hex_bytes, keccak_hex_bytes

def plot_phase(ens_df: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(ens_df["migration"], ens_df["hazard"], c=np.linspace(0,1,len(ens_df)), cmap="viridis", s=18)
    if len(ens_df):
        ax.scatter(ens_df["migration"].iloc[-1], ens_df["hazard"].iloc[-1], color="red", s=60, label="latest")
        ax.legend(frameon=False)
    ax.set_xlabel("migration")
    ax.set_ylabel("hazard")
    ax.set_title("Phase portrait")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)

def plot_visibility(ens_df: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(7,4))
    for col in ["visibility","hazard","migration"]:
        ax.plot(pd.to_datetime(ens_df["date"]), ens_df[col], label=col)
    ax.legend(frameon=False)
    ax.set_title("Visibility / hazard / migration")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)

def plot_observer_heatmap(summaries, outpath: Path):
    # axis-share cosine similarity matrix
    ids = [s["observer_id"] for s in summaries]
    axis_keys = sorted({k for s in summaries for k in s["axis_shares"].keys()})
    V = np.array([[s["axis_shares"].get(k, 0.0) for k in axis_keys] for s in summaries], dtype=float)
    n = len(ids)
    M = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            a, b = V[i], V[j]
            den = np.linalg.norm(a)*np.linalg.norm(b)
            sim = float(np.dot(a,b)/den) if den > 0 else np.nan
            M[i,j] = M[j,i] = sim
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ids, rotation=90, fontsize=6)
    ax.set_yticklabels(ids, fontsize=6)
    ax.set_title("Observer agreement (axis-share cosine)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)

def plot_constraints(summaries, outpath: Path):
    keys = sorted({k for s in summaries for k in s["constraint_shares"].keys()})
    vals = np.array([[s["constraint_shares"].get(k,0.0) for k in keys] for s in summaries], dtype=float)
    mean = vals.mean(axis=0) if vals.size else np.array([])
    ser = pd.Series(mean, index=keys).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(ser.index, ser.values, color="steelblue")
    ax.set_title("Mean constraint shares across observers")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)

def generate_home(docs_dir: Path, status: dict, latest_brief: str):
    text = []
    text.append("# Structural OS")
    text.append("")
    text.append(f"**As of:** `{status['as_of_date']}`  ")
    text.append(f"**Regime:** `{status['state']['regime']}`  ")
    text.append(f"**Visibility:** `{status['ensemble']['visibility']:.2f}`  ")
    text.append(f"**Hazard:** `{status['scores']['hazard']:.2f}`  ")
    text.append(f"**Migration:** `{status['scores']['migration']:.2f}`  ")
    text.append(f"**Opportunity:** `{status['scores']['opportunity']:.2f}`  ")
    text.append("")
    text.append(f"- Dominant constraint: **{status['constraints']['dominant_constraint']}**")
    text.append(f"- Emergent constraint: **{status['constraints']['emergent_constraint']}**")
    text.append(f"- Latest brief: [open](briefs/{latest_brief})")
    text.append("")
    text.append("## Dashboard")
    text.append("")
    for p, title in [
        ("assets/phase.png","Phase portrait"),
        ("assets/visibility.png","Visibility / hazard / migration"),
        ("assets/agreement.png","Observer agreement"),
        ("assets/constraints.png","Constraint shares")
    ]:
        text.append(f"### {title}")
        text.append(f"![]({p})")
        text.append("")
    (docs_dir / "index.md").write_text("\n".join(text) + "\n", encoding="utf-8")

def generate_brief(docs_dir: Path, status: dict, ens_df: pd.DataFrame):
    d = status["as_of_date"]
    leaders = {}
    for obs in status["observers"]:
        for g in obs["leaders"]:
            leaders[g] = leaders.get(g, 0) + 1
    leaders_sorted = sorted(leaders.items(), key=lambda kv:(-kv[1], kv[0]))
    text = []
    text.append(f"# Daily Brief — {d}")
    text.append("")
    text.append(f"Regime: **{status['state']['regime']}**")
    text.append("")
    text.append("## Risk line")
    text.append("")
    text.append(f"- Visibility: `{status['ensemble']['visibility']:.2f}`")
    text.append(f"- Hazard: `{status['scores']['hazard']:.2f}`")
    text.append(f"- Migration: `{status['scores']['migration']:.2f}`")
    text.append(f"- Opportunity: `{status['scores']['opportunity']:.2f}`")
    text.append(f"- Gain: `{status['ensemble']['gain']:.2f}`")
    text.append("")
    text.append("## Observer consensus leaders")
    text.append("")
    for g, c in leaders_sorted[:8]:
        text.append(f"- {g} ({c}/{status['ensemble']['observer_count']})")
    text.append("")
    text.append("## Action")
    text.append("")
    text.append(f"- Code: `{status['action']['action_code']}`")
    text.append(f"- Risk mode: `{status['action']['risk']['mode']}`, budget `{status['action']['risk']['budget']:.2f}`")
    text.append(f"- Explore mode: `{status['action']['explore']['mode']}`, budget `{status['action']['explore']['budget']:.2f}`")
    for line in status["action"]["rationale"]:
        text.append(f"- {line}")
    text.append("")
    path = docs_dir / "briefs" / f"{d}.md"
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    # index
    briefs = sorted((docs_dir/"briefs").glob("*.md"), reverse=True)
    idx = ["# Briefs", ""]
    for b in briefs:
        if b.name == "index.md":
            continue
        idx.append(f"- [{b.stem}]({b.name})")
    (docs_dir / "briefs" / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    return path.name

def generate_observers_page(docs_dir: Path, summaries: list, registry: dict):
    text = ["# Observers", ""]
    text.append(f"Observer count: **{len(summaries)}**  ")
    text.append("")
    text.append("## Registry")
    text.append("")
    text.append("```json")
    text.append(json.dumps(registry, indent=2))
    text.append("```")
    text.append("")
    text.append("## Latest summaries")
    for s in summaries:
        text.append(f"### {s['observer_id']}")
        text.append(f"- Leaders: {', '.join(s['leaders'])}")
        text.append(f"- Hazard_i: {s['metrics']['hazard_i']:.2f}  Migration_i: {s['metrics']['migration_i']:.2f}")
        text.append(f"- eff_n={s['quality']['eff_n']} missing={s['quality']['missing_rate']:.2f}")
        text.append("")
    (docs_dir/"observers"/"index.md").write_text("\n".join(text) + "\n", encoding="utf-8")

def generate_events(docs_dir: Path, ens_df: pd.DataFrame, event_idx: list, status: dict):
    events = []
    for idx in event_idx:
        d = ens_df.iloc[idx]["date"]
        dstr = str(pd.Timestamp(d).date())
        pre = ens_df.iloc[max(0, idx-5)]["regime"]
        post = ens_df.iloc[min(len(ens_df)-1, idx+5)]["regime"]
        story = [
            f"State transition: {pre} → {ens_df.iloc[idx]['regime']} → {post}.",
            f"Hazard={ens_df.iloc[idx]['hazard']:.2f}, migration={ens_df.iloc[idx]['migration']:.2f}, visibility={ens_df.iloc[idx]['visibility']:.2f}.",
            f"Dominant constraint around event: {status['constraints']['dominant_constraint']} / emergent: {status['constraints']['emergent_constraint']}."
        ]
        md = ["# Event Slice — " + dstr, ""] + story + ["", f"![](../assets/phase.png)", "", f"![](../assets/visibility.png)", "", f"![](../assets/agreement.png)", "", f"![](../assets/constraints.png)"]
        ep = docs_dir / "events" / f"{dstr}.md"
        ep.write_text("\n".join(md) + "\n", encoding="utf-8")
        events.append({
            "event_date": dstr,
            "scores": {
                "cp_score": float(ens_df.iloc[idx]["cp_score"]),
                "hazard_peak": float(ens_df.iloc[idx]["hazard"]),
                "migration_peak": float(ens_df.iloc[idx]["migration"]),
                "visibility": float(ens_df.iloc[idx]["visibility"]),
            },
            "evidence": {
                "state_transition": {"pre": pre, "event": ens_df.iloc[idx]["regime"], "post": post},
                "story": story
            },
            "artifacts": {
                "event_page_md": f"events/{dstr}.md",
                "phase_png": "assets/phase.png",
                "visibility_png": "assets/visibility.png",
                "agreement_png": "assets/agreement.png",
                "constraints_png": "assets/constraints.png"
            }
        })
    idx_md = ["# Events", ""]
    for e in sorted(events, key=lambda x: x["event_date"], reverse=True):
        idx_md.append(f"- [{e['event_date']}]({e['event_date']}.md) — cp_score={e['scores']['cp_score']:.2f}")
    (docs_dir/"events"/"index.md").write_text("\n".join(idx_md) + "\n", encoding="utf-8")
    return {"schema_version":"1.0.0","producer_version":"1.0.0","generated_at_utc": datetime.now(timezone.utc).isoformat(),"events": events}

def make_archive(docs_dir: Path, status: dict):
    import shutil
    from datetime import datetime, timezone
    as_of = status["as_of_date"]
    daydir = ensure_dir(docs_dir/"archive"/as_of)
    # copy core files
    for rel in ["status.json", "events/events.json", "schemas/bundle.json"]:
        src = docs_dir / rel
        dst = daydir / Path(rel).name
        if src.exists():
            shutil.copyfile(src, dst)
    # archive schemas
    schemadir = ensure_dir(daydir/"schemas")
    for fn in ["action.schema.json","status.schema.json","events.schema.json"]:
        src = docs_dir/"schemas"/fn
        if src.exists():
            shutil.copyfile(src, schemadir/fn)
    files = {}
    for p in [daydir/"status.json", daydir/"events.json", daydir/"schema_bundle.json",
              schemadir/"action.schema.json", schemadir/"status.schema.json", schemadir/"events.schema.json"]:
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            b = txt.encode("utf-8")
            files[str(p.relative_to(daydir))] = {
                "bytes": len(b),
                "sha256": sha256_hex_bytes(b),
                "keccak256": keccak_hex_bytes(b)
            }
    manifest = {
        "as_of_date": as_of,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files
    }
    dump_json(daydir/"manifest.json", manifest)

    root = docs_dir/"archive"
    dates = sorted([p.name for p in root.iterdir() if p.is_dir() and len(p.name)==10], reverse=True)
    md = ["# Archive","","Daily immutable snapshots.","","| Date | Status | Events | Manifest |","|---|---|---|---|"]
    for d in dates:
        md.append(f"| {d} | [status](./{d}/status.json) | [events](./{d}/events.json) | [manifest](./{d}/manifest.json) |")
    (root/"index.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    dump_json(root/"index.json", {"dates": dates})
