# Structural OS v1.0.0 — Observer Fleet Edition

A deployable GitHub Actions + GitHub Pages observatory that:
- runs an **observer fleet** (12 protocols by default)
- aggregates them into an ensemble state
- emits `status.json`, `events/events.json`, and `action` guardrails
- archives immutable daily snapshots
- publishes a static website (MkDocs)
- optionally anchors hashes on-chain

## Quick start

1. Create a new GitHub repo and upload all files in this package.
2. In **Settings → Pages**, choose **GitHub Actions**.
3. In **Settings → Actions → General**, set **Workflow permissions = Read and write**.
4. (Optional) Add chain secrets if you want on-chain anchoring:
   - `CHAIN_RPC_URL`
   - `CHAIN_PRIVATE_KEY`
   - `CHAIN_CONTRACT_ADDRESS`
   - optional: `CHAIN_CHAIN_ID`, `CHAIN_URI`, `CHAIN_DRY_RUN`
5. Run the workflow **daily-structural-os** once manually.

## What you get

- Dashboard at GitHub Pages
- Daily `briefs/`
- `status.json`
- `events/events.json`
- `archive/YYYY-MM-DD/`
- `verify` page for archive + schema validation
- `docs/observers/registry.json` to define the observer fleet

## Architecture

- `structural_os/universe.py`: universes and groups
- `structural_os/observer.py`: single-observer run
- `structural_os/aggregate.py`: ensemble aggregation / visibility / actions
- `structural_os/sitegen.py`: dashboard, brief, archive, events
- `structural_os/cli.py`: one-command orchestration
- `docs/observers/registry.json`: observer fleet definition
- `docs/schemas/*.schema.json`: public contract
