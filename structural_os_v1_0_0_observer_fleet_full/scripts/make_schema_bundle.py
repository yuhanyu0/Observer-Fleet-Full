from __future__ import annotations
import argparse, json
from pathlib import Path
from eth_utils import keccak
from structural_os.utils import canonical_json_bytes, sha256_hex_bytes, dump_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemas", required=True)
    ap.add_argument("--schema_version", required=True)
    args = ap.parse_args()

    sp = Path(args.schemas)
    files = ["action.schema.json","status.schema.json","events.schema.json"]
    bundle = {
        "schema_version": args.schema_version,
        "files": {}
    }
    for fn in files:
        obj = json.loads((sp/fn).read_text(encoding="utf-8"))
        b = canonical_json_bytes(obj)
        bundle["files"][fn] = {
            "sha256": sha256_hex_bytes(b),
            "keccak256": "0x" + keccak(b).hex()
        }
    dump_json(sp/"bundle.json", bundle)
    md = ["# Schema Bundle","", "```json", json.dumps(bundle, indent=2), "```", ""]
    (sp/"bundle.md").write_text("\n".join(md), encoding="utf-8")
if __name__ == "__main__":
    main()
