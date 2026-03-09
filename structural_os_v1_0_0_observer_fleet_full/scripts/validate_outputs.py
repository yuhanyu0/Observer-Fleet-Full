from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from jsonschema import Draft202012Validator, RefResolver

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--schemas", required=True)
    args = ap.parse_args()

    schemas_dir = Path(args.schemas)
    action = load(schemas_dir/"action.schema.json")
    status_schema = load(schemas_dir/"status.schema.json")
    events_schema = load(schemas_dir/"events.schema.json")

    store = {
        action.get("$id","action.schema.json"): action,
        status_schema.get("$id","status.schema.json"): status_schema,
        events_schema.get("$id","events.schema.json"): events_schema,
        "action.schema.json": action,
        "status.schema.json": status_schema,
        "events.schema.json": events_schema,
    }

    resolver = RefResolver.from_schema(status_schema, store=store)
    v_status = Draft202012Validator(status_schema, resolver=resolver)
    errs = sorted(v_status.iter_errors(load(args.status)), key=lambda e: str(e.path))
    if errs:
        for e in errs[:20]:
            print("STATUS:", list(e.path), e.message)
        sys.exit(1)

    resolver2 = RefResolver.from_schema(events_schema, store=store)
    v_events = Draft202012Validator(events_schema, resolver=resolver2)
    errs2 = sorted(v_events.iter_errors(load(args.events)), key=lambda e: str(e.path))
    if errs2:
        for e in errs2[:20]:
            print("EVENTS:", list(e.path), e.message)
        sys.exit(1)

    print("OK")
if __name__ == "__main__":
    main()
