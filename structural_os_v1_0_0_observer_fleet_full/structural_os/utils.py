from __future__ import annotations
import json, hashlib
from pathlib import Path
from eth_utils import keccak

def canonical_json_bytes(obj) -> bytes:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return s.encode("utf-8")

def sha256_hex_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def keccak_hex_bytes(b: bytes) -> str:
    return "0x" + keccak(b).hex()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def dump_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def pct_rank(arr, x):
    import numpy as np
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    return float((a <= x).mean())
