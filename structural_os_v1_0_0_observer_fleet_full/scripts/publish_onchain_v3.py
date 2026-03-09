from __future__ import annotations
import argparse, json, os
from pathlib import Path
from eth_utils import keccak
from web3 import Web3

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def khex(txt: str) -> str:
    return "0x" + keccak(text=txt).hex()

ABI = [
    {
        "inputs": [
            {"internalType": "uint32", "name": "yyyymmdd", "type": "uint32"},
            {"internalType": "bytes32", "name": "statusHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "eventsHash", "type": "bytes32"},
            {"internalType": "bytes32", "name": "schemaHash", "type": "bytes32"},
            {"internalType": "string", "name": "uri", "type": "string"}
        ],
        "name": "publish",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    status_txt = read_text(Path(args.status))
    events_txt = read_text(Path(args.events))
    schema_txt = read_text(Path(args.schema))

    status_hash = khex(status_txt)
    events_hash = khex(events_txt)
    schema_hash = khex(schema_txt)

    status = json.loads(status_txt)
    d = status["as_of_date"]
    yyyymmdd = int(d.replace("-", ""))

    dry = str(os.getenv("CHAIN_DRY_RUN", "")).lower() in ("1","true","yes")
    rpc = os.getenv("CHAIN_RPC_URL", "")
    pk = os.getenv("CHAIN_PRIVATE_KEY", "")
    ca = os.getenv("CHAIN_CONTRACT_ADDRESS", "")
    uri = os.getenv("CHAIN_URI", "")

    out = {
        "as_of_date": d,
        "status_keccak256": status_hash,
        "events_keccak256": events_hash,
        "schema_keccak256": schema_hash,
        "chain_contract": ca or None,
        "uri": uri or None,
        "dry_run": dry
    }

    if rpc and pk and ca:
        w3 = Web3(Web3.HTTPProvider(rpc))
        acct = w3.eth.account.from_key(pk)
        contract = w3.eth.contract(address=Web3.to_checksum_address(ca), abi=ABI)
        if not dry:
            tx = contract.functions.publish(
                yyyymmdd, status_hash, events_hash, schema_hash, uri
            ).build_transaction({
                "from": acct.address,
                "nonce": w3.eth.get_transaction_count(acct.address),
                "chainId": int(os.getenv("CHAIN_CHAIN_ID") or w3.eth.chain_id),
            })
            if os.getenv("CHAIN_GAS_PRICE_GWEI"):
                tx["gasPrice"] = w3.to_wei(float(os.getenv("CHAIN_GAS_PRICE_GWEI")), "gwei")
            else:
                try:
                    tx["maxFeePerGas"] = w3.to_wei(float(os.getenv("CHAIN_MAX_FEE_GWEI","20")), "gwei")
                    tx["maxPriorityFeePerGas"] = w3.to_wei(float(os.getenv("CHAIN_PRIORITY_FEE_GWEI","1")), "gwei")
                except Exception:
                    pass
            signed = acct.sign_transaction(tx)
            txh = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(txh)
            out["tx_hash"] = receipt["transactionHash"].hex()
            out["block_number"] = int(receipt["blockNumber"])
        else:
            out["tx_hash"] = None

    Path(outdir/"onchain.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md = ["# On-chain", ""]
    md.append("```json")
    md.append(json.dumps(out, indent=2))
    md.append("```")
    Path(outdir/"onchain.md").write_text("\n".join(md) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
