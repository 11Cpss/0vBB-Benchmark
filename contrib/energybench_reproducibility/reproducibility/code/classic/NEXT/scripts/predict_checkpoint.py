#!/usr/bin/env python3
"""Export predictions from the paper's earlier file-split NEXT checkpoints."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from next_cnn.adapter import predict

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    columns = {}
    metadata = None
    for batch in predict(args.checkpoint, args.data, batch_size=args.batch_size,
                         device=args.device, split="test"):
        batch = dict(batch)
        batch_metadata = batch.pop("__metadata__", None)
        if metadata is None:
            metadata = batch_metadata
        elif metadata != batch_metadata:
            raise ValueError("prediction metadata changed between batches")
        if columns and set(batch) != set(columns):
            raise ValueError("prediction column schema changed between batches")
        for key, value in batch.items():
            columns.setdefault(key, []).append(np.asarray(value))
    if not columns:
        raise RuntimeError("inference returned no events")
    arrays = {k: np.concatenate(v) for k, v in columns.items()}
    if len(np.unique(arrays["event_id"])) != len(arrays["event_id"]):
        raise ValueError("duplicate event IDs")
    arrays["energy_keV"] = np.asarray(arrays["energy_condition"], dtype=np.float64) * 1000.0
    arrays["weight"] = arrays["sample_weight"]
    arrays["group"] = arrays["category"]
    arrays["__metadata_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps({"events": len(arrays["event_id"]), "output": str(args.output)}))

if __name__ == "__main__":
    main()
