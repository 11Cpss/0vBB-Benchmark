#!/usr/bin/env python3
"""Render active numerical tables and the NEXT capacity figure from final inputs."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
from tables import render_all
ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "figures/data")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data_dir.iterdir() if p.is_file()}
    names = render_all(args.data_dir, args.output_dir / "tables")
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/plot_next_capacity.py"),
                    "--data-dir", str(args.data_dir), "--output-dir", str(args.output_dir / "figures")], check=True)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data_dir.iterdir() if p.is_file()}
    assert before == after, "Rendering modified final input data"
    report = {"tables": names, "figures": ["next_capacity_scores.pdf"],
              "operation": "render final display inputs; no metric recomputation", "inputs_unchanged": True}
    (args.output_dir / "rebuild_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
