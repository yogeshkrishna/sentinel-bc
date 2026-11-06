"""Inspect the cached locked evaluation; optionally regenerate its figures."""
import argparse
import json
from sentinel_bc.config import ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "results" / args.mode
    if not (directory / "metrics.json").exists():
        parser.error(f"No saved {args.mode} evaluation. Run python train.py --mode {args.mode}")
    if args.figures:
        from sentinel_bc.reporting import draw_figures
        draw_figures(directory)
    print(json.dumps(json.loads((directory / "metrics.json").read_text())["test"], indent=2))
