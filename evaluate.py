"""Inspect the cached locked evaluation; optionally regenerate its figures."""
import argparse
import json
import shutil
from pathlib import Path
from sentinel_bc.config import ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--run", help="Completed run ID from Train or train.py")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime/training")
    parser.add_argument("--figures", action="store_true", help="Copy cached figures to runtime/evaluation")
    args = parser.parse_args()
    directory = ROOT / "results" / args.mode
    if args.run or args.mode == "quick":
        from sentinel_bc.runs import RunManager
        manager = RunManager(args.runs_dir, restore_active=False)
        run = args.run or next((r["run_id"] for r in manager.completed() if r["mode"] == "quick"), None)
        if run:
            try:
                _, _, directory = manager.validate_run(run, load_engine=False)
            except (ValueError, OSError) as exc:
                parser.error(str(exc))
    if not (directory / "metrics.json").exists():
        parser.error(f"No saved {args.mode} evaluation. Run python train.py --mode {args.mode}")
    if args.figures:
        version = json.loads((directory / "metrics.json").read_text())["version"]
        target = ROOT / "runtime/evaluation" / version
        shutil.copytree(directory / "figures", target, dirs_exist_ok=True)
        print(f"Figures copied to {target}")
    print(json.dumps(json.loads((directory / "metrics.json").read_text())["test"], indent=2))
