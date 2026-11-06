"""Train explicitly; the application never calls this entry point."""
import argparse
import logging
from pathlib import Path
from sentinel_bc.config import ROOT
import time
from sentinel_bc.runs import RunManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runtime/training", help="Registry directory; a unique child is created for every run")
    parser.add_argument("--data", type=Path, help="Compatible labeled WDBC-schema CSV; omit for built-in data")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, help="Optional neural epoch budget (2–200); YAML defaults otherwise")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    manager = RunManager(args.output_root)
    run_id = manager.start(args.mode, args.data, args.seed, args.epochs)
    print(f"Run: {run_id}\nDirectory: {manager.root / run_id}", flush=True)
    previous = None
    try:
        while True:
            state = manager.status(run_id)
            event = (state["phase"], state.get("detail", ""))
            if event != previous:
                print(f"[{state['elapsed_seconds']:.0f}s] {' — '.join(event)}", flush=True)
                previous = event
            if state["status"] not in ("QUEUED", "RUNNING", "CANCELLING"):
                print(state.get("error", "Training complete. Activate this run in the Gradio Training Studio."))
                raise SystemExit(0 if state["status"] == "COMPLETED" else 1)
            time.sleep(1)
    except KeyboardInterrupt:
        print(manager.cancel(run_id))
        raise SystemExit(130)
