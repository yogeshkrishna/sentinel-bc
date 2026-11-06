"""Local-only launch. No training, public tunnel or paid services."""
import argparse
from pathlib import Path
from sentinel_bc.ui import build_ui
from sentinel_bc.config import ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_ui(ROOT / "artifacts" / args.mode).queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False,
        allowed_paths=[str(ROOT / "results" / args.mode), str(ROOT / "runtime")], show_error=True)
