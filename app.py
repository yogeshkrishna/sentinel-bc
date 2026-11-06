"""Local Training Studio; fitting starts only through an explicit user action."""
import argparse
from pathlib import Path
from sentinel_bc.ui import build_ui
from sentinel_bc.config import ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--runs-dir", type=Path, help="Optional isolated run registry")
    parser.add_argument("--database", type=Path, help="Optional review database for demonstrations/tests")
    parser.add_argument("--no-benchmark", action="store_true", help="Start without the historical bundle")
    args = parser.parse_args()
    build_ui(ROOT / "artifacts" / ("not-installed" if args.no_benchmark else args.mode), args.database, args.runs_dir).queue(default_concurrency_limit=2).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False,
        allowed_paths=[str(ROOT / "runtime/studio-assets"), str(ROOT / "runtime/reports")], show_error=True)
