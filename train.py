"""Train explicitly; the application never calls this entry point."""
import argparse
import logging
from pathlib import Path
from sentinel_bc.config import ROOT
from sentinel_bc.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    train(args.mode, args.output_root)
