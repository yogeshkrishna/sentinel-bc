"""Repository paths and the single experiment configuration."""
from pathlib import Path
import json
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

