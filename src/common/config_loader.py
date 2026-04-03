from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    full_path = ROOT_DIR / config_path
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
