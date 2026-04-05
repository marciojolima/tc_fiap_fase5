from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GLOBAL_CONFIG_PATH = "configs/pipeline_global_config.yaml"


def load_config(config_path: str = DEFAULT_GLOBAL_CONFIG_PATH) -> dict:
    full_path = ROOT_DIR / config_path
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
