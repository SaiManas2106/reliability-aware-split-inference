import os, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:5000")

def load_splits():
    with open(PROJECT_ROOT / "configs" / "splits_resnet18.json", "r", encoding="utf-8") as f:
        return json.load(f)
