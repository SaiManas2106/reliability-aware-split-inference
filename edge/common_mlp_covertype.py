import os, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:5000")
DEFAULT_CHECKPOINT = os.environ.get("MLP_CKPT", str(PROJECT_ROOT / "artifacts" / "mlp_covertype.pt"))

def load_splits():
    with open(PROJECT_ROOT / "configs" / "splits_mlp_covertype.json", "r", encoding="utf-8") as f:
        return json.load(f)
