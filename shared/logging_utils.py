import csv, os, time
from typing import Dict, Any

def now_ms() -> float:
    return time.time() * 1000.0

def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_csv_header_if_needed(path: str, fieldnames):
    ensure_parent(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

def append_csv_row(path: str, row: Dict[str, Any], fieldnames):
    write_csv_header_if_needed(path, fieldnames)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)
