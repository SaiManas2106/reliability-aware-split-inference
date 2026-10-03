import argparse, csv
import requests
import torch

from shared.data_covertype import get_covertype_test_loader
from shared.split_mlp_covertype import MLPCovertypeSplitter
from shared.serialization import tensor_to_bytes, bytes_to_tensor
from edge.common_mlp_covertype import DEFAULT_SERVER_URL, DEFAULT_CHECKPOINT, load_splits

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--out", type=str, default="results/correctness_mlp_covertype.csv")
    ap.add_argument("--server", type=str, default=DEFAULT_SERVER_URL)
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    args = ap.parse_args()

    splitter = MLPCovertypeSplitter(checkpoint_path=args.checkpoint, device=torch.device("cpu"))
    split_ids = [s["split_id"] for s in load_splits()]
    rows = []

    for split_id in split_ids:
        worst = 0.0
        loader = get_covertype_test_loader(limit=args.n, batch_size=1)
        for x, _ in loader:
            y_full = splitter.full(x)
            act = splitter.front(x, split_id)
            resp = requests.post(
                f"{args.server}/infer_back_mlp_covertype",
                data={"split_id": split_id},
                files={"act": ("act.pt", tensor_to_bytes(act), "application/octet-stream")},
                timeout=120
            )
            resp.raise_for_status()
            y_split = bytes_to_tensor(resp.content)
            worst = max(worst, max_abs_diff(y_full, y_split))

        passed = worst <= args.tol
        rows.append({"split_id": split_id, "worst_max_abs_diff": worst, "tol": args.tol, "passed": passed})
        print(f"{split_id}: worst={worst:.6g} passed={passed}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split_id","worst_max_abs_diff","tol","passed"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
