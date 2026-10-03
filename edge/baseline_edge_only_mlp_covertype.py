import argparse
import psutil
import torch

from shared.data_covertype import get_covertype_test_loader
from shared.split_mlp_covertype import MLPCovertypeSplitter
from shared.logging_utils import now_ms, append_csv_row
from edge.common_mlp_covertype import DEFAULT_CHECKPOINT

LOG_FIELDS = ["request_id","strategy","split_id","t_start_ms","t_end_ms","latency_ms","bytes_up","bytes_down","cpu_percent"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--log", type=str, default="logs/mlp_covertype_edge_only.csv")
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    args = ap.parse_args()

    splitter = MLPCovertypeSplitter(checkpoint_path=args.checkpoint, device=torch.device("cpu"))
    loader = get_covertype_test_loader(limit=args.n, batch_size=1)

    for i, (x, _) in enumerate(loader):
        t0 = now_ms()
        _ = splitter.full(x)
        t1 = now_ms()
        append_csv_row(args.log, {
            "request_id": i,
            "strategy": "edge_only_mlp_covertype",
            "split_id": "",
            "t_start_ms": f"{t0:.3f}",
            "t_end_ms": f"{t1:.3f}",
            "latency_ms": f"{(t1-t0):.3f}",
            "bytes_up": 0,
            "bytes_down": 0,
            "cpu_percent": psutil.cpu_percent(interval=None)
        }, LOG_FIELDS)
    print(f"Done. Wrote {args.log}")

if __name__ == "__main__":
    main()
