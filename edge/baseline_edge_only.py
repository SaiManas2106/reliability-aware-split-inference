import argparse
import psutil
import torch

from shared.data import get_cifar10_loader
from shared.split_resnet import ResNet18Splitter
from shared.logging_utils import now_ms, append_csv_row

LOG_FIELDS = ["request_id","strategy","split_id","t_start_ms","t_end_ms","latency_ms","bytes_up","bytes_down","cpu_percent"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--log", type=str, default="logs/edge_only_run1.csv")
    ap.add_argument("--data_root", type=str, default="./data")
    args = ap.parse_args()

    splitter = ResNet18Splitter(device=torch.device("cpu"))
    loader = get_cifar10_loader(root=args.data_root, train=False, batch_size=1, limit=args.n)

    for i, (x, _) in enumerate(loader):
        t0 = now_ms()
        _ = splitter.full(x)
        t1 = now_ms()
        cpu = psutil.cpu_percent(interval=None)

        append_csv_row(args.log, {
            "request_id": i,
            "strategy": "edge_only",
            "split_id": "",
            "t_start_ms": f"{t0:.3f}",
            "t_end_ms": f"{t1:.3f}",
            "latency_ms": f"{(t1-t0):.3f}",
            "bytes_up": 0,
            "bytes_down": 0,
            "cpu_percent": cpu
        }, LOG_FIELDS)

    print(f"Done. Wrote {args.log}")

if __name__ == "__main__":
    main()
