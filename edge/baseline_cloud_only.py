import argparse
import psutil
import requests

from shared.data import get_cifar10_loader
from shared.serialization import tensor_to_bytes, bytes_to_tensor
from shared.logging_utils import now_ms, append_csv_row
from edge.common import DEFAULT_SERVER_URL

LOG_FIELDS = ["request_id","strategy","split_id","t_start_ms","t_end_ms","latency_ms","bytes_up","bytes_down","server_compute_ms","cpu_percent"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--log", type=str, default="logs/cloud_only_run1.csv")
    ap.add_argument("--server", type=str, default=DEFAULT_SERVER_URL)
    ap.add_argument("--data_root", type=str, default="./data")
    args = ap.parse_args()

    loader = get_cifar10_loader(root=args.data_root, train=False, batch_size=1, limit=args.n)

    for i, (x, _) in enumerate(loader):
        x_bytes = tensor_to_bytes(x)
        t0 = now_ms()
        resp = requests.post(f"{args.server}/infer_full",
                             files={"x": ("x.pt", x_bytes, "application/octet-stream")},
                             timeout=60)
        t1 = now_ms()
        resp.raise_for_status()
        _ = bytes_to_tensor(resp.content)

        cpu = psutil.cpu_percent(interval=None)
        append_csv_row(args.log, {
            "request_id": i,
            "strategy": "cloud_only",
            "split_id": "",
            "t_start_ms": f"{t0:.3f}",
            "t_end_ms": f"{t1:.3f}",
            "latency_ms": f"{(t1-t0):.3f}",
            "bytes_up": len(x_bytes),
            "bytes_down": len(resp.content),
            "server_compute_ms": resp.headers.get("X-Server-Compute-ms", ""),
            "cpu_percent": cpu
        }, LOG_FIELDS)

    print(f"Done. Wrote {args.log}")

if __name__ == "__main__":
    main()
