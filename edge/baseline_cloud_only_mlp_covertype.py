import argparse
import psutil
import requests

from shared.data_covertype import get_covertype_test_loader
from shared.serialization import tensor_to_bytes, bytes_to_tensor
from shared.logging_utils import now_ms, append_csv_row
from edge.common_mlp_covertype import DEFAULT_SERVER_URL

LOG_FIELDS = ["request_id","strategy","split_id","t_start_ms","t_end_ms","latency_ms","bytes_up","bytes_down","server_compute_ms","cpu_percent"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--log", type=str, default="logs/mlp_covertype_cloud_only.csv")
    ap.add_argument("--server", type=str, default=DEFAULT_SERVER_URL)
    args = ap.parse_args()

    loader = get_covertype_test_loader(limit=args.n, batch_size=1)

    for i, (x, _) in enumerate(loader):
        x_bytes = tensor_to_bytes(x)
        t0 = now_ms()
        resp = requests.post(
            f"{args.server}/infer_full_mlp_covertype",
            files={"x": ("x.pt", x_bytes, "application/octet-stream")},
            timeout=120
        )
        t1 = now_ms()
        resp.raise_for_status()
        _ = bytes_to_tensor(resp.content)

        append_csv_row(args.log, {
            "request_id": i,
            "strategy": "cloud_only_mlp_covertype",
            "split_id": "",
            "t_start_ms": f"{t0:.3f}",
            "t_end_ms": f"{t1:.3f}",
            "latency_ms": f"{(t1-t0):.3f}",
            "bytes_up": len(x_bytes),
            "bytes_down": len(resp.content),
            "server_compute_ms": resp.headers.get("X-Server-Compute-ms", ""),
            "cpu_percent": psutil.cpu_percent(interval=None)
        }, LOG_FIELDS)
    print(f"Done. Wrote {args.log}")

if __name__ == "__main__":
    main()
