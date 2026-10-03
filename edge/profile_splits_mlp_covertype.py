import argparse, csv
import requests
import torch

from shared.data_covertype import get_covertype_test_loader
from shared.split_mlp_covertype import MLPCovertypeSplitter
from shared.serialization import tensor_to_bytes, bytes_to_tensor
from shared.logging_utils import now_ms
from edge.common_mlp_covertype import DEFAULT_SERVER_URL, DEFAULT_CHECKPOINT, load_splits

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=str, default="results/profiling_mlp_covertype.csv")
    ap.add_argument("--server", type=str, default=DEFAULT_SERVER_URL)
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    args = ap.parse_args()

    splitter = MLPCovertypeSplitter(checkpoint_path=args.checkpoint, device=torch.device("cpu"))
    split_ids = [s["split_id"] for s in load_splits()]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "split_id","front_ms_mean","back_ms_mean","act_bytes_mean","http_roundtrip_ms_mean","server_compute_ms_mean"
        ])
        w.writeheader()

        for split_id in split_ids:
            loader = get_covertype_test_loader(limit=args.n, batch_size=1)
            front_ms, back_ms, act_bytes_list, http_ms, server_ms = [], [], [], [], []

            for x, _ in loader:
                t0 = now_ms()
                act = splitter.front(x, split_id)
                t1 = now_ms()
                front_ms.append(t1 - t0)
                act_b = tensor_to_bytes(act)
                act_bytes_list.append(len(act_b))

                t2 = now_ms()
                resp = requests.post(
                    f"{args.server}/infer_back_mlp_covertype",
                    data={"split_id": split_id},
                    files={"act": ("act.pt", act_b, "application/octet-stream")},
                    timeout=120
                )
                t3 = now_ms()
                resp.raise_for_status()
                _ = bytes_to_tensor(resp.content)

                http_ms.append(t3 - t2)
                sc = float(resp.headers.get("X-Server-Compute-ms", "nan"))
                server_ms.append(sc)
                back_ms.append(sc)

            row = {
                "split_id": split_id,
                "front_ms_mean": f"{mean(front_ms):.3f}",
                "back_ms_mean": f"{mean(back_ms):.3f}",
                "act_bytes_mean": f"{mean(act_bytes_list):.1f}",
                "http_roundtrip_ms_mean": f"{mean(http_ms):.3f}",
                "server_compute_ms_mean": f"{mean(server_ms):.3f}",
            }
            w.writerow(row)
            print("Profiled", split_id, row)

    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
