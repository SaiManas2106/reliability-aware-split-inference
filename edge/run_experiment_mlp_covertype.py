import argparse, os
import psutil
import requests
import torch

from shared.data_covertype import get_covertype_test_loader
from shared.split_mlp_covertype import MLPCovertypeSplitter
from shared.serialization import tensor_to_bytes, bytes_to_tensor
from shared.logging_utils import now_ms, append_csv_row
from shared.net_measure import BandwidthEstimator
from edge.common_mlp_covertype import DEFAULT_SERVER_URL, DEFAULT_CHECKPOINT, load_splits

LOG_FIELDS = [
    "request_id","strategy","scenario","chosen_split",
    "t_start_ms","t_end_ms","latency_ms",
    "front_ms","http_ms","server_compute_ms",
    "bytes_up","bytes_down",
    "deadline_ms","missed_deadline",
    "switched","cpu_percent"
]

def load_profile_csv(path):
    import csv
    prof = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prof[row["split_id"]] = row
    return prof

def predict_total_ms(profile, split_id, bw_bytes_per_ms):
    p = profile[split_id]
    front = float(p["front_ms_mean"])
    back = float(p["back_ms_mean"])
    act_bytes = float(p["act_bytes_mean"])
    net = (act_bytes / bw_bytes_per_ms) if bw_bytes_per_ms and bw_bytes_per_ms > 0 else float("inf")
    return front + back + net

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["edge_only","cloud_only","fixed_split","mean_dynamic","reliability_dynamic"])
    ap.add_argument("--scenario", type=str, default="stable")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--server", type=str, default=DEFAULT_SERVER_URL)
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--split_id", type=str, default="h3")
    ap.add_argument("--deadline_ms", type=float, default=80.0)
    ap.add_argument("--log", type=str, default="logs/mlp_covertype_experiment.csv")
    ap.add_argument("--profile_csv", type=str, default="results/profiling_mlp_covertype.csv")
    ap.add_argument("--hold_requests", type=int, default=30)
    ap.add_argument("--hysteresis_ms", type=float, default=4.0)
    ap.add_argument("--conservative_percentile", type=float, default=20.0)
    args = ap.parse_args()

    splitter = MLPCovertypeSplitter(checkpoint_path=args.checkpoint, device=torch.device("cpu"))
    split_ids = [s["split_id"] for s in load_splits()]
    profile = None
    if args.strategy in ("mean_dynamic", "reliability_dynamic"):
        if not os.path.exists(args.profile_csv):
            raise RuntimeError("Profiling table missing. Run: python -m edge.profile_splits_mlp_covertype")
        profile = load_profile_csv(args.profile_csv)

    bw = BandwidthEstimator(window=30)
    chosen = args.split_id
    hold_left = args.hold_requests
    loader = get_covertype_test_loader(limit=args.n, batch_size=1)

    for i, (x, _) in enumerate(loader):
        switched = 0
        t0 = now_ms()
        front_ms = 0.0
        http_ms = 0.0
        bytes_up = 0
        bytes_down = 0
        server_compute_ms = ""
        chosen_split = ""

        if args.strategy == "edge_only":
            tA = now_ms()
            _ = splitter.full(x)
            tB = now_ms()
            front_ms = tB - tA

        elif args.strategy == "cloud_only":
            x_bytes = tensor_to_bytes(x)
            tA = now_ms()
            resp = requests.post(
                f"{args.server}/infer_full_mlp_covertype",
                files={"x": ("x.pt", x_bytes, "application/octet-stream")},
                timeout=120
            )
            tB = now_ms()
            resp.raise_for_status()
            _ = bytes_to_tensor(resp.content)
            http_ms = tB - tA
            bytes_up = len(x_bytes)
            bytes_down = len(resp.content)
            server_compute_ms = resp.headers.get("X-Server-Compute-ms", "")

        else:
            if args.strategy == "fixed_split":
                chosen_split = args.split_id
            elif args.strategy == "mean_dynamic":
                est_bw = bw.mean_bytes_per_ms() or 1000.0
                preds = {s: predict_total_ms(profile, s, est_bw) for s in split_ids}
                chosen_split = min(preds, key=preds.get)
            elif args.strategy == "reliability_dynamic":
                default_bw = bw.mean_bytes_per_ms() or 1000.0
                est_bw = bw.percentile_bytes_per_ms(args.conservative_percentile) or default_bw
                if hold_left > 0:
                    chosen_split = chosen
                    hold_left -= 1
                else:
                    preds = {s: predict_total_ms(profile, s, est_bw) for s in split_ids}
                    best = min(preds, key=preds.get)
                    if best != chosen and (preds[chosen] - preds[best]) >= args.hysteresis_ms:
                        chosen = best
                        switched = 1
                    chosen_split = chosen
                    hold_left = args.hold_requests

            tA = now_ms()
            act = splitter.front(x, chosen_split)
            tB = now_ms()
            front_ms = tB - tA
            act_b = tensor_to_bytes(act)
            bytes_up = len(act_b)

            tC = now_ms()
            resp = requests.post(
                f"{args.server}/infer_back_mlp_covertype",
                data={"split_id": chosen_split},
                files={"act": ("act.pt", act_b, "application/octet-stream")},
                timeout=120
            )
            tD = now_ms()
            resp.raise_for_status()
            _ = bytes_to_tensor(resp.content)
            http_ms = tD - tC
            bytes_down = len(resp.content)
            server_compute_ms = resp.headers.get("X-Server-Compute-ms", "")
            bw.add(bytes_up, http_ms)

        t1 = now_ms()
        latency_ms = t1 - t0
        missed = 1 if latency_ms > args.deadline_ms else 0

        append_csv_row(args.log, {
            "request_id": i,
            "strategy": args.strategy + "_mlp_covertype",
            "scenario": args.scenario,
            "chosen_split": chosen_split,
            "t_start_ms": f"{t0:.3f}",
            "t_end_ms": f"{t1:.3f}",
            "latency_ms": f"{latency_ms:.3f}",
            "front_ms": f"{front_ms:.3f}",
            "http_ms": f"{http_ms:.3f}",
            "server_compute_ms": server_compute_ms,
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "deadline_ms": f"{args.deadline_ms:.1f}",
            "missed_deadline": missed,
            "switched": switched,
            "cpu_percent": psutil.cpu_percent(interval=None)
        }, LOG_FIELDS)

    print(f"Done. Wrote {args.log}")

if __name__ == "__main__":
    main()
