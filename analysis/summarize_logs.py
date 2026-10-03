import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--deadlines", default="100,150,200")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    lat = df["latency_ms"].astype(float)

    print(f"n={len(lat)}")
    print(f"mean={lat.mean():.3f} ms")
    print(f"median={lat.median():.3f} ms")
    print(f"p95={lat.quantile(0.95):.3f} ms")
    print(f"p99={lat.quantile(0.99):.3f} ms")

    for d in [float(x) for x in args.deadlines.split(",")]:
        miss = (lat > d).mean()
        print(f"miss_rate_{int(d)}ms={miss*100:.2f}%")

    if "switched" in df.columns:
        print("switch_count=", int(df["switched"].fillna(0).sum()))

if __name__ == "__main__":
    main()
