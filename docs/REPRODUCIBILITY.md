# Reproducibility and Interpretation

## What this release supports

This release supports:

- inspection of the split-model and controller implementation
- inspection of the candidate split configurations
- checking the preserved correctness and profiling outputs
- inspection and independent analysis of the request-level CSV logs
- repetition of the local workflow with the included entry points

`SHA256SUMS.txt` allows the files in a downloaded release to be checked against the curated package.

## What can change in a new run

Cloud load, network conditions, operating-system scheduling, downloaded dependency builds, and hardware performance can change. A new run should therefore be treated as a new measurement under the documented setup. Small numerical differences from the preserved CSV files are expected.

## Preserved environment information

The checked local project environment used Python 3.12.10 and the package versions in `requirements.txt`. The exact package inventory from the original remote server was not preserved.

## External condition controls

Network and load conditions were applied outside the Python experiment clients. A complete command-by-command history of those external controls was not preserved. The request rows also do not record the external control state independently. The filename mapping and the documented experimental procedure identify the preserved campaigns.

## Analysis boundary

`analysis/summarize_logs.py` calculates request count, mean latency, median latency, p95, p99, deadline miss rate for supplied deadlines, and switch count when available. It does not recreate every table or figure in the thesis.

The request-level CSV files contain the fields used for the additional analysis. These include timestamps, transferred bytes, chosen split, switch indicator, and the available edge CPU indicator. Fixed and dynamic strategy logs also contain deadline fields. The edge-only and cloud-only baseline logs are evaluated against the campaign deadlines documented in `EXPERIMENT_MATRIX.md`. Any new aggregation should state its formula and the exact input files used.

Latency is stored to three decimal places, while the recorded deadline-miss flag is calculated from the unrounded runtime value. A request very slightly above its deadline can therefore display a rounded latency equal to the deadline while still having `missed_deadline` set to `1`. Recomputing miss counts only from the rounded latency column can differ by a boundary row.

## Data and model downloads

The repository does not redistribute CIFAR-10, UCI Covertype, or pretrained torchvision weights. The Python data and model loaders download them when required. Review their original terms before use.
