# Reliability-Aware Dynamic Split Inference

This repository contains the research artefact for the master's thesis
*Reliability-Aware Dynamic Split Inference for Edge-Cloud DNNs under Variable Conditions* by Sai Manas Masetty.

The version used for the final thesis is release [`v1.0.0`](https://github.com/SaiManas2106/reliability-aware-split-inference/releases/tag/v1.0.0).

The artefact supports inspection of the implementation and the preserved result evidence. It also provides the files needed to repeat the documented workflow. Exact numerical results can vary because cloud load, network conditions, and hardware performance can change over time.

## Contents

- `edge/` contains the experiment clients, baselines, split validation, profiling, and runtime strategies.
- `server/` contains the Flask inference server.
- `shared/` contains the split-model implementations, data loaders, serialization, measurement, and logging support.
- `configs/` contains the candidate split definitions for ResNet-18, MobileNetV2, and the Covertype MLP.
- `train/` contains the MLP training script.
- `artifacts/` contains the retained MLP checkpoint.
- `results/` contains correctness and profiling CSV files.
- `logs/` contains the preserved request-level measurements.
- `analysis/` contains the available log-summary script.
- `docs/` describes the experiment matrix and the reproduction boundary.
- `SHA256SUMS.txt` records checksums for the files in the release.

The package does not include downloaded copies of CIFAR-10, UCI Covertype, or torchvision model weights. The data loaders obtain these external resources through torchvision and scikit-learn when required. See `THIRD_PARTY_NOTICES.md`.

## Environment

The recovered local environment used Python 3.12.10 with the package versions listed in `requirements.txt`. These versions document the checked local project environment. They are not a record of every package installed on the original remote server.

Create and activate a virtual environment, then install the dependencies.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first CNN run may download CIFAR-10 and pretrained torchvision weights. The first Covertype data-loading run may download the UCI Covertype dataset through scikit-learn. Internet access and enough local storage are therefore required for a new run.

## Basic use

Run commands from the repository root. Start the server in one terminal.

```powershell
python -m server.app
```

The default server address is `http://127.0.0.1:5000`. Set the `HOST` and `PORT` environment variables when another bind address or port is required. The experiment clients accept a `--server` option.

In a second terminal, check one model split and create a local profile.

```powershell
New-Item -ItemType Directory -Force local_outputs
python -m edge.validate_splits --n 50 --tol 1e-4 --out local_outputs/correctness_resnet18.csv
python -m edge.profile_splits --n 200 --out local_outputs/profiling_resnet18.csv
```

The explicit `local_outputs/` paths keep new outputs separate from the preserved evidence included in the release.

The following command runs the ResNet-18 reliability-oriented strategy using the stable-run deadline reported in the thesis.

```powershell
python -m edge.run_experiment --strategy reliability_dynamic --scenario stable_local --n 500 --deadline_ms 200 --profile_csv local_outputs/profiling_resnet18.csv --log local_outputs/reliability_dynamic.csv
```

Equivalent model-specific entry points are available for MobileNetV2 and the Covertype MLP. Use `--help` with an entry point to see its arguments. The preserved campaign settings and filenames are listed in `docs/EXPERIMENT_MATRIX.md`.

## Inspecting a preserved log

The included summary script reports request count, mean, median, p95, p99, deadline miss rate, and switch count when the relevant fields are present.

```powershell
python -m analysis.summarize_logs --log logs/aws_reliability_dynamic.csv --deadlines 200,250
```

The script does not generate every thesis table or figure. Other reported measures use the timestamp, byte, CPU, split, and deadline fields in the request-level CSV files. `docs/REPRODUCIBILITY.md` explains this boundary.

## Safety

The Flask service and tensor serialization were created for a controlled research environment. They do not provide authentication or production hardening. Do not expose the service directly to the public Internet. See `SECURITY.md`.

## Citation

Citation metadata is provided in `CITATION.cff`. Use the version-specific release when citing the artefact.

## License

Original source code and configuration files are available under the MIT License. Original documentation, logs, result CSV files, and the retained MLP checkpoint are available under the Creative Commons Attribution 4.0 International License. Third-party resources remain subject to their original terms. See `LICENSE`, `LICENSE-CODE`, `LICENSE-DATA`, and `THIRD_PARTY_NOTICES.md`.
