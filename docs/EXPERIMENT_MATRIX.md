# Experiment Matrix

This document maps the principal thesis campaigns to the preserved request-level logs. Request counts are per run.

## Principal campaigns

| Model | Campaign | Compared configurations | Requests | Deadline |
|---|---|---|---:|---:|
| ResNet-18 | Stable | Edge-only, cloud-only, fixed `layer2`, `layer3`, and `layer4`, mean-dynamic, and reliability-oriented | 500 | 200 ms |
| ResNet-18 | Five variable conditions | Fixed `layer4`, mean-dynamic, and reliability-oriented | 1200 | 250 ms |
| MobileNetV2 | Stable | Edge-only, cloud-only, fixed `features_6`, `features_10`, `features_13`, `features_16`, and `features_18`, mean-dynamic, and reliability-oriented | 5000 | 200 ms |
| MobileNetV2 | Five variable conditions | Fixed `features_16`, mean-dynamic, and reliability-oriented | 5000 | 200 ms |
| Covertype MLP | Stable | Edge-only, cloud-only, fixed `h1`, `h2`, and `h3`, mean-dynamic, and reliability-oriented | 5000 | 80 ms |
| Covertype MLP | Five variable conditions | Fixed `h3`, mean-dynamic, and reliability-oriented | 5000 | 80 ms |

The inventory contains 23 principal stable runs and 45 principal variable runs. This gives 68 principal strategy runs. Correctness checks, offline profiling, and additional local or exploratory ResNet-18 runs are separate from this count.

## Stable-run files

### ResNet-18

- `logs/edge_only_run1.csv`
- `logs/cloud_only_aws_run1.csv`
- `logs/aws_fixed_layer2.csv`
- `logs/aws_fixed_layer3.csv`
- `logs/aws_fixed_layer4.csv`
- `logs/aws_mean_dynamic.csv`
- `logs/aws_reliability_dynamic.csv`

### MobileNetV2

- `logs/mobilenetv2_edge_only.csv`
- `logs/mobilenetv2_cloud_only_aws.csv`
- `logs/mobilenetv2_fixed_f6.csv`
- `logs/mobilenetv2_fixed_f10.csv`
- `logs/mobilenetv2_fixed_f13.csv`
- `logs/mobilenetv2_fixed_f16.csv`
- `logs/mobilenetv2_fixed_f18.csv`
- `logs/mobilenetv2_mean_dynamic.csv`
- `logs/mobilenetv2_reliability_dynamic.csv`

### Covertype MLP

- `logs/mlp_covertype_edge_only.csv`
- `logs/mlp_covertype_cloud_only_aws.csv`
- `logs/mlp_covertype_fixed_h1.csv`
- `logs/mlp_covertype_fixed_h2.csv`
- `logs/mlp_covertype_fixed_h3.csv`
- `logs/mlp_covertype_mean_dynamic.csv`
- `logs/mlp_covertype_reliability_dynamic.csv`

## Variable-condition files

Each model has three files for each of these conditions:

- `bwflap`
- `jitterburst`
- `serverload`
- `edgeload`
- `bothload`

The three strategies are the stable-selected fixed split, mean-dynamic, and reliability-oriented. Filenames combine the model prefix, condition, and strategy. For example, the ResNet-18 bandwidth-fluctuation files are:

- `logs/aws_bwflap_fixed_layer4.csv`
- `logs/aws_bwflap_mean_dynamic.csv`
- `logs/aws_bwflap_reliability_dynamic.csv`

MobileNetV2 uses the `mobilenetv2_` prefix and fixed split `f16`. The MLP uses the `mlp_covertype_` prefix and fixed split `h3`.

The scenario field inside older variable-run CSV rows retains the program's default scenario label. Use the documented filename and campaign mapping above to identify those preserved runs.

## Supporting evidence

- Split correctness is recorded in `results/correctness_*_aws.csv`.
- Candidate profiling is recorded in `results/profiling_*_aws.csv`.
- Switching analysis uses `chosen_split`, `switched`, and `request_id` in the dynamic logs.
- Latency analysis uses `latency_ms` in the request logs. The fixed and dynamic strategy logs also record `deadline_ms` and `missed_deadline`. The edge-only and cloud-only baseline logs do not store those two deadline fields, so their miss rates are calculated from `latency_ms` using the campaign deadline in the table above.
- Communication analysis uses `bytes_up` and `bytes_down`.
- Throughput analysis uses request timestamps and run duration.
- The available edge CPU indicator is recorded in `cpu_percent`.
