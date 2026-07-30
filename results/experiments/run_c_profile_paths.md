# Run C — sequence, profile, and extra-MSA factorial

## Question

The normal `msa_feat` concatenates the full MSA profile into every cluster row. Does the apparent query-only prediction still use this profile, and does the extra-MSA effect depend on it?

## Conditions

| Condition | Query identity | Full MSA profile | Extra MSA |
|---|---:|---:|---:|
| Sequence only | yes | no; replaced by query one-hot | none |
| Profile only | yes | yes | none |
| Extra, no profile | yes | no; replaced by query one-hot | 64 rows |
| Extra + profile | yes | yes | 64 rows |

Three seeds (`42`, `43`, `44`), three recycles, no masking, and the final Run C checkpoint are used throughout. Evaluation uses checkpoint-compatible revision `dc8de3f` plus [`checkpoint_compat_empty_extra.patch`](checkpoint_compat_empty_extra.patch).

Command:

```bash
PYTHONPATH=/tmp/af2-eval-dc8de3f/src python scripts/evaluate_profile_paths.py \
  --checkpoint checkpoints/ckpt_runC.pt \
  --output results/metrics/run_c_profile_paths.csv
```

Raw log: `logs/eval_runC_profile_paths.log`

## Results

Values are mean ± sample standard deviation over three fixed MSA subsets.

| Condition | `crambin` | `cystatin-b` | `hbb` | `interferon_gamma` | `profilin-1` | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Sequence only | **10.01 ± 0.00** | **15.45 ± 0.00** | 12.18 ± 0.00 | 24.90 ± 0.00 | **14.72 ± 0.00** | 15.45 |
| Profile only | **10.01 ± 0.00** | **15.45 ± 0.00** | 12.18 ± 0.00 | 24.90 ± 0.00 | **14.72 ± 0.00** | 15.45 |
| Extra, no profile | 10.95 ± 0.17 | 17.57 ± 0.31 | **6.07 ± 0.08** | **21.44 ± 0.14** | 16.30 ± 0.46 | **14.46** |
| Extra + profile | 10.95 ± 0.17 | 17.57 ± 0.31 | **6.07 ± 0.08** | **21.44 ± 0.14** | 16.30 ± 0.46 | **14.46** |

Machine-readable files:

- [`../metrics/run_c_profile_paths.csv`](../metrics/run_c_profile_paths.csv)
- [`../metrics/run_c_profile_paths_summary.csv`](../metrics/run_c_profile_paths_summary.csv)

## Interpretation

Adding the full MSA profile changes no sequence-only result and changes extra-MSA results by at most `0.0001 Å`, below meaningful numerical precision. The final checkpoint therefore ignores two intended evolutionary routes:

1. non-query rows in the main MSA representation;
2. the 22-channel full-MSA profile concatenated into `msa_feat`.

All measurable evolutionary influence enters through explicit extra-MSA rows. This validates the earlier “query-only” result functionally, although its original tensor technically retained the profile channels.

The equality is a property of this trained checkpoint's outputs, not a claim that the architecture cannot use profiles.
