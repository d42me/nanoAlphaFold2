# Run D — main-MSA path evaluation

**Status: complete**

## Question

Did removing the extra-MSA tower force Run D to use non-query rows or profile channels in the main MSA representation?

## Conditions

| Condition | Query identity | Full MSA profile | Main MSA rows | Extra MSA |
|---|---:|---:|---:|---:|
| Sequence only | yes | no | 1 | 0 |
| Profile only | yes | yes | 1 | 0 |
| Cluster, no profile | yes | no | up to 192 | 0 |
| Cluster + profile | yes | yes | up to 192 | 0 |

Evaluation uses seeds `42`, `43`, and `44`, three recycles, no masking, the five held-out proteins, and Kabsch-aligned Cα RMSD.

The evaluator refuses non-final checkpoints: `checkpoints/ckpt_runD_no_extra.pt` must report step 250,000.

Command:

```bash
PYTHONPATH=/tmp/af2-eval-dc8de3f/src python scripts/evaluate_cluster_paths.py \
  --checkpoint checkpoints/ckpt_runD_no_extra.pt \
  --output results/metrics/run_d_cluster_paths.csv
```

Artifacts:

- Raw log: `logs/eval_runD_cluster_paths.log`
- Measurements: [`../metrics/run_d_cluster_paths.csv`](../metrics/run_d_cluster_paths.csv)
- Aggregates: [`../metrics/run_d_cluster_paths_summary.csv`](../metrics/run_d_cluster_paths_summary.csv)

## Results

Values are mean ± sample standard deviation over three fixed MSA subsets.

| Condition | `crambin` | `cystatin-b` | `hbb` | `interferon_gamma` | `profilin-1` | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Sequence only | 15.58 ± 0.00 | **12.34 ± 0.00** | 7.46 ± 0.00 | 23.53 ± 0.00 | **13.71 ± 0.00** | 14.52 |
| Profile only | **14.12 ± 0.00** | 12.52 ± 0.00 | 6.34 ± 0.00 | **20.34 ± 0.00** | 14.16 ± 0.00 | **13.50** |
| Cluster, no profile | 14.26 ± 0.02 | **12.18 ± 0.02** | 8.09 ± 0.01 | 23.71 ± 0.01 | 13.77 ± 0.01 | 14.40 |
| Cluster + profile | 14.43 ± 0.03 | 13.02 ± 0.02 | **6.24 ± 0.00** | 20.55 ± 0.01 | 14.22 ± 0.01 | 13.69 |

## Interpretation

Removing the extra tower successfully made the full-MSA profile useful: profile-only improves the aggregate by `1.03 Å` over strict sequence-only, with large gains on `crambin`, `hbb`, and `interferon_gamma`.

Non-query cluster rows have only a small, mixed effect:

- without profile, they improve `crambin` by `1.33 Å` and `cystatin-b` by `0.15 Å`, but worsen the other three proteins;
- with profile, they improve `hbb` by only `0.10 Å` and worsen the aggregate by `0.20 Å` relative to profile-only.

Run D therefore redirected the shortcut from the extra-MSA tower primarily into precomputed profile features. It activated the main-row path slightly, but did not learn robust co-evolutionary reasoning over homolog rows.
