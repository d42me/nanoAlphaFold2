# Run C — extra-MSA covariance ablation

## Question

Does the extra-MSA tower obtain its effect from cross-position co-evolution, or mainly from per-column amino-acid frequencies and conservation?

## Controlled perturbations

All extra-MSA conditions use the same 64 sampled homolog rows for each seed.

| Condition | Preserves column frequencies | Preserves cross-position covariance |
|---|---:|---:|
| Query only | — | — |
| Real extra MSA | yes | yes |
| Row-permuted extra MSA | yes | yes |
| Independently column-shuffled | yes | **no** |

A whole-row permutation is a negative control: the extra tower should be insensitive to sequence-row ordering. Independent row permutations at every residue column preserve each column's exact amino-acid/deletion distribution while destroying which residues co-occur in one homolog.

Shared settings: final Run C checkpoint, depth 64, seeds `42–44`, three recycles, no masking, checkpoint-compatible revision `dc8de3f` plus [`checkpoint_compat_empty_extra.patch`](checkpoint_compat_empty_extra.patch).

Command:

```bash
PYTHONPATH=/tmp/af2-eval-dc8de3f/src python scripts/evaluate_extra_covariance.py \
  --checkpoint checkpoints/ckpt_runC.pt \
  --output results/metrics/run_c_extra_covariance.csv
```

Raw log: `logs/eval_runC_extra_covariance.log`

## Results

Values are mean ± sample standard deviation over three fixed subsets.

| Condition | `crambin` | `cystatin-b` | `hbb` | `interferon_gamma` | `profilin-1` | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Query only | 10.01 ± 0.00 | 15.45 ± 0.00 | 12.18 ± 0.00 | 24.90 ± 0.00 | 14.72 ± 0.00 | 15.45 |
| Real extra | 10.95 ± 0.17 | 17.57 ± 0.31 | **6.07 ± 0.08** | **21.44 ± 0.14** | 16.30 ± 0.46 | 14.46 |
| Row-permuted | 10.95 ± 0.17 | 17.57 ± 0.31 | **6.07 ± 0.08** | **21.44 ± 0.14** | 16.30 ± 0.46 | 14.46 |
| Column-shuffled | **10.77 ± 0.21** | **17.07 ± 0.50** | 6.29 ± 0.05 | 21.52 ± 0.27 | **15.71 ± 0.20** | **14.27** |

Machine-readable files:

- [`../metrics/run_c_extra_covariance.csv`](../metrics/run_c_extra_covariance.csv)
- [`../metrics/run_c_extra_covariance_summary.csv`](../metrics/run_c_extra_covariance_summary.csv)

## Interpretation

The row-permutation control is exactly invariant, confirming that row ordering does not affect the result. Destroying covariance has only a small cost on the two MSA-responsive proteins:

- `hbb`: `+0.22 Å` worse than real extra MSA, while retaining roughly `5.89 Å` of the `6.11 Å` gain over sequence-only;
- `interferon_gamma`: `+0.09 Å` worse, retaining roughly `3.38 Å` of the `3.46 Å` gain.

For all three MSA-negative proteins, destroying covariance improves the prediction by `0.19–0.58 Å`. The aggregate column-shuffled mean is consequently better than the real-extra mean.

The final checkpoint's extra-MSA behavior is therefore driven primarily by per-column marginals/conservation-like information, not learned cross-position covariance. Genuine covariance contributes a small positive increment for `hbb` and `interferon_gamma`, while learned covariance is mildly harmful on the other proteins.

This explains why homolog-family transfer exists without broad novel-fold generalization: the model has mostly learned family-level marginal cues rather than a robust co-evolutionary geometry mechanism.
