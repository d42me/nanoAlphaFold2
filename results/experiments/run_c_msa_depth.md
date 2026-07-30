# Run C — MSA-depth dose response

## Question

Does held-out accuracy improve progressively as more homologous sequences are available, and are the negative full-MSA results caused only by very deep/noisy alignments?

## Design

- Checkpoint: `checkpoints/ckpt_runC.pt`, step 250,000
- Checkpoint SHA-256: `9073afa220b47a1c570ed5caa239eacd6f41162263add6ee7e1cf82fcf4ea5ea`
- Depths: `1`, `8`, `32`, `64`, `192`
- Seeds: `42`, `43`, `44`
- Held-out proteins: five from `configs/splits/val.txt`
- Recycles: 3
- Masking: 0%
- Metric: Kabsch-aligned Cα RMSD in ångströms; lower is better

Depth 1 contains the query row and no extra-MSA rows. Its standard feature tensor retains the full-MSA profile, but a later strict profile ablation shows that replacing this profile with the query one-hot leaves outputs unchanged. At every larger depth, both cluster and extra-MSA budgets equal the requested depth. Fixed-seed permutations make the sampled evidence approximately nested as depth increases. `crambin` has only 125 extra rows remaining at the 192-row cluster setting.

Evaluation uses checkpoint-compatible revision `dc8de3f` plus the recorded empty-extra-MSA guard in [`checkpoint_compat_empty_extra.patch`](checkpoint_compat_empty_extra.patch).

Command:

```bash
PYTHONPATH=/tmp/af2-eval-dc8de3f/src python scripts/evaluate_msa_depth.py \
  --checkpoint checkpoints/ckpt_runC.pt \
  --output results/metrics/run_c_msa_depth.csv
```

Raw log: `logs/eval_runC_msa_depth.log`

## Results

Values are mean ± sample standard deviation over three MSA-sampling seeds.

| Depth | `crambin` | `cystatin-b` | `hbb` | `interferon_gamma` | `profilin-1` | Mean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **10.01 ± 0.00** | **15.45 ± 0.00** | 12.18 ± 0.00 | 24.90 ± 0.00 | **14.72 ± 0.00** | 15.45 |
| 8 | 12.18 ± 2.11 | 17.62 ± 0.89 | 6.39 ± 0.22 | 21.82 ± 0.21 | 16.30 ± 0.59 | 14.86 |
| 32 | 10.61 ± 0.40 | 17.65 ± 0.95 | 6.29 ± 0.10 | 21.73 ± 0.10 | 16.07 ± 0.09 | 14.47 |
| 64 | 10.95 ± 0.17 | 17.57 ± 0.31 | **6.07 ± 0.08** | **21.44 ± 0.14** | 16.30 ± 0.46 | **14.46** |
| 192 | 11.16 ± 0.28 | 17.46 ± 0.10 | 6.18 ± 0.02 | 21.64 ± 0.01 | 16.00 ± 0.22 | 14.49 |

Machine-readable files:

- [`../metrics/run_c_msa_depth.csv`](../metrics/run_c_msa_depth.csv) — all 75 measurements
- [`../metrics/run_c_msa_depth_summary.csv`](../metrics/run_c_msa_depth_summary.csv) — aggregated means and deviations

## Interpretation

The aggregate benefit appears by depth 32 and saturates around depth 64. Increasing to 192 rows reduces sampling variance but does not improve mean accuracy.

Two distinct behaviors are visible:

1. **MSA-responsive proteins:** `hbb` improves by roughly 5.8 Å with only eight rows and reaches its best result near depth 64. `interferon_gamma` improves by roughly 3.5 Å, although its absolute prediction remains poor.
2. **MSA-negative proteins:** `crambin`, `cystatin-b`, and `profilin-1` are best query-only. Their degradation already appears at shallow depth and persists, so it is not merely a very-deep-alignment noise problem.

This supports the earlier conclusion: the model has learned a real evolutionary mechanism, but it applies that evidence reliably only in some families. More MSA depth cannot repair a mislearned or irrelevant family signal.

## Follow-up

The path-factor experiment is complete: [`run_c_msa_paths.md`](run_c_msa_paths.md). It shows that nearly the entire depth response comes through the extra-MSA tower; increasing non-query rows in the main MSA path has negligible influence at this checkpoint. [`run_c_profile_paths.md`](run_c_profile_paths.md) further shows that the full-MSA profile channels also have negligible influence.
