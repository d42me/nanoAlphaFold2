# Run C — cluster-MSA versus extra-MSA path ablation

## Question

At the best aggregate depth of 64, does evolutionary information affect predictions through the rich cluster-MSA path, the cheaper extra-MSA tower, or both?

## Controlled design

For each of three seeds, one permutation of homolog rows defines a fixed cluster subset and a fixed extra subset. Conditions selectively remove paths without changing the retained row identities.

| Condition | Main MSA input | Extra-MSA input |
|---|---:|---:|
| Query only | query row | none |
| Cluster only | query + 63 homologs | none |
| Extra only | query row | 64 homologs |
| Both paths | query + 63 homologs | 64 homologs |

Shared settings:

- Checkpoint: `checkpoints/ckpt_runC.pt`, step 250,000
- Checkpoint SHA-256: `9073afa220b47a1c570ed5caa239eacd6f41162263add6ee7e1cf82fcf4ea5ea`
- Seeds: `42`, `43`, `44`
- Recycles: 3
- Masking: 0%
- Metric: Kabsch-aligned Cα RMSD in ångströms; lower is better
- Source: checkpoint-compatible revision `dc8de3f` plus [`checkpoint_compat_empty_extra.patch`](checkpoint_compat_empty_extra.patch)

Command:

```bash
PYTHONPATH=/tmp/af2-eval-dc8de3f/src python scripts/evaluate_msa_paths.py \
  --checkpoint checkpoints/ckpt_runC.pt \
  --output results/metrics/run_c_msa_paths.csv
```

Raw log: `logs/eval_runC_msa_paths.log`

## Results

Values are mean ± sample standard deviation over three fixed MSA subsets.

| Condition | `crambin` | `cystatin-b` | `hbb` | `interferon_gamma` | `profilin-1` | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Query only | **10.01 ± 0.00** | **15.45 ± 0.00** | 12.18 ± 0.00 | 24.90 ± 0.00 | **14.72 ± 0.00** | 15.45 |
| Cluster only | **10.01 ± 0.00** | **15.45 ± 0.00** | 12.18 ± 0.00 | 24.90 ± 0.00 | **14.72 ± 0.00** | 15.45 |
| Extra only | 10.95 ± 0.17 | 17.57 ± 0.31 | **6.07 ± 0.08** | **21.44 ± 0.14** | 16.30 ± 0.46 | **14.46** |
| Both paths | 10.95 ± 0.17 | 17.57 ± 0.31 | **6.07 ± 0.08** | **21.44 ± 0.14** | 16.30 ± 0.46 | **14.46** |

Machine-readable files:

- [`../metrics/run_c_msa_paths.csv`](../metrics/run_c_msa_paths.csv) — all 60 measurements
- [`../metrics/run_c_msa_paths_summary.csv`](../metrics/run_c_msa_paths_summary.csv) — aggregated means and deviations

## Verification of the surprising equality

At four-decimal RMSD precision, cluster-only equals query-only and extra-only equals both-paths for every protein and seed. An internal-coordinate check on `hbb` confirmed this is not caused by Kabsch alignment hiding a meaningful shape change:

- Query-only versus cluster-only: maximum coordinate change `6.3e-5 Å`; maximum pairwise-distance change `0.011 Å`
- Query-only versus extra-only: maximum coordinate change `19.88 Å`; maximum pairwise-distance change `26.43 Å`

The both-path condition exactly reproduces the independently generated depth-64 measurements.

## Interpretation

For this final checkpoint, non-query rows in the main MSA representation have effectively no influence on predicted structure. The complete MSA benefit—and complete MSA harm—comes through the extra-MSA tower. Although the query-row conditions here retain the full-MSA profile channels, [`run_c_profile_paths.md`](run_c_profile_paths.md) confirms that removing those channels also leaves outputs unchanged.

This sharpens the earlier conclusions:

- `hbb` homolog transfer is real, but it is implemented almost entirely by the extra-MSA → pair-representation path.
- The degradation on `crambin`, `cystatin-b`, and `profilin-1` also originates in that path.
- The earlier depth curve is functionally an extra-MSA-depth response; increasing cluster depth contributed negligibly at the tested checkpoint.

This is a checkpoint-specific learned behavior, not a claim that the architecture can never use cluster-MSA rows.

## Next ablation

Destroy cross-position covariance inside the extra MSA while preserving each column's amino-acid frequencies. If `hbb` performance collapses, the tower uses co-evolution; if it remains, the gain comes primarily from conservation/profile-like marginals.
