# Run E — profile and cluster-row factorial

**Status: completed**

## Question

Did 50% profile dropout make Run E use actual non-query homolog rows, or did its aggregate validation gain come from profile channels?

## Method

Evaluate the best (step 36k) and final (step 250k) checkpoints under four strict conditions, using MSA depth 192, seeds 42–44, three recycles, and the five held-out proteins:

1. **Sequence only** — query row with query one-hot profile.
2. **Profile only** — query row with full-MSA profile.
3. **Cluster, no profile** — query plus non-query rows with query one-hot profile.
4. **Cluster + profile** — query plus non-query rows with full-MSA profile.

Commands:

```bash
PYTHONPATH=/tmp/af2-eval-dc8de3f/src python scripts/evaluate_cluster_paths.py \
  --checkpoint checkpoints/ckpt_runE_profile_dropout_best.pt \
  --output results/metrics/run_e_cluster_paths_best.csv

PYTHONPATH=/tmp/af2-eval-dc8de3f/src python scripts/evaluate_cluster_paths.py \
  --checkpoint checkpoints/ckpt_runE_profile_dropout.pt \
  --output results/metrics/run_e_cluster_paths_final.csv
```

## Results

| Checkpoint | Sequence only | Profile only | Cluster, no profile | Cluster + profile |
|---|---:|---:|---:|---:|
| Best, step 36k | 13.19 Å | 13.25 Å | **12.88 Å** | 12.88 Å |
| Final, step 250k | 13.57 Å | 13.56 Å | **13.39 Å** | 13.39 Å |

Best-checkpoint per-protein means:

| Protein | Sequence only | Cluster, no profile | Cluster gain |
|---|---:|---:|---:|
| `crambin` | **7.71 Å** | 8.21 Å | -0.50 Å |
| `cystatin-b` | 11.97 Å | **11.72 Å** | +0.25 Å |
| `hbb` | 9.50 Å | **9.36 Å** | +0.14 Å |
| `interferon_gamma` | 22.93 Å | **21.44 Å** | +1.49 Å |
| `profilin-1` | 13.85 Å | **13.68 Å** | +0.17 Å |

Positive gain means lower RMSD after adding non-query rows.

## Conclusion

Run E is the first long-run checkpoint in this series to show an aggregate benefit from actual main-MSA homolog rows without profile channels: `0.31 Å` at the best checkpoint and `0.18 Å` at the final checkpoint. Full-MSA profile channels have effectively no impact, so the gain is not a profile shortcut.

The effect remains modest and mixed—most notably, cluster rows worsen `crambin`—but it is sufficient to trigger the planned covariance controls. Whole-row permutation should be invariant; independently shuffling rows per residue column will reveal how much of the gain depends on cross-position covariance rather than per-column marginals.
