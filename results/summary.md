# Results summary

## Current conclusion

The compact model memorizes training structures and transfers within a homologous family, but it does not yet generalize to unrelated folds. Increasing capacity or extending cosine training did not improve combined held-out performance. Run C's validation mean peaked early and worsened while training fit continued to improve, indicating overfitting.

## Multi-protein runs

| Run | Parameters | Train proteins | Training | Best mean held-out RMSD | Step | Final mean held-out RMSD |
|---|---:|---:|---|---:|---:|---:|
| A | 2.02M | 59 | 150k, constant LR | **12.62 Å** | 32k | 13.08 Å |
| B | 7.69M | 59 | 100k, constant LR | 13.26 Å | 72k | 13.75 Å |
| C | 7.69M | 56 | 250k, cosine `1e-3 → 1e-4`, four exclusions | 13.57 Å | 62k | 14.56 Å |

Runs A/B and C are not perfectly controlled because Run C excludes four poor teachers. Per-protein and provenance details are in [`experiments/run_c_training.md`](experiments/run_c_training.md) and the raw logs referenced there.

## Earlier evidence

- Single-protein distillation reached approximately `0.06–0.09 Å` Cα RMSD, demonstrating memorization capacity.
- The initial eight-protein experiment predicted held-out `hbb` at `4.94 Å` with its MSA versus `12.42 Å` query-only, demonstrating homolog-family transfer.

## Run C MSA ablation

| Protein | Full MSA | Query-only | MSA benefit |
|---|---:|---:|---:|
| `crambin` | 11.15 Å | **10.01 Å** | −1.14 Å |
| `cystatin-b` | 17.53 Å | **15.45 Å** | −2.08 Å |
| `hbb` | **6.19 Å** | 12.18 Å | **+5.99 Å** |
| `interferon_gamma` | **21.65 Å** | 24.90 Å | +3.25 Å |
| `profilin-1` | 16.26 Å | **14.72 Å** | −1.54 Å |

Full MSA improves the mean from `15.45 Å` to `14.55 Å`, but worsens the median from `14.72 Å` to `16.26 Å`. The mean gain is dominated by `hbb`, confirming homolog-family transfer rather than reliable broad MSA use. See [`experiments/run_c_msa_ablation.md`](experiments/run_c_msa_ablation.md).

## MSA-depth curve

| Depth | Mean held-out RMSD |
|---:|---:|
| 1 | 15.45 Å |
| 8 | 14.86 Å |
| 32 | 14.47 Å |
| 64 | **14.46 Å** |
| 192 | 14.49 Å |

The aggregate gain saturates around 32–64 rows. `hbb` and `interferon_gamma` improve sharply, while `crambin`, `cystatin-b`, and `profilin-1` remain best query-only at every tested depth. Degradation on those proteins is therefore not only a deep-MSA noise effect. See [`experiments/run_c_msa_depth.md`](experiments/run_c_msa_depth.md).

## MSA-path ablation at depth 64

| Condition | Mean held-out RMSD |
|---|---:|
| Query only | 15.45 Å |
| Cluster only | 15.45 Å |
| Extra only | **14.46 Å** |
| Both paths | **14.46 Å** |

At four-decimal RMSD precision, cluster-only matches query-only and extra-only matches both paths for every protein and seed. Internal-coordinate checks confirm that the main MSA rows have a negligible structural effect; nearly all learned MSA influence enters through the extra-MSA tower. See [`experiments/run_c_msa_paths.md`](experiments/run_c_msa_paths.md).

## Profile and covariance controls

The strict sequence/profile factorial shows that the full-MSA profile channels have no measurable effect: sequence-only equals sequence + profile, and extra MSA without profile equals extra MSA with profile at four-decimal RMSD precision.

Independently shuffling each extra-MSA column preserves amino-acid frequencies but destroys cross-position covariance:

| Extra-MSA condition | Mean held-out RMSD |
|---|---:|
| Real rows | 14.46 Å |
| Whole-row permutation control | 14.46 Å |
| Independently column-shuffled | **14.27 Å** |

Column shuffling costs only `0.22 Å` on `hbb` and `0.09 Å` on `interferon_gamma`; most of their extra-MSA gain remains. It improves all three MSA-negative proteins. The model therefore relies primarily on per-column marginal/conservation-like information, with only a small useful covariance contribution in the two responsive families. See [`experiments/run_c_profile_paths.md`](experiments/run_c_profile_paths.md) and [`experiments/run_c_extra_covariance.md`](experiments/run_c_extra_covariance.md).

## Active: Run D

Run D is training the checkpoint-compatible large architecture with the extra-MSA tower removed (`n_extra=0`, `n_ext=0`). The main MSA retains 192 rows, forcing any learned evolutionary transfer through the cluster-MSA path. See [`experiments/run_d_no_extra.md`](experiments/run_d_no_extra.md).

Success requires more than low training RMSD: after training, cluster-MSA input must change held-out predictions relative to strict sequence-only and ideally recover the `hbb` transfer gain.
