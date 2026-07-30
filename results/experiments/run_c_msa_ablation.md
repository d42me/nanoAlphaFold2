# Run C — full-MSA versus query-only ablation

## Question

Does the final Run C checkpoint use evolutionary information from homologous sequences, or can it make the same predictions from the query sequence alone?

## Conditions

Both conditions use the same final checkpoint, held-out proteins, random seed, and recycle count.

| Condition | Cluster MSA | Extra MSA | Masking |
|---|---:|---:|---:|
| Full MSA | up to 192 rows | up to 192 rows | 0% |
| Query-only | query row only | zero rows | 0% |

The standard query-row tensor technically retains the precomputed full-MSA profile channels. A later strict factorial ([`run_c_profile_paths.md`](run_c_profile_paths.md)) replaces those channels with the query one-hot and produces identical results at four-decimal RMSD precision. “Query-only” is therefore functionally accurate for this checkpoint.

Shared settings:

- Checkpoint: `checkpoints/ckpt_runC.pt`, step 250,000
- Checkpoint SHA-256: `9073afa220b47a1c570ed5caa239eacd6f41162263add6ee7e1cf82fcf4ea5ea`
- Held-out split: `configs/splits/val.txt`
- Seed: 42
- Recycles: 3
- Metric: Kabsch-aligned Cα RMSD in ångströms; lower is better

## Source compatibility

Run C was trained with the architecture represented by Git revision `dc8de3f`. The later architecture-refinement commit changes attention parameterization and structure-module weight sharing, so it cannot load this checkpoint.

Evaluation therefore used a detached `dc8de3f` worktree. Query-only input has zero extra-MSA rows; the legacy source required a one-line guard to skip the extra-MSA tower for an empty tensor. The exact patch is stored in [`checkpoint_compat_empty_extra.patch`](checkpoint_compat_empty_extra.patch). This guard does not affect full-MSA evaluation, whose values reproduce the final training log.

Command:

```bash
PYTHONPATH=/tmp/af2-eval-dc8de3f/src python scripts/evaluate_msa_ablation.py \
  --checkpoint checkpoints/ckpt_runC.pt \
  --output results/metrics/run_c_msa_ablation.csv
```

Raw evaluation log: `logs/eval_runC_msa_ablation.log`

## Results

`Δ benefit = query-only RMSD − full-MSA RMSD`; positive values mean the MSA helped.

| Protein | Full MSA | Query-only | MSA benefit |
|---|---:|---:|---:|
| `crambin` | 11.15 Å | **10.01 Å** | −1.14 Å |
| `cystatin-b` | 17.53 Å | **15.45 Å** | −2.08 Å |
| `hbb` | **6.19 Å** | 12.18 Å | **+5.99 Å** |
| `interferon_gamma` | **21.65 Å** | 24.90 Å | +3.25 Å |
| `profilin-1` | 16.26 Å | **14.72 Å** | −1.54 Å |
| **Mean** | **14.55 Å** | 15.45 Å | +0.90 Å |
| **Median** | 16.26 Å | **14.72 Å** | −1.54 Å |

Machine-readable measurements: [`../metrics/run_c_msa_ablation.csv`](../metrics/run_c_msa_ablation.csv)

## Interpretation

The final checkpoint uses MSA evidence strongly for `hbb`, confirming homolog-family transfer. MSA evidence also improves `interferon_gamma`, but both predictions remain poor in absolute terms. For the other three held-out proteins, adding the MSA makes RMSD worse.

The 0.90 Å mean improvement is therefore not evidence of broad evolutionary generalization: it is dominated by two proteins, especially `hbb`, while the median result worsens. The model has learned to exploit useful homolog evidence in at least one family but does not use MSA evidence reliably across unrelated folds.

## Next ablation

Measure an MSA-depth curve (`1`, `8`, `32`, `64`, `192` cluster rows, with matched extra-MSA settings) to determine whether `hbb` improves progressively with evolutionary depth and whether the three negative cases degrade monotonically or only under deep/noisy alignments.
