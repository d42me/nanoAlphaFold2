# Run E — no extra MSA with stochastic profile dropout

**Status: completed**

## Question

Can stochastic removal of the precomputed MSA profile force useful learning from actual non-query homolog rows after the extra-MSA tower has already been removed?

Run D showed that removing the extra tower redirected evolutionary learning mainly into profile channels: profile-only reached `13.50 Å` mean held-out RMSD, while 192 cluster rows without profile reached `14.40 Å` and added only mixed value.

## Controlled change

Run E matches Run D except that each training batch independently replaces the full-MSA profile channels with the query one-hot with probability `0.5`.

```text
Run D: profile dropout = 0.0
Run E: profile dropout = 0.5
```

The replacement remains a valid categorical profile rather than an all-zero out-of-distribution tensor.

## Configuration

| Field | Value |
|---|---|
| Source architecture | Git revision `dc8de3f` |
| Model size | 7,116,276 parameters |
| Training proteins | 56 |
| Held-out proteins | `crambin`, `cystatin-b`, `hbb`, `interferon_gamma`, `profilin-1` |
| Exclusions | `configs/splits/exclude_runC.txt` |
| Steps | 250,000 |
| Learning rate | cosine, `1e-3 → 1e-4` |
| Cluster MSA | 192 rows |
| Extra MSA | disabled |
| Profile dropout | 0.5 per training batch |
| Evaluation | full profile, seed 42, no masking, three recycles |
| Metric | Kabsch-aligned Cα RMSD |

Command:

```bash
PYTHONPATH=/tmp/af2-eval-dc8de3f/src python scripts/train_multi.py \
  --tag ckpt_runE_profile_dropout \
  --big \
  --steps 250000 \
  --cosine \
  --min-lr 1e-4 \
  --exclude-file configs/splits/exclude_runC.txt \
  --no-extra-msa \
  --profile-dropout 0.5
```

Artifacts:

- Latest checkpoint: `checkpoints/ckpt_runE_profile_dropout.pt`
- Best-validation checkpoint: `checkpoints/ckpt_runE_profile_dropout_best.pt`
- Log: `logs/train_runE_profile_dropout.log`
- tmux session: `runE`

## Results

| Metric | Value |
|---|---:|
| Best held-out mean | **12.92 Å at step 36,000** |
| Final held-out mean | **13.42 Å** |
| Final training mean / median | 3.22 / 2.74 Å |

| Protein | Best-checkpoint RMSD | Final RMSD |
|---|---:|---:|
| `crambin` | 8.21 Å | 8.41 Å |
| `cystatin-b` | 11.79 Å | 12.69 Å |
| `hbb` | 9.37 Å | 8.78 Å |
| `interferon_gamma` | 21.43 Å | 20.78 Å |
| `profilin-1` | 13.81 Å | 16.45 Å |

Run E improved the best aggregate validation score by `0.45 Å` over Run D, but lost much of Run D's strong `hbb` transfer. Validation again peaked early and degraded while training fit continued to improve.

## Causal follow-up

The strict factorial found that non-query cluster rows improve the mean by `0.31 Å` at the best checkpoint and `0.18 Å` at the final checkpoint when profile channels are absent. Profile channels have effectively no impact. This is the first positive aggregate evidence that a long run uses actual homolog rows through the main MSA stack, although the effect remains modest and mixed across proteins. See [`run_e_cluster_paths.md`](run_e_cluster_paths.md).
