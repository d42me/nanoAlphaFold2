# Run C — cleaned-teacher cosine training

## Question

Does a longer cosine schedule on the larger model improve held-out fold prediction after removing four teachers that both Runs A and B fit poorly?

## Configuration

| Field | Value |
|---|---|
| Parameters | 7,692,788 |
| Training proteins | 56 |
| Held-out proteins | `crambin`, `cystatin-b`, `hbb`, `interferon_gamma`, `profilin-1` |
| Excluded teachers | `rps27a`, `histone_h2b_type_1-j`, `nadh_dehydrogenase_[ubiquinone]_1_beta_s_2`, `v-type_proton_atpase_subunit_g_2` |
| Steps | 250,000 |
| Learning rate | cosine, `1e-3 → 1e-4` |
| Evaluation | fixed MSA sample, seed 42, no masking, three recycles |
| Metric | Kabsch-aligned Cα RMSD |

Command:

```bash
PYTHONPATH=src python scripts/train_multi.py \
  --tag ckpt_runC \
  --big \
  --steps 250000 \
  --cosine \
  --min-lr 1e-4 \
  --exclude-file configs/splits/exclude_runC.txt \
  --resume checkpoints/ckpt_runC.pt
```

Artifacts:

- Final checkpoint: `checkpoints/ckpt_runC.pt` at step 250,000
- Raw log: `logs/train_runC.log`
- Preserved pre-reorganization checkpoint: `/home/ubuntu/af2-from-scratch-artifacts/checkpoints/ckpt_runC.pt` at step 76,000

The run was interrupted during a directory reorganization after step 78,000 and resumed from step 76,000. Model weights and cosine position were restored; Adam moments were unavailable for that first resume. Subsequent checkpoints contain model, optimizer, and scheduler state.

## Held-out results

| Protein | Best RMSD | Best step | Final RMSD |
|---|---:|---:|---:|
| `crambin` | 10.00 Å | 100k | 11.15 Å |
| `cystatin-b` | 14.02 Å | 22k | 17.53 Å |
| `hbb` | **4.59 Å** | 26k | 6.19 Å |
| `interferon_gamma` | 19.01 Å | 16k | 21.65 Å |
| `profilin-1` | 14.92 Å | 74k | 16.26 Å |

- Best combined validation mean: **13.57 Å at step 62k**
- Final validation mean: **14.56 Å**
- Final validation median: **16.26 Å**

Per-protein best values occur at different steps and therefore do not describe one recoverable checkpoint.

## Final training-set evaluation

- Mean RMSD: **3.11 Å**
- Median RMSD: **2.85 Å**
- Range: **1.24–6.61 Å**
- Proteins below 3 Å: **30/56**

The final-step `0.18 Å` printed in the log is the active protein and sampled MSA batch, not the dataset-wide evaluation.

## Interpretation

Run C improved training fit but did not improve unrelated-fold transfer. Its combined validation result peaked early and degraded with continued optimization. `hbb` remains the strongest held-out result and is the likely homolog-transfer case; the other four held-out proteins remain poor.
