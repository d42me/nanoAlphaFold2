# Run E — no extra MSA with stochastic profile dropout

**Status: active**

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

## Success criterion

After training, strict factorial evaluation must show that non-query cluster rows materially improve held-out predictions when profile channels are absent. A better profile-enabled score alone would indicate another profile shortcut rather than successful row-level evolutionary reasoning.
