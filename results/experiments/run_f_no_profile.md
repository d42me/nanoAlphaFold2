# Run F — no extra MSA and no full-MSA profile

**Status: queued after Run E factorials**

## Question

Does always removing the precomputed full-MSA profile force nanoAlphaFold2 to learn useful information from actual non-query homolog rows?

Run E used 50% profile dropout and improved the best held-out mean from Run D's `13.37 Å` to `12.92 Å`, but its causal source remains unresolved and its `hbb` transfer weakened.

## Controlled change

Run F matches Run E except that profile dropout increases from `0.5` to `1.0`. Every training batch therefore receives the query one-hot in the profile channels while retaining up to 192 actual main-MSA rows.

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
| Profile dropout | **1.0 per training batch** |
| Evaluation | full profile, seed 42, no masking, three recycles |
| Metric | Kabsch-aligned Cα RMSD |

Command:

```bash
PYTHONPATH=/tmp/af2-eval-dc8de3f/src python scripts/train_multi.py \
  --tag ckpt_runF_no_profile \
  --big \
  --steps 250000 \
  --cosine \
  --min-lr 1e-4 \
  --exclude-file configs/splits/exclude_runC.txt \
  --no-extra-msa \
  --profile-dropout 1.0
```

Artifacts:

- Latest checkpoint: `checkpoints/ckpt_runF_no_profile.pt`
- Best-validation checkpoint: `checkpoints/ckpt_runF_no_profile_best.pt`
- Log: `logs/train_runF_no_profile.log`
- Scheduler session: `runE-ablations`

## Success criterion

At the best checkpoint, 192 cluster rows without profile must outperform sequence-only evaluation across the held-out set. Otherwise, the main MSA stack still has not learned useful homolog-row reasoning.
