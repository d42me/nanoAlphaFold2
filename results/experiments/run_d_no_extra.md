# Run D — no extra-MSA tower

**Status: complete**

## Question

Can the main cluster-MSA path learn useful evolutionary transfer when the shortcut-like extra-MSA path is removed during training?

Run C evaluation showed that:

- non-query cluster rows had negligible influence;
- full-MSA profile channels had negligible influence;
- nearly all MSA benefit and harm entered through the extra-MSA tower;
- most extra-tower benefit survived destruction of cross-position covariance.

## Controlled change

Run D matches Run C except for one architectural/data-path ablation:

```text
Run C: n_extra = 1, n_ext = 192
Run D: n_extra = 0, n_ext = 0
```

The main cluster MSA remains at `n_clu = 192`.

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
| Evaluation | seed 42, no masking, three recycles |
| Metric | Kabsch-aligned Cα RMSD |

Command:

```bash
PYTHONPATH=/tmp/af2-eval-dc8de3f/src python scripts/train_multi.py \
  --tag ckpt_runD_no_extra \
  --big \
  --steps 250000 \
  --cosine \
  --min-lr 1e-4 \
  --exclude-file configs/splits/exclude_runC.txt \
  --no-extra-msa
```

Artifacts:

- Final checkpoint: `checkpoints/ckpt_runD_no_extra.pt`
- Step-72k snapshot: `checkpoints/ckpt_runD_no_extra_step72000.pt`
- Log: `logs/train_runD_no_extra.log`
- Follow-up: [`run_d_cluster_paths.md`](run_d_cluster_paths.md)

## Final training results

- Best combined validation mean: **13.37 Å at step 84k**
- Final validation mean: **13.70 Å**
- Final validation median: **14.23 Å**
- Final training mean: **2.99 Å**
- Final training median: **2.38 Å**
- Training proteins below 3 Å: **37/56**

| Protein | Best RMSD | Best step | Final RMSD |
|---|---:|---:|---:|
| `crambin` | 12.76 Å | 38k | 14.44 Å |
| `cystatin-b` | 12.43 Å | 64k | 13.03 Å |
| `hbb` | **5.67 Å** | 100k | 6.23 Å |
| `interferon_gamma` | 18.71 Å | 12k | 20.55 Å |
| `profilin-1` | 14.11 Å | 180k | 14.23 Å |

Validation peaked well before training ended while training fit continued to improve, again indicating overfitting.

## Mechanistic outcome

The post-run factorial shows that removing the extra tower redirected evolutionary learning mainly into the full-MSA **profile channels**, not robust use of non-query cluster rows. Profile-only improves the mean from `14.52 Å` sequence-only to `13.50 Å`; adding 192 cluster rows to the profile worsens it slightly to `13.69 Å`. See [`run_d_cluster_paths.md`](run_d_cluster_paths.md).
