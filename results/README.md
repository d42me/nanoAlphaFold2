# Experiment results

This directory is the canonical record of experimental outcomes.

- [`summary.md`](summary.md) — current conclusions and cross-run comparison
- [`experiments/run_c_training.md`](experiments/run_c_training.md) — Run C configuration and training outcome
- [`experiments/run_c_msa_ablation.md`](experiments/run_c_msa_ablation.md) — full-MSA versus query-only result
- [`experiments/run_c_msa_depth.md`](experiments/run_c_msa_depth.md) — five-depth, three-seed dose response
- [`experiments/run_c_msa_paths.md`](experiments/run_c_msa_paths.md) — cluster versus extra-MSA path contribution
- [`experiments/run_c_profile_paths.md`](experiments/run_c_profile_paths.md) — strict sequence/profile/extra factorial
- [`experiments/run_c_extra_covariance.md`](experiments/run_c_extra_covariance.md) — covariance-preserving and covariance-destroying controls
- [`experiments/run_d_no_extra.md`](experiments/run_d_no_extra.md) — completed no-extra-MSA training ablation
- [`experiments/run_d_cluster_paths.md`](experiments/run_d_cluster_paths.md) — post-training profile and cluster-row factorial
- [`experiments/run_e_profile_dropout.md`](experiments/run_e_profile_dropout.md) — completed 50% profile-dropout training ablation
- [`experiments/run_e_cluster_paths.md`](experiments/run_e_cluster_paths.md) — positive main-MSA homolog-row factorial
- [`experiments/run_f_no_profile.md`](experiments/run_f_no_profile.md) — active 100% profile-dropout training ablation
- `metrics/*.csv` — machine-readable per-run and per-protein measurements

Large artifacts remain outside this directory:

- `checkpoints/` — model and optimizer states
- `logs/` — raw training output
- `data/` — teacher structures and alignments

Every experiment record must identify its checkpoint, split files, evaluation seed, recycle count, metric, and exact command. Report both final and best validation values when available; a best value without its step is ambiguous.

## Metric convention

Unless an experiment says otherwise, structure quality is Cα RMSD in ångströms after Kabsch alignment. Lower is better. Held-out evaluation uses `model.eval()`, no MSA masking, seed 42, and three recycles.
