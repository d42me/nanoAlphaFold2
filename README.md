# nanoAlphaFold2

A compact, educational implementation of the major AlphaFold 2 ideas in PyTorch. Each conceptual stage has one numbered notebook and one focused Python module, with plain-language explanations followed by the precise architecture.

The implementation includes gated MSA attention, global extra-MSA column attention, shared row/column dropout, recycling, invariant point attention, local-frame geometry, FAPE, distograms, and pLDDT supervision.

> This is a teaching implementation, not a production protein-structure predictor or a drop-in replacement for AlphaFold 2. It predicts a Cα backbone trace rather than an all-atom structure.

## Learn the architecture in order

| Step | Notebook | Implementation | Purpose |
|---:|---|---|---|
| 0 | [`00_pipeline_overview.ipynb`](notebooks/00_pipeline_overview.ipynb) | — | End-to-end map and tensor glossary |
| 1 | [`01_feature_extraction.ipynb`](notebooks/01_feature_extraction.ipynb) | [`feature_extraction.py`](src/af2_from_scratch/feature_extraction.py) | A3M alignment → numerical features |
| 2 | [`02_feature_embedding.ipynb`](notebooks/02_feature_embedding.ipynb) | [`feature_embedding.py`](src/af2_from_scratch/feature_embedding.py) | Features → MSA, pair, and extra-MSA representations |
| 3 | [`03_evoformer.ipynb`](notebooks/03_evoformer.ipynb) | [`evoformer.py`](src/af2_from_scratch/evoformer.py) | Exchange information between MSA and residue pairs |
| 4 | [`04_geometry.ipynb`](notebooks/04_geometry.ipynb) | [`geometry.py`](src/af2_from_scratch/geometry.py) | Rotations, translations, frames, and alignment |
| 5 | [`05_structure_module.ipynb`](notebooks/05_structure_module.ipynb) | [`structure_module.py`](src/af2_from_scratch/structure_module.py) | Representations → residue positions and orientations |
| 6 | [`06_full_model.ipynb`](notebooks/06_full_model.ipynb) | [`model.py`](src/af2_from_scratch/model.py) | Recycling, structure prediction, and output heads |
| 7 | [`07_single_protein_training.ipynb`](notebooks/07_single_protein_training.ipynb) | [`train_single.py`](scripts/train_single.py) | Distill one teacher structure |
| 8 | [`08_multi_protein_training.ipynb`](notebooks/08_multi_protein_training.ipynb) | [`train_multi.py`](scripts/train_multi.py) | Test generalization on held-out proteins |

## Architecture

```text
A3M alignment
     |
     v
+------------------------+
| 1. Feature extraction  | --> msa_feat, extra_msa_feat,
| feature_extraction.py  |     target_feat, residue_index
+-----------+------------+
            |
            v
+------------------------+
| 2. Feature embedding   | --> m: MSA representation
| feature_embedding.py   |     z: pair representation
+-----------+------------+     e: extra-MSA representation
            |
            v
+------------------------+
| 3. Evoformer           | --> refined m and z
| evoformer.py           |
+-----------+------------+
            |
            v
+------------------------+     geometry.py supplies rotations,
| 5. Structure module    | <--- transforms, and local frames
| structure_module.py    |
+-----------+------------+
            |
            v
C-alpha coordinates + distance and confidence predictions
```

Geometry is a mathematical dependency of the structure module, teacher loader, and geometric losses; it is not a separate neural-network stage.

## Repository structure

```text
nanoAlphaFold2/
├── src/af2_from_scratch/       # importable implementation
│   ├── config.py               # all educational model/training knobs
│   ├── feature_extraction.py   # stage 1
│   ├── feature_embedding.py    # stage 2
│   ├── evoformer.py            # stage 3
│   ├── geometry.py             # stage 4
│   ├── structure_module.py     # stage 5
│   ├── model.py                # stage 6 orchestration
│   ├── teacher.py              # CIF teacher loading
│   ├── losses.py               # FAPE, distogram, and pLDDT targets
│   └── dataset.py              # multi-protein dataset
├── notebooks/                  # numbered learning path
├── scripts/                    # data and training entry points
├── examples/tautomerase/       # small tracked example
├── configs/splits/             # tracked experiment splits
├── tests/                      # fast architecture regression tests
├── results/                    # committed summaries and machine-readable metrics
├── data/                       # generated data; ignored
├── checkpoints/                # generated weights; ignored
├── logs/                       # generated logs; ignored
├── pyproject.toml              # package metadata and dependency extras
└── uv.lock                     # reproducible dependency resolution
```

## Installation with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once, then clone and synchronize the project environment:

```bash
git clone https://github.com/d42me/nanoAlphaFold2.git
cd nanoAlphaFold2
uv sync --all-extras
```

`uv sync` creates `.venv/` and installs the locked Python dependencies. `--all-extras` includes notebooks, data fetching, tests, and Ruff. The stable Python import remains `af2_from_scratch` so existing notebooks and checkpoints keep working.

For only the core PyTorch package:

```bash
uv sync
```

To add selected capabilities instead:

```bash
uv sync --extra notebooks --extra data
```

## Run the notebooks

```bash
uv run jupyter lab notebooks/
```

Start with `00_pipeline_overview.ipynb`, then follow the numerical order. Every architecture notebook contains an ASCII data-flow diagram and points to the corresponding module under `src/af2_from_scratch/`.

## Train the example

The bundled tautomerase example is sufficient for the single-protein walkthrough:

```bash
uv run python scripts/train_single.py 2>&1 | tee logs/train_single.log
```

For multi-protein training (`--all-extras` already includes the data dependency):

```bash
uv run python scripts/fetch_data.py
uv run python scripts/train_multi.py 2>&1 | tee logs/train_multi.log
```

Resume a multi-protein run from a checkpoint with:

```bash
uv run python scripts/train_multi.py --resume checkpoints/multi.pt
```

Large alignments, checkpoints, and logs stay local through directory-specific `.gitignore` files.

## Experiment results

Committed conclusions and metrics live in [`results/`](results/). Start with [`results/summary.md`](results/summary.md); each detailed experiment record links its exact checkpoint, source revision, split, command, and raw log. `scripts/evaluate_msa_ablation.py` writes reproducible per-protein full-MSA and query-only measurements to CSV.

## Development

Install all development dependencies once:

```bash
uv sync --all-extras
```

Format, lint, and test through the locked environment:

```bash
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Ruff formats both Python files and notebook code cells; Markdown cells remain unchanged. The regression suite covers tensor shapes and key architectural behavior such as recycling, attention gating, shared dropout, IPA invariance, lDDT-Cα, and FAPE normalization.

## Tensor glossary

```text
S = selected MSA sequences      E = extra MSA sequences
R = protein residues            c_* = learned channel width

m  : (S, R, c_m)      MSA representation
z  : (R, R, c_z)      residue-pair representation
e  : (E, R, c_e)      extra-MSA representation
s  : (R, c_s)         one vector per query residue
T  : (R, 4, 4)        residue position and orientation
ca : (R, 3)           predicted C-alpha coordinates
```

## Deliberate simplifications

- Fewer and narrower Evoformer blocks and fewer structure-module iterations
- Random MSA sampling instead of AlphaFold's complete clustering and feature pipeline
- Cα backbone prediction without torsion angles, side chains, or all-atom reconstruction
- Teacher-coordinate distillation instead of AlphaFold's complete training-data pipeline and loss suite
- No template stack, multimer handling, relaxation, or production confidence suite

These choices keep the core ideas visible and make small experiments practical on a single GPU. Checkpoints created before an architecture change are not guaranteed to remain compatible.
