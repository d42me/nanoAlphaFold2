# Generated training data

This directory is intentionally ignored by Git because A3M alignments and teacher CIF files quickly become large.

Populate the starter dataset with:

```bash
python scripts/fetch_data.py
```

Each protein uses this layout:

```text
data/<protein>/
├── <protein>.a3m
├── seq.fasta
└── teacher.cif
```

The tracked train/validation manifests live in `../configs/splits/`.
