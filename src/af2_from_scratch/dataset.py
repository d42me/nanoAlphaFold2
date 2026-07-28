"""Multi-protein dataset for AlphaFold 2 from Scratch experiments.

Each protein directory provides an A3M alignment and teacher CIF. Parsed MSA
features and teacher targets are cached in memory after validating residue count.
"""

from pathlib import Path

from .feature_extraction import msa_features
from .teacher import load_teacher


class ProteinDataset:
    def __init__(self, root="data"):
        self.names = []
        self.features = {}
        self.targets = {}

        for protein_dir in sorted(Path(root).iterdir()):
            if not protein_dir.is_dir():
                continue
            name = protein_dir.name
            alignment_path = protein_dir / f"{name}.a3m"
            teacher_path = protein_dir / "teacher.cif"
            if not alignment_path.exists() or not teacher_path.exists():
                continue

            features = msa_features(alignment_path)
            n_res = features["msa_aatype"].shape[1]
            targets = load_teacher(teacher_path, n_res=n_res)
            self.names.append(name)
            self.features[name] = features
            self.targets[name] = targets
            n_seq = features["msa_aatype"].shape[0]
            print(
                f"  {name:13s} {n_res:4d} residues, {n_seq:6d} MSA sequences",
                flush=True,
            )
