"""Measure held-out structure quality as MSA depth increases."""

import argparse
import csv
import hashlib
from pathlib import Path

import torch

from af2_from_scratch import AF2Config, AlphaFold2FromScratch
from af2_from_scratch.dataset import ProteinDataset
from af2_from_scratch.feature_extraction import sample_batch
from af2_from_scratch.geometry import kabsch_rmsd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = PROJECT_ROOT / "configs" / "splits"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proteins-file", type=Path, default=SPLIT_DIR / "val.txt")
    parser.add_argument("--depths", nargs="+", type=int, default=(1, 8, 32, 64, 192))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--recycles", type=int, default=3)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = AF2Config(**checkpoint["cfg"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AlphaFold2FromScratch(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    requested_proteins = set(args.proteins_file.read_text().split())
    dataset = ProteinDataset(DATA_DIR)
    proteins = [name for name in dataset.names if name in requested_proteins]
    missing = requested_proteins.difference(proteins)
    if missing:
        raise ValueError(f"proteins missing from dataset: {sorted(missing)}")

    checkpoint_hash = sha256(checkpoint_path)
    rows = []
    with torch.no_grad():
        for depth in args.depths:
            n_ext = 0 if depth == 1 else depth
            for seed in args.seeds:
                for protein in proteins:
                    batch = sample_batch(
                        dataset.features[protein],
                        n_clu=depth,
                        n_ext=n_ext,
                        mask_p=0.0,
                        seed=seed,
                    )
                    actual_msa_rows = batch["msa_feat"].shape[0]
                    actual_extra_rows = batch["extra_msa_feat"].shape[0]
                    batch = {key: value.to(device) for key, value in batch.items()}
                    predicted_ca = model(batch, recycles=args.recycles)["ca"]
                    target_ca = dataset.targets[protein]["CA"].to(device)
                    rmsd = kabsch_rmsd(predicted_ca, target_ca).item()
                    rows.append(
                        {
                            "checkpoint": str(
                                checkpoint_path.relative_to(PROJECT_ROOT)
                            ),
                            "checkpoint_sha256": checkpoint_hash,
                            "step": checkpoint["step"],
                            "requested_depth": depth,
                            "protein": protein,
                            "msa_rows": actual_msa_rows,
                            "extra_msa_rows": actual_extra_rows,
                            "recycles": args.recycles,
                            "seed": seed,
                            "ca_rmsd_angstrom": f"{rmsd:.4f}",
                        }
                    )
                    print(
                        f"depth={depth:3d} seed={seed} {protein:18s} {rmsd:6.2f} Å",
                        flush=True,
                    )

    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
