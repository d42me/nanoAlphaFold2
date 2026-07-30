"""Compare a trained checkpoint with full-MSA and query-only inputs."""

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
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("full_msa", "query_only"),
        default=("full_msa", "query_only"),
    )
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
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

    condition_sizes = {
        "full_msa": (cfg.n_clu, cfg.n_ext),
        "query_only": (1, 0),
    }
    checkpoint_hash = sha256(checkpoint_path)
    rows = []
    with torch.no_grad():
        for condition in args.conditions:
            n_clu, n_ext = condition_sizes[condition]
            for protein in proteins:
                batch = sample_batch(
                    dataset.features[protein],
                    n_clu=n_clu,
                    n_ext=n_ext,
                    mask_p=0.0,
                    seed=args.seed,
                )
                actual_msa_rows = batch["msa_feat"].shape[0]
                actual_extra_rows = batch["extra_msa_feat"].shape[0]
                batch = {key: value.to(device) for key, value in batch.items()}
                predicted_ca = model(batch, recycles=args.recycles)["ca"]
                target_ca = dataset.targets[protein]["CA"].to(device)
                rmsd = kabsch_rmsd(predicted_ca, target_ca).item()
                rows.append(
                    {
                        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                        "checkpoint_sha256": checkpoint_hash,
                        "step": checkpoint["step"],
                        "condition": condition,
                        "protein": protein,
                        "msa_rows": actual_msa_rows,
                        "extra_msa_rows": actual_extra_rows,
                        "recycles": args.recycles,
                        "seed": args.seed,
                        "ca_rmsd_angstrom": f"{rmsd:.4f}",
                    }
                )
                print(f"{condition:10s} {protein:18s} {rmsd:6.2f} Å", flush=True)

    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
