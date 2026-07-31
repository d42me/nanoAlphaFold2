"""Test whether main-MSA effects depend on cross-position covariance."""

import argparse
import csv
import hashlib
from pathlib import Path

import torch

from af2_from_scratch import AF2Config, AlphaFold2FromScratch
from af2_from_scratch.dataset import ProteinDataset
from af2_from_scratch.geometry import kabsch_rmsd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = PROJECT_ROOT / "configs" / "splits"
CONDITIONS = ("query_only", "real_cluster", "row_permuted", "column_shuffled")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proteins-file", type=Path, default=SPLIT_DIR / "val.txt")
    parser.add_argument("--depth", type=int, default=192)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--recycles", type=int, default=3)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def covariance_batch(features, depth, seed, condition):
    sequences = features["msa_aatype"]
    deletions = features["deletion_matrix"]
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(sequences.shape[0] - 1, generator=generator) + 1
    cluster_rows = torch.cat([torch.tensor([0]), permutation[: depth - 1]])
    row_features = torch.cat(
        [sequences[cluster_rows], deletions[cluster_rows][..., None]], dim=-1
    )

    shuffle_generator = torch.Generator().manual_seed(seed + 10_000)
    if condition == "query_only":
        row_features = row_features[:1]
    elif condition == "row_permuted":
        order = torch.randperm(len(cluster_rows) - 1, generator=shuffle_generator) + 1
        row_features = torch.cat([row_features[:1], row_features[order]])
    elif condition == "column_shuffled":
        shuffled = row_features.clone()
        for column in range(row_features.shape[1]):
            order = torch.randperm(len(cluster_rows) - 1, generator=shuffle_generator) + 1
            shuffled[1:, column] = row_features[order, column]
        row_features = shuffled

    query_profile = sequences[0].expand(len(row_features), -1, -1)
    msa_feat = torch.cat([row_features, query_profile], dim=-1)
    return {
        "msa_feat": msa_feat,
        "extra_msa_feat": torch.empty(0, sequences.shape[1], 23),
        "target_feat": features["target_feat"],
        "residue_index": features["residue_index"],
    }


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
        for condition in CONDITIONS:
            for seed in args.seeds:
                for protein in proteins:
                    batch = covariance_batch(
                        dataset.features[protein], args.depth, seed, condition
                    )
                    actual_msa_rows = batch["msa_feat"].shape[0]
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
                            "depth": args.depth,
                            "condition": condition,
                            "protein": protein,
                            "msa_rows": actual_msa_rows,
                            "extra_msa_rows": 0,
                            "recycles": args.recycles,
                            "seed": seed,
                            "ca_rmsd_angstrom": f"{rmsd:.4f}",
                        }
                    )
                    print(
                        f"{condition:15s} seed={seed} {protein:18s} {rmsd:6.2f} Å",
                        flush=True,
                    )

    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
