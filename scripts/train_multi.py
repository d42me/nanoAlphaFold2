"""Train AlphaFold 2 from Scratch across proteins and evaluate held-out folds.

Run after populating ``data/`` with ``scripts/fetch_data.py``. Split manifests
live in ``configs/splits/`` so every experiment uses an explicit, tracked split.
"""

import argparse
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from af2_from_scratch import AF2Config, AlphaFold2FromScratch
from af2_from_scratch.dataset import ProteinDataset
from af2_from_scratch.feature_extraction import sample_batch
from af2_from_scratch.geometry import kabsch_rmsd
from af2_from_scratch.losses import distogram_target, fape_ca, lddt_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = PROJECT_ROOT / "configs" / "splits"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_VALIDATION = {"crambin", "hbb"}
DEFAULT_EXCLUSIONS = {"rps27a"}


def read_names(path, fallback):
    return set(path.read_text().split()) if path.exists() else fallback


def evaluate(model, dataset, targets, cfg, device):
    """Return fixed-seed C-alpha RMSD for every loaded protein."""
    model.eval()
    scores = {}
    with torch.no_grad():
        for name in dataset.names:
            batch = {
                key: value.to(device)
                for key, value in sample_batch(
                    dataset.features[name],
                    cfg.n_clu,
                    cfg.n_ext,
                    mask_p=0.0,
                    seed=42,
                ).items()
            }
            scores[name] = kabsch_rmsd(
                model(batch, recycles=3)["ca"], targets[name]["CA"]
            ).item()
    model.train()
    return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="multi", help="checkpoint name suffix")
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument(
        "--big", action="store_true", help="use the wider/deeper capacity ablation"
    )
    parser.add_argument(
        "--cosine", action="store_true", help="apply cosine learning-rate decay"
    )
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=SPLIT_DIR / "exclude.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = AF2Config(steps=args.steps, n_clu=192, n_ext=192)
    if args.big:
        cfg.c_m, cfg.c_z, cfg.c_e, cfg.c_s = 96, 128, 48, 192
        cfg.n_evo, cfg.n_ipa = 8, 3

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validation_names = read_names(SPLIT_DIR / "val.txt", DEFAULT_VALIDATION)
    exclusions = read_names(args.exclude_file, DEFAULT_EXCLUSIONS)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("loading dataset:", flush=True)
    dataset = ProteinDataset(DATA_DIR)
    include_path = SPLIT_DIR / "include.txt"
    if include_path.exists():
        included = read_names(include_path, set(dataset.names))
        dataset.names = [name for name in dataset.names if name in included]
    dataset.names = [name for name in dataset.names if name not in exclusions]

    train_names = [name for name in dataset.names if name not in validation_names]
    val_names = [name for name in dataset.names if name in validation_names]
    print(
        f"train ({len(train_names)}): {train_names}\n"
        f"val:   {val_names}  excluded: {sorted(exclusions)}",
        flush=True,
    )

    targets = {
        name: {key: value.to(device) for key, value in dataset.targets[name].items()}
        for name in dataset.names
    }
    distograms = {name: distogram_target(targets[name]["CA"]) for name in dataset.names}

    model = AlphaFold2FromScratch(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.steps, eta_min=args.min_lr
        )
        if args.cosine
        else None
    )
    print(
        f"student params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
        flush=True,
    )

    started_at = time.time()
    for step in range(1, cfg.steps + 1):
        name = random.choice(train_names)
        batch = {
            key: value.to(device)
            for key, value in sample_batch(
                dataset.features[name], cfg.n_clu, cfg.n_ext, cfg.mask_p
            ).items()
        }
        output = model(batch)
        target = targets[name]
        fape_loss = fape_ca(output["T"], target["frames"], output["ca"], target["CA"])
        distogram_loss = F.cross_entropy(
            output["disto_logits"].flatten(0, 1), distograms[name].flatten()
        )
        confidence_loss = F.cross_entropy(
            output["plddt_logits"], lddt_target(output["ca"], target["CA"])
        )
        loss = fape_loss + 0.3 * distogram_loss + 0.01 * confidence_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            rmsd = kabsch_rmsd(output["ca"].detach(), target["CA"])
            print(
                f"step {step:7d} | {name:13s} | loss {loss.item():6.3f} | "
                f"fape {fape_loss.item():6.3f} | RMSD {rmsd.item():5.2f} A | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                f"{100 / (time.time() - started_at):.1f} it/s",
                flush=True,
            )
            started_at = time.time()

        if step % 2000 == 0:
            scores = evaluate(model, dataset, targets, cfg, device)
            train_scores = [
                f"{name}:{score:.2f}"
                for name, score in scores.items()
                if name not in validation_names
            ]
            val_scores = [
                f"{name}:{score:.2f}"
                for name, score in scores.items()
                if name in validation_names
            ]
            print(f"  >> TRAIN {train_scores}\n  >> VAL   {val_scores}", flush=True)
            torch.save(
                {"cfg": cfg.__dict__, "model": model.state_dict(), "step": step},
                CHECKPOINT_DIR / f"{args.tag}.pt",
            )

    print("done.", flush=True)


if __name__ == "__main__":
    main()
