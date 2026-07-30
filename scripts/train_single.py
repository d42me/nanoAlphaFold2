"""Train nanoAlphaFold2 on the bundled tautomerase example.

Run from any directory with:

    python scripts/train_single.py

The student receives only the MSA and learns to reproduce the bundled teacher
structure. Checkpoints are written to ``checkpoints/`` and ignored by Git.
"""

from pathlib import Path
import time

import torch
import torch.nn.functional as F

from af2_from_scratch import AF2Config, AlphaFold2FromScratch
from af2_from_scratch.feature_extraction import msa_features, sample_batch
from af2_from_scratch.geometry import kabsch_rmsd
from af2_from_scratch.losses import distogram_target, fape_ca, lddt_target
from af2_from_scratch.teacher import load_teacher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = PROJECT_ROOT / "examples" / "tautomerase"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def main():
    cfg = AF2Config()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    features = msa_features(EXAMPLE_DIR / "alignment.a3m")
    targets = {
        name: value.to(device)
        for name, value in load_teacher(
            EXAMPLE_DIR / "teacher.cif",
            n_res=features["msa_aatype"].shape[1],
        ).items()
    }
    distogram = distogram_target(targets["CA"])
    print(
        f"teacher CA span: {targets['CA'].max(0).values - targets['CA'].min(0).values} A",
        flush=True,
    )

    model = AlphaFold2FromScratch(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(
        f"student params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
        flush=True,
    )

    started_at = time.time()
    for step in range(1, cfg.steps + 1):
        batch = {
            name: value.to(device)
            for name, value in sample_batch(
                features, cfg.n_clu, cfg.n_ext, cfg.mask_p
            ).items()
        }
        output = model(batch)
        fape_loss = fape_ca(output["T"], targets["frames"], output["ca"], targets["CA"])
        distogram_loss = F.cross_entropy(
            output["disto_logits"].flatten(0, 1), distogram.flatten()
        )
        confidence_loss = F.cross_entropy(
            output["plddt_logits"], lddt_target(output["ca"], targets["CA"])
        )
        loss = fape_loss + 0.3 * distogram_loss + 0.01 * confidence_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            rmsd = kabsch_rmsd(output["ca"].detach(), targets["CA"])
            print(
                f"step {step:6d} | loss {loss.item():7.3f} | "
                f"fape {fape_loss.item():6.3f} | "
                f"disto {distogram_loss.item():5.3f} | "
                f"conf {confidence_loss.item():5.3f} | "
                f"CA-RMSD {rmsd.item():5.2f} A | "
                f"{50 / (time.time() - started_at):.1f} it/s",
                flush=True,
            )
            started_at = time.time()

        if step % 1000 == 0:
            torch.save(
                {"cfg": cfg.__dict__, "model": model.state_dict()},
                CHECKPOINT_DIR / f"single_{step}.pt",
            )

    torch.save(
        {"cfg": cfg.__dict__, "model": model.state_dict()},
        CHECKPOINT_DIR / "single_final.pt",
    )
    print("done.", flush=True)


if __name__ == "__main__":
    main()
