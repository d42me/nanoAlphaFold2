"""Training signals for AlphaFold 2 from Scratch, distilled from a teacher:

  1. FAPE on CA      - the iconic AF2 loss: compare positions in LOCAL frames (frame-invariant)
  2. distogram CE    - pair repr must predict the binned pairwise CA distances
  3. pLDDT CE        - per-residue confidence, trained AF2-style against the model's own lDDT-CA
Weights follow AF2: FAPE 1.0, distogram 0.3, pLDDT 0.01.
"""

import torch
from .geometry import apply, invert


def distogram_target(ca, n_bins=64, lo=2.0, hi=22.0):
    """True CA-CA distances -> bin indices (R, R) in [0, 63]. AF2 uses 64 bins over 2-22 Angstrom."""
    d = torch.cdist(ca, ca)
    return torch.bucketize(d, torch.linspace(lo, hi, n_bins - 1, device=ca.device))


def lddt_target(ca_pred, ca_true, n_bins=50):
    """How AF2 really supervises pLDDT: per-residue lDDT-CA between (detached) prediction and teacher.
    lDDT = fraction of CA pairs within 0.5/1/2/4 Angstrom after Kabsch alignment."""
    with torch.no_grad():
        p = ca_pred - ca_pred.mean(0)
        q = ca_true - ca_true.mean(0)  # center
        U, _, Vh = torch.linalg.svd(p.T @ q)
        d = torch.sign(torch.linalg.det(U @ Vh))
        R = (
            U @ torch.diag(torch.tensor([1.0, 1.0, d], device=p.device)) @ Vh
        )  # optimal rotation
        err = torch.cdist(p @ R, q)  # (R, R) aligned pairwise errors
        lddt = (
            torch.stack([err < t for t in (0.5, 1.0, 2.0, 4.0)]).float().mean(0)
        )  # lDDT-CA per pair
        per_res = lddt.mean(-1)  # per-residue score in [0, 1]
        return (per_res * n_bins).long().clamp(0, n_bins - 1)


def fape_ca(T_pred, T_true, ca_pred, ca_true, clamp=10.0):
    """FAPE restricted to CA: read every CA j in every residue i's LOCAL frame and compare pred vs teacher.
    In local frames, global rotation/translation cancels - that's what makes FAPE frame-invariant."""
    lp = apply(
        invert(T_pred).unsqueeze(1), ca_pred
    )  # (R_i, R_j, 3): all predicted CAs in each predicted frame
    lt = apply(
        invert(T_true).unsqueeze(1), ca_true
    )  # (R_i, R_j, 3): same for the teacher
    return (
        (lp - lt).norm(dim=-1).clamp(max=clamp).mean()
    )  # clamp limits outlier influence (AF2: 10 Angstrom)
