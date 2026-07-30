"""Training signals for nanoAlphaFold2, distilled from a teacher:

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


def lddt_ca(ca_pred, ca_true, cutoff=15.0):
    """Per-residue lDDT-Cα from errors in corresponding pairwise distances."""
    predicted_distances = torch.cdist(ca_pred, ca_pred)
    true_distances = torch.cdist(ca_true, ca_true)
    distance_errors = (predicted_distances - true_distances).abs()
    not_self = ~torch.eye(
        ca_true.shape[0],
        dtype=torch.bool,
        device=ca_true.device,
    )
    included = (true_distances < cutoff) & not_self
    thresholds = ca_true.new_tensor([0.5, 1.0, 2.0, 4.0])
    pair_scores = (
        (distance_errors[..., None] < thresholds).to(distance_errors.dtype).mean(-1)
    )
    score_sum = (pair_scores * included).sum(-1)
    return score_sum / included.sum(-1).clamp_min(1)


def lddt_target(ca_pred, ca_true, n_bins=50):
    """Quantize detached per-residue lDDT-Cα scores for the pLDDT head."""
    with torch.no_grad():
        scores = lddt_ca(ca_pred, ca_true)
        return (scores * n_bins).long().clamp(0, n_bins - 1)


def fape_ca(
    T_pred,
    T_true,
    ca_pred,
    ca_true,
    clamp=10.0,
    length_scale=10.0,
):
    """Normalized Cα FAPE after expressing every point in every local frame."""
    lp = apply(
        invert(T_pred).unsqueeze(1), ca_pred
    )  # (R_i, R_j, 3): all predicted CAs in each predicted frame
    lt = apply(
        invert(T_true).unsqueeze(1), ca_true
    )  # (R_i, R_j, 3): same for the teacher
    errors = (lp - lt).norm(dim=-1).clamp(max=clamp)
    return errors.mean() / length_scale
