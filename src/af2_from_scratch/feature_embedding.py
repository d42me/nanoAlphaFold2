"""Stage 2 — feature embedding: create the model's initial representations.

Pipeline::

    msa_feat ───────────────► m  (S, R, c_m)  MSA representation
    target_feat + relpos ───► z  (R, R, c_z)  pair representation
    extra_msa_feat ─────────► e  (E, R, c_e)  extra-MSA representation
    previous m[0] and z ────► recycled additions to m and z

This stage is linear projections plus broadcasting; attention begins in the
Evoformer. ``InputEmbedder`` creates ``m``, ``z``, and ``e`` while
``RecyclingEmbedder`` adds information from the preceding recycle. Full
AlphaFold also recycles pseudo-beta distance features; this compact version
recycles only the previous query row and pair representation.
"""

import torch.nn as nn
import torch.nn.functional as F


class InputEmbedder(nn.Module):
    """Algorithms 3+4 of the AF2 supplement, compact."""

    def __init__(self, cfg, msa_dim=45, extra_dim=23, tf_dim=21, vbins=32):
        super().__init__()
        self.vbins = vbins
        self.tf_i = nn.Linear(
            tf_dim, cfg.c_z
        )  # query embedding, will become the "row" copy in z
        self.tf_j = nn.Linear(
            tf_dim, cfg.c_z
        )  # query embedding, will become the "column" copy in z
        self.tf_m = nn.Linear(
            tf_dim, cfg.c_m
        )  # query embedding, added to every MSA row
        self.msa = nn.Linear(msa_dim, cfg.c_m)  # raw per-row MSA features (45) -> c_m
        self.ext = nn.Linear(extra_dim, cfg.c_e)  # raw extra-MSA features (23) -> c_e
        self.rel = nn.Linear(
            2 * vbins + 1, cfg.c_z
        )  # one-hot relative position -> pair channels

    def relpos(self, idx):
        """Alg 4: sequence distance d_ij = clamp(i-j) one-hotted -> linear. Tells z where residues sit relative to each other."""
        d = (idx[:, None] - idx[None, :]).clamp(
            -self.vbins, self.vbins
        ) + self.vbins  # outer difference via broadcasting
        return self.rel(F.one_hot(d.long(), 2 * self.vbins + 1).float())  # (R, R, c_z)

    def forward(self, b):
        t = b["target_feat"]  # (R, 21) one-hot query
        z = (
            self.tf_i(t)[:, None] + self.tf_j(t)[None]
        )  # THE outer sum: (R,1,c)+(1,R,c) -> (R,R,c)
        z = z + self.relpos(b["residue_index"])  # add relative-position information
        m = (
            self.msa(b["msa_feat"]) + self.tf_m(t)[None]
        )  # (S,R,c_m): same query embedding on every row
        e = self.ext(
            b["extra_msa_feat"]
        )  # (E,R,c_e): extra MSA is embedded separately, cheaper
        return m, z, e


class RecyclingEmbedder(nn.Module):
    """Add normalized query-row and pair outputs from the previous recycle.

    Gradient detachment is handled by ``AlphaFold2FromScratch.forward``. Full
    AlphaFold also adds previous pseudo-beta distance features to ``z``.
    """

    def __init__(self, cfg):
        super().__init__()
        self.nm = nn.LayerNorm(
            cfg.c_m
        )  # prev MSA row-0 (= single repr) re-enters via the MSA channels
        self.nz = nn.LayerNorm(cfg.c_z)  # prev pair repr re-enters the pair channels

    def forward(self, m, z, m_prev, z_prev):
        if m_prev is not None:
            m = m.clone()
            m[0] = m[0] + self.nm(m_prev)
        if z_prev is not None:
            z = z + self.nz(z_prev)
        return m, z
