"""Stage 6 — full model: orchestrate embedding, recycling, folding, and heads.

Pipeline::

    feature batch ─► InputEmbedder ─► m, z, e
                                      │
                         ┌─ recycle + Evoformer ◄─┐
                         └────────────────────────┘
                                      │
                       m[0] ─► single representation s
                                      │
                              StructureModule
                                      │
                frames + Cα positions + distance/confidence logits

``AlphaFold2FromScratch`` owns orchestration only; each stage remains in its
focused module. Outputs include residue frames, Cα positions, a distogram,
and per-residue pLDDT logits.
"""

import torch.nn as nn
from .feature_embedding import InputEmbedder, RecyclingEmbedder
from .evoformer import Evoformer
from .structure_module import StructureModule


class AlphaFold2FromScratch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = InputEmbedder(cfg)
        self.rec = RecyclingEmbedder(cfg)
        self.evo = Evoformer(cfg)
        self.sm = StructureModule(cfg)
        self.to_s = nn.Linear(
            cfg.c_m, cfg.c_s
        )  # MSA row 0 (c_m) -> single repr (c_s) for the structure module
        self.disto = nn.Linear(
            cfg.c_z, 64
        )  # pair repr -> 64 distance bins (distogram head)
        self.plddt = nn.Linear(
            cfg.c_s, 50
        )  # single repr -> 50 confidence bins (pLDDT head)

    def forward(self, batch, recycles=None):
        recycles = self.cfg.recycles if recycles is None else recycles
        m, z, e = self.emb(batch)
        m_prev = z_prev = None  # recycle 0 has no predecessor
        for _ in range(recycles + 1):
            m, z = self.rec(
                m, z, m_prev, z_prev
            )  # re-inject previous outputs (LayerNormed)
            m, z = self.evo(m, z, e)  # the trunk
            m_prev, z_prev = (
                m[0].detach(),
                z.detach(),
            )  # AF2 rule: no gradients through recycled inputs
        s = self.to_s(m[0])  # single repr = projected MSA row 0 = the query sequence
        T, s = self.sm(s, z)  # structure module -> per-residue frames + refined s
        zsym = z + z.transpose(0, 1)  # symmetrize: distance i->j must equal j->i
        return {
            "T": T,
            "ca": T[..., :3, 3],  # frames; CA = frame origins
            "disto_logits": self.disto(zsym),  # (R, R, 64)
            "plddt_logits": self.plddt(s),
        }  # (R, 50)
