"""Stage 3 — Evoformer: repeatedly exchange MSA and residue-pair information.

One block::

    m ─► row attention (+ z bias) ─► column attention ─► transition
    │                                                       │
    └──────────────── outer-product mean ───────────────────► z
                                                            │
       triangle multiplication ×2 ─► triangle attention ×2 ─► transition

The extra-MSA tower runs first and refines ``z``; the main blocks then refine
``m`` and ``z`` together. This compact version keeps every major operation type
while shrinking channel widths and block counts configured in ``config.py``.
"""

import torch
import torch.nn as nn


class MHA(nn.Module):
    """Multi-head attention over dim -2 of x, with optional additive score bias of shape (h, q, k)."""

    def __init__(self, c_in, c=32, heads=8):
        super().__init__()
        self.h, self.c = heads, c
        self.qkv = nn.Linear(
            c_in, 3 * heads * c, bias=False
        )  # one fused projection for q, k, v
        self.out = nn.Linear(heads * c, c_in)

    def forward(self, x, bias=None):
        q, k, v = self.qkv(x).chunk(3, -1)  # 3x (..., N, h*c)

        def split_heads(tensor):
            return tensor.unflatten(-1, (self.h, self.c))

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        a = (
            torch.einsum("...qhc,...khc->...hqk", q, k) * self.c**-0.5
        )  # scores: contract the channel dim
        if bias is not None:
            a = a + bias  # pair bias enters HERE (broadcast over batch dims)
        o = torch.einsum(
            "...hqk,...khc->...qhc", a.softmax(-1), v
        )  # softmax over keys, then weight the values
        return self.out(o.flatten(-2))  # merge heads back


class MSARowAttn(nn.Module):
    """Alg 7: attention along each MSA ROW (across residues), biased by z - pair info steers the MSA."""

    def __init__(self, c_m, c_z, c=32, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(c_m)
        self.mha = MHA(c_m, c, heads)
        self.bias = nn.Linear(c_z, heads, bias=False)  # z_ij -> per-head score bias

    def forward(self, m, z):
        b = self.bias(z).permute(2, 0, 1)  # (h, R_q, R_k)
        return self.mha(self.norm(m), b[None])  # bias broadcasts over all S rows


class MSAColAttn(nn.Module):
    """Alg 8: attention along each MSA COLUMN (across sequences) - where co-evolution is read out."""

    def __init__(self, c_m, c=32, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(c_m)
        self.mha = MHA(c_m, c, heads)

    def forward(self, m):
        return self.mha(self.norm(m).transpose(0, 1)).transpose(
            0, 1
        )  # (S,R,c)->(R,S,c): attend over S, then transpose back


class Transition(nn.Module):
    """Alg 9/15: 2-layer MLP with 4x expansion - the per-position 'thinking' step."""

    def __init__(self, c, n=4):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, n * c)
        self.fc2 = nn.Linear(n * c, c)

    def forward(self, x):
        return self.fc2(self.fc1(self.norm(x)).relu())


class OuterProductMean(nn.Module):
    """Alg 10: MSA -> pair. THE bridge between m and z: mean over sequences of per-sequence outer products."""

    def __init__(self, c_in, c_z, c=32):
        super().__init__()
        self.norm = nn.LayerNorm(c_in)
        self.a = nn.Linear(c_in, c)
        self.b = nn.Linear(c_in, c)
        self.o = nn.Linear(c * c, c_z)

    def forward(self, m):
        m = self.norm(m)
        z = (
            torch.einsum("sic,sjd->ijcd", self.a(m), self.b(m)) / m.shape[0]
        )  # each-with-each over residues, mean over seqs
        return self.o(z.flatten(-2))  # (R,R,c*c) -> (R,R,c_z)


class TriMult(nn.Module):
    """Alg 11/12: triangle multiplicative update - z_ij learns from all paths i->k->j ('if i~k and k~j then i~j')."""

    def __init__(self, c_z, c=32, outgoing=True):
        super().__init__()
        self.out = outgoing
        self.norm = nn.LayerNorm(c_z)
        self.norm2 = nn.LayerNorm(c)
        self.pa = nn.Linear(c_z, c)
        self.pb = nn.Linear(c_z, c)  # projections
        self.ga = nn.Linear(c_z, c)
        self.gb = nn.Linear(c_z, c)  # element-wise gates (sigmoid)
        self.g = nn.Linear(c_z, c_z)
        self.o = nn.Linear(c, c_z)  # output gate + projection back to c_z

    def forward(self, z):
        z = self.norm(z)
        a = self.ga(z).sigmoid() * self.pa(z)  # gated projection a
        b = self.gb(z).sigmoid() * self.pb(z)  # gated projection b
        eq = (
            "ikc,jkc->ijc" if self.out else "kic,kjc->ijc"
        )  # outgoing: sum_k a_ik*b_jk; incoming: sum_k a_ki*b_kj
        return self.g(z).sigmoid() * self.o(self.norm2(torch.einsum(eq, a, b)))


class TriAttn(nn.Module):
    """Alg 13/14: triangle attention - attention over rows ('start node') or columns ('end node') of z, biased by z."""

    def __init__(self, c_z, c=32, heads=4, start=True):
        super().__init__()
        self.start = start
        self.norm = nn.LayerNorm(c_z)
        self.mha = MHA(c_z, c, heads)
        self.bias = nn.Linear(c_z, heads, bias=False)
        self.g = nn.Linear(c_z, c_z)

    def forward(self, z):
        if not self.start:
            z = z.transpose(0, 1)  # column attention = row attention on z transposed
        zn = self.norm(z)
        b = self.bias(zn).permute(2, 0, 1)  # (h, R, R) self-bias
        out = self.g(zn).sigmoid() * self.mha(zn, b)  # gated attention
        return out if self.start else out.transpose(0, 1)  # transpose back if needed


class EvoformerBlock(nn.Module):
    """Alg 6: the full communication round m<->z. col_attn=False turns it into the cheaper extra-MSA block (Alg 18 style)."""

    def __init__(self, cfg, col_attn=True, c_m=None):
        super().__init__()
        c_m = c_m or cfg.c_m  # extra-MSA tower runs on narrower c_e channels
        h = cfg.c_hidden
        self.row = MSARowAttn(c_m, cfg.c_z, h, cfg.heads)
        self.col = MSAColAttn(c_m, h, cfg.heads) if col_attn else None
        self.tm = Transition(c_m)
        self.opm = OuterProductMean(c_m, cfg.c_z, h)
        self.tmo = TriMult(cfg.c_z, h, True)
        self.tmi = TriMult(cfg.c_z, h, False)
        self.tas = TriAttn(cfg.c_z, h, cfg.pair_heads, True)
        self.tae = TriAttn(cfg.c_z, h, cfg.pair_heads, False)
        self.tp = Transition(cfg.c_z)
        self.drop = nn.Dropout(0.1)  # residual dropout, keeps the tiny model honest

    def forward(self, m, z):
        m = m + self.drop(self.row(m, z))  # MSA stack: row attn (z-biased)
        if self.col is not None:
            m = m + self.drop(self.col(m))  #           col attn (skipped for extra MSA)
        m = m + self.drop(self.tm(m))  #           transition
        z = z + self.drop(self.opm(m))  # m -> z: outer product mean
        z = z + self.drop(self.tmo(z))  # pair stack: triangle multiplication x2
        z = z + self.drop(self.tmi(z))
        z = z + self.drop(self.tas(z))  #             triangle attention x2
        z = z + self.drop(self.tae(z))
        return m, z + self.drop(self.tp(z))  #             transition


class Evoformer(nn.Module):
    """The trunk. Extra MSA runs FIRST and refines z (its sequences never enter m directly); then the main blocks run."""

    def __init__(self, cfg):
        super().__init__()
        self.extra = nn.ModuleList(
            [
                EvoformerBlock(cfg, col_attn=False, c_m=cfg.c_e)
                for _ in range(cfg.n_extra)
            ]
        )
        self.blocks = nn.ModuleList([EvoformerBlock(cfg) for _ in range(cfg.n_evo)])

    def forward(self, m, z, e):
        for b in self.extra:
            e, z = b(
                e, z
            )  # extra MSA: row attn + OPM + pair ops (no col attn) -> better z
        for b in self.blocks:
            m, z = b(m, z)  # main Evoformer
        return m, z
