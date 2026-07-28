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
    """Gated multi-head self-attention with optional score bias.

    Global attention averages the queries into one vector and shares one key and
    value projection across heads. The output is broadcast back across positions
    by the position-specific gate.
    """

    def __init__(
        self,
        c_in,
        c=32,
        heads=8,
        gated=False,
        global_attention=False,
    ):
        super().__init__()
        self.h = heads
        self.c = c
        self.global_attention = global_attention
        key_value_heads = 1 if global_attention else heads

        self.q = nn.Linear(c_in, heads * c, bias=False)
        self.k = nn.Linear(c_in, key_value_heads * c, bias=False)
        self.v = nn.Linear(c_in, key_value_heads * c, bias=False)
        self.out = nn.Linear(heads * c, c_in)
        self.gate = nn.Linear(c_in, heads * c) if gated else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.ones_(self.gate.bias)

    def forward(self, x, bias=None):
        q = self.q(x).unflatten(-1, (self.h, self.c))
        if self.global_attention:
            q = q.mean(dim=-3, keepdim=True)
            k = self.k(x).unsqueeze(-2)
            v = self.v(x).unsqueeze(-2)
        else:
            k = self.k(x).unflatten(-1, (self.h, self.c))
            v = self.v(x).unflatten(-1, (self.h, self.c))

        scores = torch.einsum("...qhc,...khc->...hqk", q, k) * self.c**-0.5
        if bias is not None:
            scores = scores + bias

        output = torch.einsum(
            "...hqk,...khc->...qhc",
            scores.softmax(-1),
            v,
        ).flatten(-2)
        if self.gate is not None:
            output = self.gate(x).sigmoid() * output
        return self.out(output)


class SharedDropout(nn.Module):
    """Apply one dropout mask shared across a row or column dimension."""

    def __init__(self, p, shared_dim):
        super().__init__()
        self.shared_dim = shared_dim
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        mask_shape = list(x.shape)
        mask_shape[self.shared_dim] = 1
        mask = self.dropout(x.new_ones(mask_shape))
        return x * mask


class DropoutRowwise(SharedDropout):
    def __init__(self, p):
        super().__init__(p=p, shared_dim=-2)


class DropoutColumnwise(SharedDropout):
    def __init__(self, p):
        super().__init__(p=p, shared_dim=-3)


class MSARowAttn(nn.Module):
    """Alg 7: attention along each MSA ROW (across residues), biased by z - pair info steers the MSA."""

    def __init__(self, c_m, c_z, c=32, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(c_m)
        self.mha = MHA(c_m, c, heads, gated=True)
        self.bias = nn.Linear(c_z, heads, bias=False)  # z_ij -> per-head score bias

    def forward(self, m, z):
        b = self.bias(z).permute(2, 0, 1)  # (h, R_q, R_k)
        return self.mha(self.norm(m), b[None])  # bias broadcasts over all S rows


class MSAColAttn(nn.Module):
    """Alg 8: gated attention across sequences at each residue position."""

    def __init__(self, c_m, c=32, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(c_m)
        self.mha = MHA(c_m, c, heads, gated=True)

    def forward(self, m):
        return self.mha(self.norm(m).transpose(0, 1)).transpose(0, 1)


class MSAColumnGlobalAttention(nn.Module):
    """Memory-efficient extra-MSA attention across sequences (Algorithm 19)."""

    def __init__(self, c_m, c=8, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(c_m)
        self.mha = MHA(
            c_m,
            c,
            heads,
            gated=True,
            global_attention=True,
        )

    def forward(self, m):
        return self.mha(self.norm(m).transpose(0, 1)).transpose(0, 1)


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
    """One MSA-to-pair communication round from Algorithm 6 or 18."""

    def __init__(self, cfg, c_m=None, global_column_attention=False):
        super().__init__()
        c_m = c_m or cfg.c_m
        hidden = cfg.c_hidden
        column_attention = (
            MSAColumnGlobalAttention if global_column_attention else MSAColAttn
        )

        self.row = MSARowAttn(c_m, cfg.c_z, hidden, cfg.heads)
        self.col = column_attention(c_m, hidden, cfg.heads)
        self.tm = Transition(c_m)
        self.opm = OuterProductMean(c_m, cfg.c_z, hidden)
        self.tmo = TriMult(cfg.c_z, hidden, True)
        self.tmi = TriMult(cfg.c_z, hidden, False)
        self.tas = TriAttn(cfg.c_z, hidden, cfg.pair_heads, True)
        self.tae = TriAttn(cfg.c_z, hidden, cfg.pair_heads, False)
        self.tp = Transition(cfg.c_z)

        self.msa_row_dropout = DropoutRowwise(cfg.msa_dropout)
        self.pair_row_dropout = DropoutRowwise(cfg.pair_dropout)
        self.pair_column_dropout = DropoutColumnwise(cfg.pair_dropout)

    def forward(self, m, z):
        m = m + self.msa_row_dropout(self.row(m, z))
        m = m + self.col(m)
        m = m + self.tm(m)
        z = z + self.opm(m)
        z = z + self.pair_row_dropout(self.tmo(z))
        z = z + self.pair_row_dropout(self.tmi(z))
        z = z + self.pair_row_dropout(self.tas(z))
        z = z + self.pair_column_dropout(self.tae(z))
        z = z + self.tp(z)
        return m, z


class Evoformer(nn.Module):
    """The trunk. Extra MSA runs FIRST and refines z (its sequences never enter m directly); then the main blocks run."""

    def __init__(self, cfg):
        super().__init__()
        self.extra = nn.ModuleList(
            [
                EvoformerBlock(
                    cfg,
                    c_m=cfg.c_e,
                    global_column_attention=True,
                )
                for _ in range(cfg.n_extra)
            ]
        )
        self.blocks = nn.ModuleList([EvoformerBlock(cfg) for _ in range(cfg.n_evo)])

    def forward(self, m, z, e):
        if e.shape[0] > 0:
            for block in self.extra:
                e, z = block(e, z)
        for block in self.blocks:
            m, z = block(m, z)
        return m, z
