"""Stage 5 — structure module: turn Evoformer features into backbone geometry.

Iterative update::

    s + z + current T
            │
            ▼
       invariant point attention ─► transition ─► backbone update
            ▲                                         │
            └──────────────── new residue frames T ◄──┘

Frames begin at the origin; each round predicts a local quaternion and
translation. Their origins are the final Cα coordinates. Unlike full AlphaFold,
this teaching model omits side-chain torsion prediction and atom reconstruction.
"""

import torch
import torch.nn as nn
from .geometry import quat_to_rot, make_T, apply, invert


class IPA(nn.Module):
    """Invariant Point Attention: attention that also reasons about 3D POINTS placed in each residue's frame.
    'Invariant' = rotating/translating the whole protein changes nothing (points are compared in global space,
    then read out in the local frame)."""

    def __init__(self, cfg, c=16, n_qp=4, n_vp=4):
        super().__init__()
        h, cs, cz = cfg.ipa_heads, cfg.c_s, cfg.c_z
        self.h, self.c, self.n_qp, self.n_vp = h, c, n_qp, n_vp
        self.ns = nn.LayerNorm(cs)
        self.nz = nn.LayerNorm(cz)
        self.qkv = nn.Linear(
            cs, 3 * h * c, bias=False
        )  # scalar q/k/v (like classic attention)
        self.qp = nn.Linear(
            cs, h * n_qp * 3
        )  # query POINTS (3D), placed in residue frame
        self.kvp = nn.Linear(
            cs, h * (n_qp + n_vp) * 3
        )  # key POINTS (same count as queries) + value POINTS
        self.bias = nn.Linear(cz, h, bias=False)  # pair repr -> score bias
        self.gamma = nn.Parameter(
            torch.zeros(h)
        )  # learnable weight of the 3D point term
        self.out = nn.Linear(
            h * (c + cz + 4 * n_vp), cs
        )  # scalars + pair + points + point norms -> c_s

    def forward(self, s, z, T):
        s_, z_ = self.ns(s), self.nz(z)
        q, k, v = self.qkv(s_).chunk(3, -1)  # 3x (R, h*c)

        def split_heads(tensor):
            return tensor.unflatten(-1, (self.h, self.c))

        q, k, v = split_heads(q), split_heads(k), split_heads(v)  # (R, h, c)
        # --- point q/k/v in each residue's LOCAL frame, then lifted to GLOBAL space via apply(T, .) ---
        R_ = s_.shape[0]
        Ti = T.unsqueeze(1)  # (R, 1, 4, 4): same T for all of a residue's points

        def merge_point_axes(tensor, n_points):
            return tensor.reshape(R_, self.h * n_points, 3)

        qp = apply(Ti, merge_point_axes(self.qp(s_), self.n_qp)).reshape(
            R_, self.h, self.n_qp, 3
        )  # query POINTS, lifted to GLOBAL space
        kp, vp = self.kvp(s_).split(
            [self.h * self.n_qp * 3, self.h * self.n_vp * 3], -1
        )
        kp = apply(Ti, merge_point_axes(kp, self.n_qp)).reshape(
            R_, self.h, self.n_qp, 3
        )
        vp = apply(Ti, merge_point_axes(vp, self.n_vp)).reshape(
            R_, self.h, self.n_vp, 3
        )
        # --- three score terms: content + pair bias + 3D proximity ---
        a = torch.einsum("qhc,khc->hqk", q, k) * self.c**-0.5  # content scores
        a = a + self.bias(z_).permute(2, 0, 1)  # pair bias
        d2 = (
            (qp.unsqueeze(1) - kp.unsqueeze(0)).pow(2).sum(-1)
        )  # (R_q, R_k, h, n_qp): squared point distances
        a = (
            a - self.gamma.exp()[:, None, None] * d2.sum(-1).permute(2, 0, 1) / 2
        )  # nearby points attract, learnable strength
        w = a.softmax(-1)  # attention weights
        # --- three outputs: scalar values, pair values, point values (read out in the LOCAL frame!) ---
        o_s = torch.einsum("hqk,khc->qhc", w, v).flatten(-2)  # (R, h*c)
        o_z = torch.einsum("hqk,qkc->qhc", w, z_).flatten(-2)  # (R, h*c_z)
        o_p = torch.einsum("hqk,khpc->qhpc", w, vp)  # weighted points, GLOBAL space
        o_p = apply(
            invert(T).unsqueeze(1), o_p.reshape(R_, self.h * self.n_vp, 3)
        )  # back into each residue's frame = invariance
        o_p = o_p.reshape(R_, self.h, self.n_vp, 3)
        return self.out(
            torch.cat([o_s, o_z, o_p.flatten(-3), o_p.norm(dim=-1).flatten(-2)], -1)
        )  # points + their norms (norms are invariant)


class BackboneUpdate(nn.Module):
    """Alg 23: predict a local rigid update (quaternion + translation) per residue from s."""

    def __init__(self, c_s):
        super().__init__()
        self.lin = nn.Linear(c_s, 6)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(
            self.lin.bias
        )  # zero-init: training starts from identity updates

    def forward(self, s, T):
        b = self.lin(s)
        q = torch.cat(
            [torch.ones_like(b[..., :1]), b[..., :3]], -1
        )  # quaternion (1, q1, q2, q3); AF trick: fixed real part
        dT = make_T(
            quat_to_rot(q), b[..., 3:] * 0.1
        )  # 0.1 translation scale: stabilizes early training
        return T @ dT  # compose: global = current ∘ local update


class StructureModule(nn.Module):
    """Alg 22 skeleton: n_ipa rounds of [IPA -> single-repr update -> frame update]. CA positions = frame origins."""

    def __init__(self, cfg):
        super().__init__()
        self.n = cfg.n_ipa
        self.ipa = nn.ModuleList([IPA(cfg) for _ in range(self.n)])
        self.tr = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cfg.c_s),  # single-repr transition between IPA rounds
                    nn.Linear(cfg.c_s, cfg.c_s),
                    nn.ReLU(),
                    nn.Linear(cfg.c_s, cfg.c_s),
                )
                for _ in range(self.n)
            ]
        )
        self.upd = BackboneUpdate(cfg.c_s)  # shared across layers (AF2 shares it too)
        self.ns0 = nn.LayerNorm(cfg.c_s)
        self.nz0 = nn.LayerNorm(cfg.c_z)

    def forward(self, s, z):
        s, z = self.ns0(s), self.nz0(z)
        R = s.shape[0]
        T = make_T(
            torch.eye(3, device=s.device).expand(
                R, 3, 3
            ),  # start: identity frame at the origin per residue
            torch.zeros(R, 3, device=s.device),
        )
        for ipa, tr in zip(self.ipa, self.tr):
            s = s + ipa(s, z, T)  # 3D-aware attention updates the single repr
            s = s + tr(s)  # transition
            T = self.upd(s, T)  # frames move; next IPA round sees the new geometry
        return T, s  # T[..., :3, 3] = predicted CA positions
