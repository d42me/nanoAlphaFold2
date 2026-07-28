"""Stage 4 — geometry: rigid-transform tools used by structure and losses.

Data flow::

    quaternion q ─► rotation R ─┐
                                ├─► transform T = [R | t]
    translation t ──────────────┘             │
                                      apply / invert
                                             │
                              local points ◄──┴──► global points

``frames_from_backbone`` creates teacher residue frames and ``kabsch_rmsd``
compares structures independently of global rotation and translation. This
compact model predicts backbone/Cα geometry only, so side-chain frames are not
included.
"""

import torch


def quat_to_rot(q):
    """Unnormalized quaternion (..., 4) -> rotation matrix (..., 3, 3). Normalizing first keeps gradients tame."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).unflatten(-1, (3, 3))


def make_T(R, t):
    """Assemble (..., 4, 4) homogeneous transform from rotation (..., 3, 3) and translation (..., 3)."""
    T = torch.zeros(*R.shape[:-2], 4, 4, device=R.device, dtype=R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def apply(T, x):
    """T @ x for points x (..., 3): rotate, then translate. Broadcasting handles batched frames/points."""
    return torch.einsum("...ij,...j->...i", T[..., :3, :3], x) + T[..., :3, 3]


def invert(T):
    """Inverse of [R|t] is [R^T | -R^T t]."""
    Rt = T[..., :3, :3].transpose(-1, -2)
    return make_T(Rt, -torch.einsum("...ij,...j->...i", Rt, T[..., :3, 3]))


def kabsch_rmsd(p, q):
    """RMSD between point sets after optimal rigid alignment (Kabsch). Frame-invariant model quality metric."""
    p, q = p - p.mean(0), q - q.mean(0)  # center both point clouds
    U, _, Vh = torch.linalg.svd(p.T @ q)  # SVD of the covariance
    d = torch.sign(torch.linalg.det(U @ Vh))  # guard against reflections
    R = U @ torch.diag(torch.tensor([1.0, 1.0, d], device=p.device)) @ Vh
    return (
        ((p @ R - q).norm(dim=-1) ** 2).mean().sqrt()
    )  # RMSD after applying the optimal rotation


def frames_from_backbone(N, CA, C):
    """Gram-Schmidt orthonormal frame per residue from backbone atoms (AF2-style): origin at CA.
    Used to build TEACHER frames for the FAPE loss from the .cif coordinates."""
    e1 = C - CA
    e1 = e1 / e1.norm(dim=-1, keepdim=True)  # first axis: CA -> C direction
    v2 = N - CA
    e2 = v2 - e1 * (e1 * v2).sum(-1, keepdim=True)  # orthogonalize N against e1
    e2 = e2 / e2.norm(dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2, dim=-1)  # right-handed third axis
    return make_T(
        torch.stack([e1, e2, e3], dim=-1), CA
    )  # columns = local axes, origin = CA
