"""Load CIF teacher structures for nanoAlphaFold2 training.

Column indices come from each file's own ``_atom_site`` header, so tutorial and
AlphaFold DB layouts both work. AlphaFold DB pLDDT values are retained for
analysis; confidence supervision still uses the model's own lDDT target.
"""

import torch
from .geometry import frames_from_backbone


def load_teacher(cif_path, n_res):
    """Returns backbone N/CA/C positions, teacher pLDDT (if present), and per-residue local frames."""
    N = torch.zeros(n_res, 3)
    CA = torch.zeros(n_res, 3)
    C = torch.zeros(n_res, 3)
    plddt = torch.full((n_res,), float("nan"))
    cols, seen = {}, set()
    for line in open(cif_path):
        if line.startswith("_atom_site."):  # header lines define the column order
            cols[line.strip().split(".")[1]] = len(
                cols
            )  # e.g. cols["label_seq_id"] = 8
            continue
        if not line.startswith("ATOM"):
            continue
        f = line.split()
        res = int(f[cols["label_seq_id"]]) - 1  # 1-based -> 0-based
        pos = torch.tensor(
            [
                float(f[cols["Cartn_x"]]),
                float(f[cols["Cartn_y"]]),
                float(f[cols["Cartn_z"]]),
            ]
        )
        atom = f[cols["label_atom_id"]]
        if atom == "N":
            N[res] = pos
        elif atom == "CA":
            CA[res] = pos
            b = f[cols["B_iso_or_equiv"]]
            if b not in ".?":
                plddt[res] = float(b)  # AF DB: real pLDDT; tutorial cif: '.'
        elif atom == "C":
            C[res] = pos
        seen.add(res)
    assert len(seen) == n_res, f"expected {n_res} residues, got {len(seen)}"
    return {
        "N": N,
        "CA": CA,
        "C": C,
        "plddt": plddt,
        "frames": frames_from_backbone(N, CA, C),
    }  # teacher's local frames (Gram-Schmidt) for FAPE
