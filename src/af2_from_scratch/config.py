"""Shared educational configuration for AlphaFold 2 from Scratch.

Every model, data, and training knob lives here. Comments show the comparable
AlphaFold 2 value where this teaching implementation intentionally shrinks it.
All major block types remain; widths, depths, and MSA row counts are reduced.
"""

from dataclasses import dataclass


@dataclass
class AF2Config:
    # --- representation widths (params scale ~ c^2, so this is the big lever) ---
    c_m: int = 64  # MSA representation width            (AF2: 256)
    c_z: int = 64  # pair representation width           (AF2: 128)
    c_e: int = 32  # extra-MSA representation width      (AF2: 64)
    c_s: int = 128  # single representation width         (AF2: 384)
    # --- attention geometry ---
    heads: int = 8  # MSA attention heads                 (AF2: 8)
    pair_heads: int = 4  # triangle attention heads            (AF2: 4)
    ipa_heads: int = 4  # IPA heads                           (AF2: 12)
    c_hidden: int = 32  # hidden dim inside attention/tri ops (AF2: 32/128)
    # --- depths ---
    n_evo: int = 4  # Evoformer blocks                    (AF2: 48)
    n_extra: int = 1  # extra-MSA tower blocks              (AF2: 4)
    n_ipa: int = 2  # structure module layers             (AF2: 8)
    # --- data ---
    n_clu: int = 128  # MSA rows fed to the model           (AF2: 512)
    n_ext: int = 128  # extra-MSA rows                      (AF2: ~1152)
    mask_p: float = 0.15  # fraction of MSA positions masked (BERT-style)
    # --- training ---
    recycles: int = 1  # train-time recycles (3-4 at inference)
    lr: float = 1e-3
    steps: int = 50000
    seed: int = 0
