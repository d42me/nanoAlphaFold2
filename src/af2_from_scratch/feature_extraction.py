"""Stage 1 — feature extraction: convert an A3M alignment into model inputs.

Pipeline::

    A3M text
       │
       ├─ parse + remove lowercase insertions
       ├─ deduplicate + one-hot encode
       ├─ compute the per-position MSA profile
       └─ sample rows + apply training-time masking
                    │
                    ▼
       msa_feat · extra_msa_feat · target_feat · residue_index

This educational implementation uses random MSA sampling instead of AlphaFold's
full cluster-assignment pipeline. Public flow: ``msa_features`` parses reusable
features once; ``sample_batch`` creates a fresh model-ready view for each step.
"""

import torch
import torch.nn.functional as F

AA = "ARNDCQEGHILKMFPSTWYV"  # 20 amino acids, AlphaFold's canonical order (alphabetical 3-letter codes)
IDX = {
    a: i for i, a in enumerate(AA + "X-")
}  # letter -> class id: 0-19 = AA, 20 = X (unknown), 21 = '-' (gap)


def parse_seq(sequence):
    """Remove A3M insertions and count those before each aligned residue."""
    aligned_sequence = []
    deletion_counts = []
    insertion_count = 0

    for residue in sequence:
        if residue.islower():
            insertion_count += 1
            continue

        aligned_sequence.append(residue)
        deletion_counts.append(insertion_count)
        insertion_count = 0

    return "".join(aligned_sequence), deletion_counts


def onehot(seq, n=22):
    return F.one_hot(
        torch.tensor([IDX.get(c, 20) for c in seq]), n
    ).float()  # unknown letters fall back to X; -> (N_res, n)


def msa_features(path, max_seqs=None):
    """The whole pipeline: .a3m -> feature dict."""
    lines = [
        line.strip() for line in open(path) if line[0] not in ">#"
    ]  # sequences only: skip '>' headers and the '#<length>' first line
    uniq = {}  # cleaned seq -> deletion counts; dict = order-preserving O(n) dedup
    n_res = len(
        parse_seq(lines[0])[0]
    )  # query length = the reference frame every hit must match
    for s in lines:
        c, d = parse_seq(s)
        if len(c) != n_res:
            continue  # drop fragment hits (merged a3m files contain partials)
        uniq.setdefault(
            c, d
        )  # AlphaFold dedups *after* insertion removal; first copy wins
    seqs, dels = zip(*uniq.items())  # unzip keys/values into parallel tuples
    if max_seqs:
        seqs, dels = (
            seqs[:max_seqs],
            dels[:max_seqs],
        )  # our stand-in for clustering: keep the first max_seqs (query is always seqs[0])
    aatype = torch.stack(
        [onehot(s) for s in seqs]
    )  # (N_seq, N_res, 22) one-hot MSA, incl. gap token
    return {
        "msa_aatype": aatype,  # core evolution data: which AA each homolog has at each position
        "deletion_matrix": torch.tensor(
            dels
        ).float(),  # (N_seq, N_res) how many residues were inserted before each position
        "profile": aatype.mean(
            0
        ),  # (N_res, 22) per-position AA distribution = mean over the sequence axis (a reduction!)
        "target_feat": onehot(
            seqs[0].replace("-", "X"), 21
        ),  # (N_res, 21) one-hot query sequence (no gap token needed)
        "residue_index": torch.arange(
            aatype.shape[1]
        ).float(),  # (N_res,) position ids 0..N_res-1
    }


def bert_mask(oh, prof, g, p=0.15):
    """AF-style masking: select 15% of positions; of those 70% mask / 10% random AA / 10% profile / 10% keep."""
    sel = (
        torch.rand(oh.shape[:2], generator=g) < p
    )  # which positions get replaced at all
    roll = torch.rand(
        oh.shape[:2], generator=g
    )  # decides the replacement type per position
    rnd = F.one_hot(
        torch.randint(0, 22, oh.shape[:2], generator=g), 22
    ).float()  # random-AA replacement candidates
    repl = torch.where(
        (roll < 0.7)[..., None],
        torch.zeros_like(oh),  # 70%: zero vector = our compact 'mask token'
        torch.where(
            (roll < 0.8)[..., None],
            rnd,  # 10%: uniformly random amino acid
            torch.where((roll < 0.9)[..., None], prof.expand_as(oh), oh),
        ),
    )  # 10%: MSA profile, 10%: unchanged
    return torch.where(sel[..., None], repl, oh)  # apply only at selected positions


def sample_batch(f, n_clu=128, n_ext=128, mask_p=0.15, seed=None):
    """Training-time data builder: random MSA subsample + masking -> model-ready feature dict.
    Random subsampling replaces clustering AND acts as data augmentation (new view every step)."""
    g = (
        torch.Generator().manual_seed(seed) if seed is not None else None
    )  # reproducible when seeded
    S, D, P = (
        f["msa_aatype"],
        f["deletion_matrix"],
        f["profile"],
    )  # (N,R,22) one-hots, (N,R) deletions, (R,22) profile
    perm = (
        torch.randperm(S.shape[0] - 1, generator=g) + 1
    )  # shuffle all rows EXCEPT the query (row 0)
    clu = torch.cat(
        [torch.tensor([0]), perm[: n_clu - 1]]
    )  # cluster rows: query is always row 0
    ext = perm[n_clu - 1 : n_clu - 1 + n_ext]  # extra rows: plain random crop
    masked = bert_mask(
        S[clu], P, g, mask_p
    )  # AF masks the cluster centers -> regularizer
    msa_feat = torch.cat(
        [masked, D[clu][..., None], P.expand(len(clu), -1, -1)], -1
    )  # (n_clu, R, 45) = masked one-hot(22)+deletion(1)+profile(22)
    extra_feat = torch.cat(
        [S[ext], D[ext][..., None]], -1
    )  # (n_ext, R, 23) = one-hot(22)+deletion(1), no cluster averages
    return {
        "msa_feat": msa_feat,
        "extra_msa_feat": extra_feat,
        "target_feat": f["target_feat"],
        "residue_index": f["residue_index"],
    }


if __name__ == "__main__":
    from pathlib import Path

    example = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "tautomerase"
        / "alignment.a3m"
    )
    f = msa_features(example)
    for k, v in f.items():
        print(f"{k:16s} {str(tuple(v.shape)):18s} {v.dtype}")
    ok = torch.allclose(f["profile"].sum(-1), torch.ones(f["profile"].shape[0]))
    print("profile rows sum to 1:", ok)
