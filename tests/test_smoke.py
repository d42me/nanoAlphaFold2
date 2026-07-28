from dataclasses import replace

import torch
import torch.nn as nn

from af2_from_scratch import AF2Config, AlphaFold2FromScratch
from af2_from_scratch.feature_embedding import RecyclingEmbedder
from af2_from_scratch.evoformer import (
    DropoutColumnwise,
    DropoutRowwise,
    Evoformer,
    MHA,
    MSAColumnGlobalAttention,
)
from af2_from_scratch.feature_extraction import parse_seq
from af2_from_scratch.geometry import (
    apply,
    invert,
    kabsch_align,
    kabsch_rmsd,
    make_T,
    quat_to_rot,
)
from af2_from_scratch.losses import fape_ca, lddt_ca, lddt_target
from af2_from_scratch.structure_module import IPA, StructureModule


def tiny_config():
    return AF2Config(
        c_m=16,
        c_z=16,
        c_e=8,
        c_s=32,
        heads=2,
        pair_heads=2,
        ipa_heads=2,
        c_hidden=4,
        n_evo=1,
        n_extra=1,
        n_ipa=1,
        n_clu=3,
        n_ext=2,
        recycles=0,
    )


def test_model_output_shapes():
    n_seq, n_extra, n_res = 3, 2, 5
    batch = {
        "msa_feat": torch.randn(n_seq, n_res, 45),
        "extra_msa_feat": torch.randn(n_extra, n_res, 23),
        "target_feat": torch.randn(n_res, 21),
        "residue_index": torch.arange(n_res).float(),
    }

    model = AlphaFold2FromScratch(tiny_config()).eval()
    with torch.no_grad():
        output = model(batch)

    assert output["T"].shape == (n_res, 4, 4)
    assert output["ca"].shape == (n_res, 3)
    assert output["disto_logits"].shape == (n_res, n_res, 64)
    assert output["plddt_logits"].shape == (n_res, 50)


def test_recycling_restarts_from_initial_embeddings():
    cfg = replace(tiny_config(), recycles=1)
    model = AlphaFold2FromScratch(cfg).eval()
    n_res = 4
    initial_m = torch.randn(2, n_res, cfg.c_m)
    initial_z = torch.randn(n_res, n_res, cfg.c_z)
    extra_msa = torch.empty(0, n_res, cfg.c_e)

    class StaticEmbedder(nn.Module):
        def forward(self, batch):
            return initial_m, initial_z, extra_msa

    class RecordingRecycler(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, m, z, previous_m, previous_z):
            self.inputs.append((m.detach().clone(), z.detach().clone()))
            return m, z

    class IncrementingEvoformer(nn.Module):
        def forward(self, m, z, e):
            return m + 1, z + 1

    recycler = RecordingRecycler()
    model.emb = StaticEmbedder()
    model.rec = recycler
    model.evo = IncrementingEvoformer()

    with torch.no_grad():
        model({})

    assert len(recycler.inputs) == 2
    for m, z in recycler.inputs:
        torch.testing.assert_close(m, initial_m)
        torch.testing.assert_close(z, initial_z)


def test_evoformer_accepts_an_empty_extra_msa():
    cfg = tiny_config()
    m = torch.randn(2, 4, cfg.c_m)
    z = torch.randn(4, 4, cfg.c_z)
    e = torch.empty(0, 4, cfg.c_e)

    with torch.no_grad():
        output_m, output_z = Evoformer(cfg).eval()(m, z, e)

    assert torch.isfinite(output_m).all()
    assert torch.isfinite(output_z).all()


def test_a3m_insertions_are_counted_and_removed():
    sequence, deletion_counts = parse_seq("PIAxqIHsdILEGR")

    assert sequence == "PIAIHILEGR"
    assert deletion_counts == [0, 0, 0, 2, 0, 2, 0, 0, 0, 0]


def test_recycling_updates_only_the_query_msa_row():
    cfg = tiny_config()
    embedder = RecyclingEmbedder(cfg)
    m = torch.zeros(3, 5, cfg.c_m)
    z = torch.zeros(5, 5, cfg.c_z)
    previous_query = torch.randn(5, cfg.c_m)
    previous_pairs = torch.randn(5, 5, cfg.c_z)

    recycled_m, recycled_z = embedder(m, z, previous_query, previous_pairs)

    torch.testing.assert_close(recycled_m[0], embedder.nm(previous_query))
    torch.testing.assert_close(recycled_m[1:], m[1:])
    torch.testing.assert_close(recycled_z, embedder.nz(previous_pairs))


def test_gated_attention_can_close_its_output():
    attention = MHA(c_in=4, c=2, heads=2, gated=True).eval()
    with torch.no_grad():
        attention.gate.weight.zero_()
        attention.gate.bias.fill_(-100)
        attention.out.bias.zero_()

    output = attention(torch.randn(3, 4))

    torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-6, rtol=0)


def test_extra_msa_uses_global_column_attention():
    cfg = tiny_config()
    evoformer = Evoformer(cfg)
    global_attention = evoformer.extra[0].col
    extra_msa = torch.randn(3, 5, cfg.c_e)

    assert isinstance(global_attention, MSAColumnGlobalAttention)
    assert global_attention(extra_msa).shape == extra_msa.shape
    assert global_attention.mha.k.out_features == cfg.c_hidden
    assert global_attention.mha.v.out_features == cfg.c_hidden


def test_shared_dropout_broadcasts_row_and_column_masks():
    values = torch.ones(4, 5, 6)

    torch.manual_seed(0)
    row_output = DropoutRowwise(0.5)(values)
    torch.testing.assert_close(row_output, row_output[:, :1].expand_as(row_output))

    torch.manual_seed(0)
    column_output = DropoutColumnwise(0.5)(values)
    torch.testing.assert_close(
        column_output, column_output[:1].expand_as(column_output)
    )


def test_lddt_ca_uses_corresponding_residue_pair_distances():
    true_ca = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.2, 0.4, 1.0],
        ]
    )
    transform = make_T(
        quat_to_rot(torch.tensor([1.0, 0.2, -0.1, 0.3])),
        torch.tensor([2.0, -1.0, 3.0]),
    )
    predicted_ca = apply(transform, true_ca)

    torch.testing.assert_close(lddt_ca(predicted_ca, true_ca), torch.ones(4))
    torch.testing.assert_close(lddt_target(predicted_ca, true_ca), torch.full((4,), 49))


def test_fape_is_normalized_by_its_length_scale():
    frame = make_T(torch.eye(3).unsqueeze(0), torch.zeros(1, 3))
    predicted_ca = torch.tensor([[20.0, 0.0, 0.0]])
    true_ca = torch.zeros(1, 3)

    loss = fape_ca(frame, frame, predicted_ca, true_ca)

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_kabsch_rmsd_preserves_float64_and_removes_rigid_motion():
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.1, 0.4]],
        dtype=torch.float64,
    )
    rotation = quat_to_rot(torch.tensor([1.0, 0.2, -0.1, 0.3], dtype=torch.float64))
    transform = make_T(rotation, torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64))

    moved_points = apply(transform, points)
    aligned_points = kabsch_align(moved_points, points)
    rmsd = kabsch_rmsd(points, moved_points)

    assert rmsd.dtype == torch.float64
    torch.testing.assert_close(aligned_points, points, atol=1e-7, rtol=0)
    torch.testing.assert_close(rmsd, torch.zeros_like(rmsd), atol=1e-7, rtol=0)


def test_ipa_is_invariant_to_global_rigid_motion():
    cfg = tiny_config()
    n_res = 5
    s = torch.randn(n_res, cfg.c_s)
    z = torch.randn(n_res, n_res, cfg.c_z)
    frames = make_T(
        torch.eye(3).expand(n_res, 3, 3),
        torch.randn(n_res, 3),
    )
    global_transform = make_T(
        quat_to_rot(torch.tensor([1.0, 0.2, -0.1, 0.3])),
        torch.tensor([2.0, -1.0, 3.0]),
    )
    ipa = IPA(cfg).eval()

    with torch.no_grad():
        original = ipa(s, z, frames)
        moved = ipa(s, z, global_transform @ frames)

    torch.testing.assert_close(original, moved, atol=1e-5, rtol=1e-5)


def test_structure_module_reuses_parameters_across_iterations():
    one_iteration = StructureModule(replace(tiny_config(), n_ipa=1))
    two_iterations = StructureModule(replace(tiny_config(), n_ipa=2))

    one_count = sum(parameter.numel() for parameter in one_iteration.parameters())
    two_count = sum(parameter.numel() for parameter in two_iterations.parameters())

    assert one_count == two_count


def test_structure_module_preserves_input_dtype():
    cfg = replace(tiny_config(), n_ipa=1)
    structure_module = StructureModule(cfg).double().eval()
    s = torch.randn(4, cfg.c_s, dtype=torch.float64)
    z = torch.randn(4, 4, cfg.c_z, dtype=torch.float64)

    with torch.no_grad():
        frames, refined_s = structure_module(s, z)

    assert frames.dtype == torch.float64
    assert refined_s.dtype == torch.float64


def test_rigid_transform_round_trip():
    rotation = quat_to_rot(torch.tensor([1.0, 0.2, -0.1, 0.3]))
    transform = make_T(rotation, torch.tensor([1.0, 2.0, 3.0]))
    point = torch.tensor([0.5, -0.2, 1.4])

    recovered = apply(invert(transform), apply(transform, point))

    assert torch.allclose(recovered, point, atol=1e-5)
