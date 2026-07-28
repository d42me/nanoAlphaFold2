import torch

from af2_from_scratch import AF2Config, AlphaFold2FromScratch
from af2_from_scratch.feature_extraction import parse_seq
from af2_from_scratch.geometry import apply, invert, make_T, quat_to_rot


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


def test_a3m_insertions_are_counted_and_removed():
    sequence, deletion_counts = parse_seq("PIAxqIHsdILEGR")

    assert sequence == "PIAIHILEGR"
    assert deletion_counts == [0, 0, 0, 2, 2, 4, 4, 4, 4, 4]


def test_rigid_transform_round_trip():
    rotation = quat_to_rot(torch.tensor([1.0, 0.2, -0.1, 0.3]))
    transform = make_T(rotation, torch.tensor([1.0, 2.0, 3.0]))
    point = torch.tensor([0.5, -0.2, 1.4])

    recovered = apply(invert(transform), apply(transform, point))

    assert torch.allclose(recovered, point, atol=1e-5)
