"""The mmcv stand-in FG² runs on: mmcv's parameter layout and the sampling convention of the reference forward."""
import pytest
import torch

from bevloc.baselines import fg2 as fg2_wrap


@pytest.fixture(scope="module")
def shim():
    fg2_wrap.ensure_fg2_on_path()
    from mmcv.cnn.bricks.transformer import FFN
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention, multi_scale_deformable_attn_pytorch
    return FFN, MultiScaleDeformableAttention, multi_scale_deformable_attn_pytorch


def test_ffn_layout_matches_mmcv(shim):
    FFN = shim[0]
    ffn = FFN(32)
    assert set(ffn.state_dict()) == {"layers.0.0.weight", "layers.0.0.bias", "layers.1.weight", "layers.1.bias"}
    assert ffn.layers[0][0].out_features == 1024          # mmcv default feedforward_channels
    x = torch.randn(2, 5, 32)
    assert ffn(x).shape == x.shape
    assert FFN(32, add_identity=False)(x).shape == x.shape


def test_deformable_attention_layout_and_residual(shim):
    MSDA = shim[1]
    m = MSDA(embed_dims=16, num_heads=2, num_levels=1, num_points=4, batch_first=True).eval()
    assert set(m.state_dict()) == {f"{n}.{p}" for n in ("sampling_offsets", "attention_weights", "value_proj", "output_proj")
                                   for p in ("weight", "bias")}
    H, W = 3, 4
    q = torch.randn(2, H * W, 16)
    ref = torch.rand(2, H * W, 1, 2)
    out = m(query=q, value=q, reference_points=ref, spatial_shapes=torch.tensor([[H, W]]), level_start_index=torch.tensor([0]))
    assert out.shape == q.shape
    # with zero projections the module is the identity (residual path)
    with torch.no_grad():
        for lin in (m.output_proj,):
            lin.weight.zero_(); lin.bias.zero_()
    torch.testing.assert_close(m(query=q, value=q, reference_points=ref, spatial_shapes=torch.tensor([[H, W]]),
                                 level_start_index=torch.tensor([0])), q)


def test_reference_forward_samples_pixel_centres(shim):
    """Sampling at (j+0.5)/W, (i+0.5)/H with zero offsets returns that pixel's value: the [0, 1] -> grid_sample
    (align_corners=False) convention the checkpoints were trained with."""
    fn = shim[2]
    H, W, heads, d = 3, 4, 2, 6
    value = torch.randn(1, H * W, heads, d)
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    ref = torch.stack([(xs.flatten() + 0.5) / W, (ys.flatten() + 0.5) / H], -1)       # (HW, 2) as (x, y)
    loc = ref[None, :, None, None, None, :].expand(1, H * W, heads, 1, 1, 2)
    w = torch.ones(1, H * W, heads, 1, 1)
    out = fn(value, torch.tensor([[H, W]]), loc, w)
    torch.testing.assert_close(out, value.reshape(1, H * W, heads * d), atol=1e-5, rtol=0)


@pytest.mark.skipif(not (fg2_wrap.CKPT_ROOT / "samearea" / "known_ori" / "model.pt").is_file()
                    or not (fg2_wrap.FG2_ROOT / "models").is_dir(),
                    reason="FG² checkpoint or third_party/FG2 not present")
def test_released_checkpoint_loads_strictly_through_shim():
    model, meta = fg2_wrap.load_cvm(torch.device("cpu"), area="samearea", orientation="known_ori")
    assert meta["sha256"] and sum(p.numel() for p in model.parameters()) > 1e6
