import numpy as np
import torch

from bevloc.bev.grid import BevGrid
from bevloc.bev.spherical import lidar_to_erp
from bevloc.data.ortho import Oriented, gt_homography
from bevloc.model.coarse import coarse_loss, coarse_targets, ref_cell_validity, roma_coarse_loss, window_logits
from bevloc.model.fusion_bev import CELL_CHANNELS, LIDAR_CHANNELS, BevEncoder, PointPaintSplat


def test_coarse_targets_follow_the_gt_homography():
    """Query centred in a same-GSD 4x reference, rotated 90 deg: patch centres must land where
    the Oriented model says, in row * K + col order."""
    q = Oriented((1000.0, 2000.0), 30.0, 224, 0.25)
    r = Oriented((1010.0, 1995.0), 120.0, 896, 0.25)
    H = torch.tensor(gt_homography(q, r), dtype=torch.float32)[None]
    idx, use = coarse_targets(H, torch.ones(1, 14, 14, dtype=torch.bool))
    assert use.all()
    for i, j in ((0, 0), (3, 11), (13, 13)):
        world = q.px_to_world @ np.array([16 * j + 7.5, 16 * i + 7.5, 1.0])
        u, v, _ = np.linalg.inv(r.px_to_world) @ world
        assert int(idx[0, i, j]) == int((v + 0.5) // 16) * 56 + int((u + 0.5) // 16)


def test_coarse_targets_mask_outside_and_invalid():
    H = torch.eye(3)[None].clone()
    H[0, 0, 2] = 896 - 100.0                       # shift right: most of the query leaves the reference
    valid = torch.ones(1, 14, 14, dtype=torch.bool)
    valid[0, 0, 0] = False
    _, use = coarse_targets(H, valid)
    assert not use[0, 0, 0] and use[0, 5, 0] and not use[0, 5, 13]


def test_ref_cell_validity_flags_black_cells():
    ref = torch.ones(1, 3, 224, 224)          # 224 / cells=14 -> 16 px cells, matching patch=16 elsewhere
    ref[:, :, :16, :16] = 0                    # cell (0, 0) fully black
    ref[:, :, :16, 16:32] = 0
    ref[0, :, :16, 32:40] = 0                  # cell (0, 2) half black -> still valid at min_frac=0.5
    rv = ref_cell_validity(ref, cells=14, min_frac=0.5)
    assert rv.shape == (1, 14, 14)
    assert not rv[0, 0, 0] and not rv[0, 0, 1] and rv[0, 0, 2] and rv[0, 1, 0]


def test_coarse_targets_masks_only_patches_landing_on_black_cells():
    """A hole in ONE reference cell must drop only the query patches that target it, not the frame."""
    H = torch.eye(3)[None]                    # identity: query patch (i, j) -> reference cell (i, j)
    valid = torch.ones(1, 14, 14, dtype=torch.bool)
    ref_valid = torch.ones(1, 56, 56, dtype=torch.bool)
    ref_valid[0, 3, 5] = False                # one hole, hit by query patch (3, 5) under identity H
    idx, use = coarse_targets(H, valid, ref_valid=ref_valid)
    assert not use[0, 3, 5]
    assert use[0, 0, 0] and use[0, 13, 13] and use.sum() == 14 * 14 - 1


def test_coarse_loss_is_zero_at_the_target_and_counts_cells():
    idx = torch.randint(0, 56 * 56, (2, 14, 14))
    use = torch.ones(2, 14, 14, dtype=torch.bool)
    gm = torch.full((2, 56 * 56, 14, 14), -20.0)
    gm.scatter_(1, idx[:, None], 20.0)
    loss, st = coarse_loss(gm, idx, use)
    assert loss < 1e-4 and st["acc"] == 1.0 and st["cell_err"] == 0.0
    loss0, st0 = coarse_loss(gm, idx, torch.zeros_like(use))          # nothing to supervise: no NaN
    assert float(loss0) == 0.0 and st0["n"] == 0


def test_roma_coarse_loss_ignores_unmatchable_patches_and_trains_certainty():
    idx = torch.zeros(1, 2, 2, dtype=torch.long)
    matchable = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.bool)
    gm = torch.full((1, 4, 2, 2), -20.0)
    gm[:, 0] = 20.0                                          # every patch predicts class 0
    cert = torch.full((1, 1, 2, 2), -20.0)                   # certainty says "not matchable"
    loss, st = roma_coarse_loss(gm, idx, matchable, cert, certainty_weight=0.01)
    assert st["n"] == 1 and st["acc"] == 1.0 and st["top5"] == 1.0
    assert st["ce"] < 1e-3
    # The one matchable patch is classified perfectly, so the loss is the certainty term.
    assert abs(float(loss) - 0.01 * st["cert"]) < 1e-4
    assert st["cert"] > 1.0                                  # logits say no, target says yes on one patch


def test_neighbour_hinge_is_zero_when_the_true_cell_already_wins_locally():
    from bevloc.model.coarse import neighbour_hinge
    logits = torch.full((1, 9), -5.0)
    logits[0, 4] = 5.0
    assert float(neighbour_hinge(logits, torch.tensor([4]), radius=1)) == 0.0
    logits[0, 4] = -5.0
    logits[0, 5] = 5.0
    assert float(neighbour_hinge(logits, torch.tensor([4]), radius=1, margin=1.0)) > 1.0


def test_window_logits_keeps_only_the_neighbourhood_of_the_target():
    logits = torch.zeros(1, 9)          # 3x3 grid, target is the centre class 4
    masked = window_logits(logits, torch.tensor([4]), radius=0)
    assert (masked[0] == 0).sum() == 1 and (masked[0] < -1e3).sum() == 8
    masked = window_logits(logits, torch.tensor([4]), radius=1)
    assert (masked[0] == 0).all()


def test_points_land_in_the_bevgrid_cell_and_sample_the_right_pixel():
    """The torch lift must agree with the numpy BevGrid and with lidar_to_erp."""
    g, W, H = BevGrid(224, 0.25), 64, 32
    t_cl = np.array([0.0, 0.0, -0.27])
    lift = PointPaintSplat(dim=2, n=g.n, cell=g.cell, lidar_height=1.57, t_cl=t_cl, ground_fill=False)
    pts = np.array([[10.1, 3.3, -1.0], [-6.2, -12.4, 0.5], [0.4, 20.9, 2.0]])
    u, v, _ = lidar_to_erp(pts, W, H, None, t_cl)
    feat = torch.zeros(1, 2, H, W)
    feat[0, 0] = torch.arange(W).float()[None, :] + 0.5               # channel 0 = u, channel 1 = v
    feat[0, 1] = torch.arange(H).float()[:, None] + 0.5
    bev, mask = lift(feat, torch.ones(1, 1, H, W), [torch.tensor(pts, dtype=torch.float32)], [torch.zeros(3)])
    assert bev.shape == (1, 2 + LIDAR_CHANNELS + CELL_CHANNELS, g.n, g.n) and int(mask.sum()) == 3
    rr, cc, ok = g.to_cell(pts[:, 0], pts[:, 1])
    assert ok.all()
    for k in range(3):
        assert mask[0, 0, rr[k], cc[k]] == 1
        assert abs(float(bev[0, 0, rr[k], cc[k]]) - u[k]) < 1e-2 and abs(float(bev[0, 1, rr[k], cc[k]]) - v[k]) < 1e-2
        assert abs(float(bev[0, 2, rr[k], cc[k]]) * 3.0 - (pts[k, 2] + 1.57)) < 1e-5     # height above road


def test_masked_pixels_contribute_no_image_feature():
    lift = PointPaintSplat(dim=2, n=224, cell=0.25, ground_fill=False)
    p = [torch.tensor([[8.0, 0.0, -1.0]])]
    bev, _ = lift(torch.ones(1, 2, 32, 64), torch.zeros(1, 1, 32, 64), p, [torch.zeros(1)])
    cell = bev[0, :, 80, 112]                                         # x = 8 m -> row floor(112 - 32) = 80
    assert cell[:2].abs().sum() == 0 and cell[2] != 0 and cell[5] == 0        # no image, LiDAR kept, "seen" = 0


def test_forward_multiframe_matches_single_frame_at_identity_pose():
    """One source frame, identity relative pose: forward_multiframe must reduce exactly to
    what the single-frame `forward` computes for that one frame."""
    torch.manual_seed(0)
    lift = PointPaintSplat(dim=2, n=64, cell=0.25, ground_fill=False)
    feat = torch.randn(1, 2, 32, 64)
    erp_valid = (torch.rand(1, 1, 32, 64) > 0.1).float()
    xyz = torch.tensor([[5.0, 1.0, -1.0], [8.0, -2.0, 0.5], [-3.0, 4.0, -1.2]])
    refl = torch.rand(3) * 255

    bev1, mask1 = lift(feat, erp_valid, [xyz], [refl])
    bevN, maskN = lift.forward_multiframe(
        [dict(feat=feat[0], erp_valid=erp_valid, xyz_cam=xyz, xyz_bev=xyz, refl=refl)])
    assert torch.allclose(bev1, bevN, atol=1e-5) and torch.equal(mask1, maskN)


def test_forward_multiframe_places_points_by_the_relative_pose_not_camera_frame():
    """A point observed by a SECOND frame, transformed into the query's BEV frame, must land in
    the BEV cell implied by xyz_bev -- while still sampling its image feature from xyz_cam."""
    lift = PointPaintSplat(dim=2, n=64, cell=0.25, ground_fill=False)
    feat_q = torch.zeros(2, 32, 64)
    feat_o = torch.ones(2, 32, 64) * 5.0                  # a different, distinguishable frame
    erp_valid = torch.ones(1, 1, 32, 64)
    query = dict(feat=feat_q, erp_valid=erp_valid, xyz_cam=torch.tensor([[2.0, 0.0, -1.0]]),
                xyz_bev=torch.tensor([[2.0, 0.0, -1.0]]), refl=torch.tensor([100.0]))
    # a point that frame "other" sees 3 m ahead of ITS OWN origin, but that origin is 3 m ahead
    # of the query (n=64, cell=0.25 -> +-8 m grid) -> in the query's frame it lands at x ~= 6 m,
    # a different BEV cell, while its image sample must still come from its own 3 m projection
    other = dict(feat=feat_o, erp_valid=erp_valid, xyz_cam=torch.tensor([[3.0, 0.0, -1.0]]),
                xyz_bev=torch.tensor([[6.0, 0.0, -1.0]]), refl=torch.tensor([50.0]))
    bev, mask = lift.forward_multiframe([query, other])

    rq, cq = lift.cell_index(query["xyz_bev"])
    ro, co = lift.cell_index(other["xyz_bev"])
    assert rq[0] != ro[0]
    flat = bev[0].reshape(bev.shape[1], -1)
    assert torch.allclose(flat[:2, rq[0]], torch.zeros(2), atol=1e-4)     # query's own (zero) feature
    assert torch.allclose(flat[:2, ro[0]], torch.full((2,), 5.0), atol=1e-4)  # other frame's feature, at ITS cell
    assert mask[0, 0].reshape(-1)[rq[0]] > 0 and mask[0, 0].reshape(-1)[ro[0]] > 0


def test_ground_fill_stops_at_the_first_obstacle_and_skips_masked_pixels():
    """Camera-visible ground: cells in front of a wall are filled, cells behind it are not, and
    nothing is filled where the ERP is masked (ego vehicle)."""
    lift = PointPaintSplat(dim=2, n=64, cell=0.25, lidar_height=1.5)
    ys = torch.linspace(-8, 8, 400)
    wall = torch.stack([torch.full_like(ys, 4.0), ys, torch.full_like(ys, -0.5)], 1)   # 1 m high, 4 m ahead
    g = lift.ground_points(wall, torch.ones(1, 1, 32, 64))
    ahead = g[(g[:, 1].abs() < 1.0) & (g[:, 0] > 0)]
    assert len(ahead) and float(ahead[:, 0].max()) < 4.0 - 0.4                         # stops before the wall
    assert (g[:, 0] < -2).any()                                                        # free behind the vehicle
    assert torch.allclose(g[:, 2], torch.full_like(g[:, 2], -1.5))
    assert len(lift.ground_points(wall, torch.zeros(1, 1, 32, 64))) == 0               # all masked -> nothing

    bev, mask = lift(torch.ones(1, 2, 32, 64), torch.ones(1, 1, 32, 64), [wall], [torch.zeros(len(wall))])
    assert float(mask.mean()) > 5 * len(torch.unique(lift.cell_index(wall)[0])) / 64 ** 2   # far more cells than LiDAR alone
    is_lidar = bev[0, 2 + LIDAR_CHANNELS - 1]
    r, _ = lift.cell_index(torch.tensor([[2.0, 0.0, -1.5]]))
    assert float(is_lidar.reshape(-1)[r[0]]) == 0.0                                    # a pure ground-fill cell


def test_pca_rgb_shares_one_scale_across_inputs():
    import sys
    sys.path.insert(0, str((__import__("pathlib").Path(__file__).parents[1] / "scripts")))
    from viz_features import pca_rgb
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4, 8)).astype(np.float32)
    b = rng.normal(size=(6, 6, 8)).astype(np.float32) * 5 + 3     # different scale/offset
    ra, rb = pca_rgb(a, b)
    assert ra.shape == (4, 4, 3) and rb.shape == (6, 6, 3)
    assert ra.dtype == np.uint8 and rb.dtype == np.uint8
    # a lone call on `a` alone must generally NOT match the joint-fit colours (different PCA basis)
    (ra_alone,) = pca_rgb(a)
    assert not np.array_equal(ra, ra_alone)


def test_bev_encoder_emits_matcher_tokens():
    enc = BevEncoder(cin=8, dims=(16, 16, 32, 32), out_dim=64, n_attn=1)
    out = enc(torch.randn(2, 8, 224, 224), torch.ones(2, 1, 224, 224))
    assert out.shape == (2, 64, 14, 14)
    assert abs(float(out.mean())) < 0.2 and 0.5 < float(out.std()) < 1.5      # LayerNorm'ed like ViT tokens
