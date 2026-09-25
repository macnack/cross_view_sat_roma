"""The Loc² baseline wrapper: released checkpoint loads strictly through the mmcv shim; depth-file layout."""
from pathlib import Path

import pytest
import torch

from bevloc.baselines import loc2 as loc2_wrap


def test_depth_png_path_matches_loc2_dataloader():
    # Loc² dataloader: ground path with 'panorama' -> 'unik3d_depth', extension -> .png
    p = loc2_wrap.depth_png_path("/data/vigor", "Chicago", "panorama_id_001.jpg")
    assert p == Path("/data/vigor/Chicago/unik3d_depth/panorama_id_001.png")


@pytest.mark.skipif(not (loc2_wrap.CKPT_ROOT / "samearea" / "known_ori" / "model.pt").is_file()
                    or not (loc2_wrap.LOC2_ROOT / "models").is_dir(),
                    reason="Loc² checkpoint or third_party/Loc2 not present")
def test_released_checkpoint_loads_strictly_through_shim():
    model, meta = loc2_wrap.load_matcher(torch.device("cpu"), area="samearea", orientation="known_ori")
    assert meta["sha256"] and sum(p.numel() for p in model.parameters()) > 1e6
