"""scripts/viz_panoroma_matches.py: frame picking by percentile and the per-token inlier status."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from viz_panoroma_matches import percentile_of, pick_rows, safe_name, token_status  # noqa: E402


def _rows(errs):
    return [dict(id=f"Chicago/p{k},1,2,.jpg", pose_peak_m=e, pose_fine_gated_m=e) for k, e in enumerate(errs)]


def test_pick_quantiles_takes_rank_of_final_error():
    rows = _rows([float(e) for e in np.random.default_rng(0).permutation(101)])   # errors 0..100 m
    picks = pick_rows(rows, "quantiles", (5, 50))
    assert [r["pose_fine_gated_m"] for r, _ in picks] == [5.0, 50.0]
    assert [round(p) for _, p in picks] == [5, 50]


def test_pick_quantiles_distinct_and_none_is_worst():
    rows = _rows([1.0, None, 2.0])
    picks = pick_rows(rows, "quantiles", (0, 0))
    assert len({r["id"] for r, _ in picks}) == 2
    assert percentile_of(rows, rows[1]["id"]) == 100.0


def test_pick_ids_keeps_order_and_reports_percentile():
    rows = _rows([3.0, 1.0, 2.0])
    picks = pick_rows(rows, "ids", ids=[rows[0]["id"], rows[1]["id"]])
    assert [r["id"] for r, _ in picks] == [rows[0]["id"], rows[1]["id"]]
    assert [p for _, p in picks] == [100.0, 0.0]


def test_token_status_any_inlier_mode():
    tok = np.array([[0, 0], [0, 0], [2, 1], [3, 1]])                  # (col, row); token (0,0) has two modes
    inl = np.array([False, True, False, True])
    st = token_status(2, 4, tok, inl)
    assert st[0, 0] == 1 and st[1, 2] == 0 and st[1, 3] == 1 and st.sum() == 2


def test_safe_name():
    assert safe_name("Chicago/abc_D,41.8,-87.6,.jpg") == "Chicago_abc_D"
