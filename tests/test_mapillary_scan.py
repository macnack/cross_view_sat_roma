"""Tiling and 360 filtering for the Mapillary rectangle scan. No network."""

import mapillary_dl as m

# OSM city boundary of Poznań, the default `make mapillary-scan` box.
POZNAN = (16.7315878, 52.2919238, 17.0717065, 52.5093282)


def test_even_sample_includes_endpoints():
    assert m.even_sample(list(range(5)), 12) == [0, 1, 2, 3, 4]
    picked = m.even_sample(list(range(100)), 12)
    assert picked[0] == 0 and picked[-1] == 99
    assert len(picked) == 12
    assert picked == sorted(set(picked))


def test_longest_sequence_per_user():
    seqs = [
        {"username": "A", "n_in_bbox": 10, "path_length_m": 1, "sequence": "short"},
        {"username": "A", "n_in_bbox": 40, "path_length_m": 2, "sequence": "long"},
        {"username": "B", "n_in_bbox": 3, "path_length_m": 1, "sequence": "b"},
    ]
    picks = m.longest_sequence_per_user(seqs)
    assert [p["sequence"] for p in picks] == ["long", "b"]


def test_split_until_covers_poznan_under_api_limit():
    tiles = m.split_until(POZNAN)
    assert len(tiles) >= 2
    assert all(m.tile_area(t) <= m.MAX_BBOX_AREA + 1e-12 for t in tiles)
    assert abs(sum(m.tile_area(t) for t in tiles) - m.tile_area(POZNAN)) < 1e-9


def test_group_sequences_keeps_spherical_and_skips_gaps():
    images = {
        "1": _im("1", "seq", "spherical", 1, [16.90, 52.40]),
        "2": _im("2", "seq", "spherical", 2, [16.9004, 52.40]),
        "3": _im("3", "seq", "spherical", 3, [17.20, 52.40]),  # ~20 km jump, clipped drive
        "4": _im("4", "persp", "perspective", 1, [16.90, 52.40]),
    }
    rows, dropped, n_360 = m.group_sequences(images)
    assert n_360 == 3
    assert dropped == {"perspective": 1}
    assert len(rows) == 1 and rows[0]["sequence"] == "seq"
    assert rows[0]["n_in_bbox"] == 3
    # The short step is a few tens of metres; the degree-scale jump is not added.
    assert 10 < rows[0]["path_length_m"] < 100


def _im(iid, seq, camera, t, lonlat):
    return {
        "id": iid,
        "sequence": seq,
        "camera_type": camera,
        "captured_at": t,
        "computed_geometry": {"coordinates": lonlat},
        "creator": {"username": "Fixtor", "id": "1"},
        "make": "GoPro",
        "model": "GoPro Max",
    }
