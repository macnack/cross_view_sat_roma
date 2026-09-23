"""Mapillary pairs helpers (no network, no GPU)."""
from __future__ import annotations

import pytest


def test_poznan_tiles_uses_env_root(tmp_path, monkeypatch):
    from bevloc.data.mapillary import poznan_tiles, sat_data_root
    monkeypatch.setenv("SAT_DATA_DIR", str(tmp_path))
    assert sat_data_root() == tmp_path
    for i in range(9):
        d = tmp_path / f"geoportal_poznan_15km2_e{i}_n{i}_gmix"
        d.mkdir()
        (d / "year_2025.tif").write_bytes(b"")
    assert len(poznan_tiles(2025)) == 9
    with pytest.raises(FileNotFoundError):
        poznan_tiles(2024)
