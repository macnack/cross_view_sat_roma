"""Test helper: VIGOR label folders as the reader requires them (bevloc.data.vigor.check_corrected)."""
from pathlib import Path


def corrected(lab_dir):
    """Write <name>__corrected.txt next to every plain label file of lab_dir (a <City> folder or a label root), as
    SliceMatch's splits__corrected holds them; the plain copies stay (what FG²'s / Loc²'s loaders read). Re-run after
    editing a plain file. Returns lab_dir."""
    lab_dir = Path(lab_dir)
    for f in list(lab_dir.rglob("*.txt")):
        if f.name == "satellite_list.txt" or f.stem.endswith("__corrected"):
            continue
        f.with_name(f.stem + "__corrected.txt").write_bytes(f.read_bytes())
    return lab_dir
