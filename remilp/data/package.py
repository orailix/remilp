"""Per-split raw.tar archives of the instances/ and labels/ trees."""

import tarfile
from pathlib import Path

RAW_ARCHIVE = "raw.tar"
RAW_DIRS = ("instances", "labels")


def split_dirs(root: Path) -> list[Path]:
    """Every directory under root holding a graphs/, instances/ or labels/ tree."""
    root = Path(root)
    found = {p.parent for name in ("graphs", *RAW_DIRS) for p in root.rglob(name)}
    found |= {p.parent for p in root.rglob(RAW_ARCHIVE)}
    return sorted(found)


def unpack(root: Path) -> list[Path]:
    """Extract every <split>/raw.tar under root next to itself."""
    extracted = []
    for archive in sorted(Path(root).rglob(RAW_ARCHIVE)):
        with tarfile.open(archive) as tar:
            tar.extractall(archive.parent, filter="data")
        extracted.append(archive)
    return extracted
