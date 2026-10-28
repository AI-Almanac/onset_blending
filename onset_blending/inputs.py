"""Finding a config's input files."""

from __future__ import annotations

import re
from pathlib import Path

from .config import BlendConfig, FileSet

_YEAR = re.compile(r"(19|20)\d{2}")


def file_year(path: str | Path) -> int | None:
    """The season a file holds: the first 19xx/20xx in its name."""
    match = _YEAR.search(Path(path).name)
    return int(match.group()) if match else None


def list_files(files: FileSet, config: BlendConfig, what: str) -> list[Path]:
    """The files a FileSet names, in year order."""
    if files.paths:
        paths = [config.path(p) for p in files.paths]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{what}: missing files: {', '.join(missing)}")
    else:
        folder = config.path(files.folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"{what}: {folder} is not a directory")
        pattern = re.compile(files.pattern)
        paths = [p for p in folder.iterdir() if p.is_file() and pattern.search(p.name)]
    if files.years:
        wanted = set(files.years)
        paths = [p for p in paths if file_year(p) in wanted]
    if not paths:
        raise FileNotFoundError(f"{what}: no files match {files}")
    return sorted(paths, key=lambda p: (file_year(p) or 0, str(p)))
