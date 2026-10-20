"""Compare two run folders file by file, to check that a code change left the results unchanged.

Usage::

    uv run python scripts/dev/compare_runs.py outputs/experiment/mini-golden outputs/experiment/mini

Every file under the reference folder must exist in the other one with the same content, and the
other folder must not have extra files. Content is compared by type:

- ``.npz`` and ``.npy``: same keys, shapes, dtypes and values (NaN equal to NaN);
- ``.pdf``: same pixels after rasterizing every page with ``pdftoppm`` (PDF bytes carry dates);
- ``.tex``: same text, except the lines of the macros that record when and from which revision
  the run was made (``VOLATILE_MACROS``) and durations such as ``0,3 s`` or ``12 min``;
- ``.csv``: same rows and values, except the columns that time the computation (``elapsed_s``);
- anything else (JSON, text): same bytes.

Apart from those timings, two runs of the same code give identical files (checked on the mini
run before the October refactoring).

Stage manifests (``_manifest.json``) record timings and versions, and the representations
(``reps/``) are skipped unless ``--include-reps`` is given. Tables listed in ``VOLATILE_FILES``
record durations and dates by design and are skipped as well. The exit status is 1 when any
file differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

SKIPPED_NAMES = {"_manifest.json"}
SKIPPED_SUFFIXES = {".aux", ".log", ".fls", ".fdb_latexmk", ".out", ".synctex.gz"}
VOLATILE_FILES = {"versoes.tex"}
VOLATILE_MACROS = ("\\dataExecucao", "\\revisaoGit")
DURATION = re.compile(r"\d+(?:,\d+)?\s?(?:s|min|h)\b")
TIMING_COLUMNS = re.compile(r"^elapsed")
RASTER_DPI = 50


def files_of(root: Path, include_reps: bool) -> Iterator[Path]:
    """Relative paths of the files to compare under ``root``, sorted."""

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.name in SKIPPED_NAMES or path.name in VOLATILE_FILES:
            continue
        if path.suffix in SKIPPED_SUFFIXES or path.name.endswith(".synctex.gz"):
            continue
        if not include_reps and relative.parts[0] == "reps":
            continue
        yield relative


def same_arrays(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def compare_npz(a: Path, b: Path) -> str | None:
    with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
        if sorted(left.files) != sorted(right.files):
            return f"keys {sorted(left.files)} != {sorted(right.files)}"
        for key in left.files:
            if not same_arrays(left[key], right[key]):
                return f"array '{key}' differs"
    return None


def compare_npy(a: Path, b: Path) -> str | None:
    left = np.load(a, mmap_mode="r", allow_pickle=False)
    right = np.load(b, mmap_mode="r", allow_pickle=False)
    return None if same_arrays(np.asarray(left), np.asarray(right)) else "array differs"


def rasterize(pdf: Path, out_dir: Path) -> list[np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / hashlib.sha1(str(pdf).encode()).hexdigest()
    subprocess.run(
        ["pdftoppm", "-r", str(RASTER_DPI), "-png", str(pdf), str(prefix)],
        check=True,
        capture_output=True,
    )
    pages = sorted(out_dir.glob(f"{prefix.name}*.png"))
    return [np.asarray(Image.open(page).convert("RGB")) for page in pages]


def compare_pdf(a: Path, b: Path) -> str | None:
    if a.read_bytes() == b.read_bytes():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        left = rasterize(a, Path(tmp) / "a")
        right = rasterize(b, Path(tmp) / "b")
    if len(left) != len(right):
        return f"{len(left)} pages != {len(right)} pages"
    for page, (x, y) in enumerate(zip(left, right, strict=True), start=1):
        if x.shape != y.shape:
            return f"page {page}: size {x.shape} != {y.shape}"
        changed = int((x != y).any(axis=-1).sum())
        if changed:
            return f"page {page}: {changed} pixels differ"
    return None


def stable_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not any(macro in line for macro in VOLATILE_MACROS)]
    return [DURATION.sub("<duration>", line) for line in kept]


def compare_tex(a: Path, b: Path) -> str | None:
    left, right = stable_lines(a), stable_lines(b)
    if left == right:
        return None
    for number, (x, y) in enumerate(zip(left, right, strict=False), start=1):
        if x != y:
            return f"line {number}: {x[:80]!r} != {y[:80]!r}"
    return f"{len(left)} lines != {len(right)} lines"


def stable_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return rows
    keep = [i for i, name in enumerate(rows[0]) if not TIMING_COLUMNS.match(name)]
    return [[row[i] for i in keep if i < len(row)] for row in rows]


def compare_csv(a: Path, b: Path) -> str | None:
    left, right = stable_rows(a), stable_rows(b)
    if left == right:
        return None
    if left and right and left[0] != right[0]:
        return f"columns {left[0]} != {right[0]}"
    for number, (x, y) in enumerate(zip(left, right, strict=False), start=1):
        if x != y:
            pairs = zip(left[0], x, y, strict=False)
            cells = [f"{h}: {u!r} != {v!r}" for h, u, v in pairs if u != v]
            return f"row {number}: " + "; ".join(cells[:3])
    return f"{len(left)} rows != {len(right)} rows"


def compare_bytes(a: Path, b: Path) -> str | None:
    if a.read_bytes() == b.read_bytes():
        return None
    left = a.read_bytes().splitlines()
    right = b.read_bytes().splitlines()
    for number, (x, y) in enumerate(zip(left, right, strict=False), start=1):
        if x != y:
            return f"line {number}: {x[:80]!r} != {y[:80]!r}"
    return f"{len(left)} lines != {len(right)} lines"


COMPARERS = {
    ".npz": compare_npz,
    ".npy": compare_npy,
    ".pdf": compare_pdf,
    ".tex": compare_tex,
    ".csv": compare_csv,
}


def compare(reference: Path, other: Path, include_reps: bool) -> list[str]:
    """Differences between the two folders, one message per file."""

    expected = set(files_of(reference, include_reps))
    found = set(files_of(other, include_reps))
    problems = [f"missing: {path}" for path in sorted(expected - found)]
    problems += [f"extra: {path}" for path in sorted(found - expected)]
    for path in sorted(expected & found):
        comparer = COMPARERS.get(path.suffix, compare_bytes)
        message = comparer(reference / path, other / path)
        if message:
            problems.append(f"differs: {path}: {message}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("other", type=Path)
    parser.add_argument("--include-reps", action="store_true")
    args = parser.parse_args(argv)
    problems = compare(args.reference, args.other, args.include_reps)
    n_files = sum(1 for _ in files_of(args.reference, args.include_reps))
    for problem in problems:
        print(problem)
    print(f"{n_files} files compared, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
