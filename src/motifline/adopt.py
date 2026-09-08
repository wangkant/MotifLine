"""Adopt an archived attribution directory into a motifline stage tree.

The archived ``interpret_{cbp,ag}_par_hyp/`` products cost GPU-days and the old
tree is read-only, so adoption links rather than copies: ``attr.npy`` and
``ohe.npy`` become absolute symlinks and only the normalized ``regions.bed`` is
newly written. Nothing is written until every shape check passes, so a refused
adoption leaves an empty stage directory rather than a half-linked one.

Nothing is written until every shape check passes; a check that fails raises
``ValueError`` and the stage directory is left as it was.

``normalize_peaks`` exists because the two archived peak formats disagree: the
ChromBPNet side stores the 11-column ``load_region_df`` output while the AG side
stores a 4-column ``chr/start/end/summit`` bed, and ``finemo
extract-regions-modisco-fmt -p`` needs narrowPeak-shaped columns. The expansion
performed here is exactly the one the archive itself performed when it built
``peaks_np.bed`` (region_v3/gm12878/run_interpret_ag.sh:50-53).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

_NARROWPEAK_FILLER = ["."] * 6


def _npy_shape(path):
    return np.load(path, mmap_mode="r").shape


def _read_bed(path):
    with open(path) as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def normalize_peaks_text(src) -> dict:
    """Parse and validate ``src``, returning the text to write -- writes nothing.

    Split out of ``normalize_peaks`` so a caller can fail on a malformed bed before it
    creates anything on disk.
    """
    lines = _read_bed(src)
    widths = {ln.count("\t") + 1 for ln in lines}
    if len(widths) > 1:
        raise ValueError(f"{src}: ragged bed, saw {sorted(widths)} columns")
    ncol = widths.pop() if widths else 0
    if ncol == 4:
        out = []
        for ln in lines:
            chrom, start, end, summit = ln.split("\t")
            out.append("\t".join([chrom, start, end, *_NARROWPEAK_FILLER, summit, "True"]))
        text = "".join(x + "\n" for x in out)
        expanded = True
    elif ncol in (10, 11):
        text = "".join(ln + "\n" for ln in lines)
        expanded = False
    else:
        raise ValueError(f"{src}: expected 4, 10 or 11 columns, saw {ncol}")
    return {"text": text, "rows": len(lines), "src_columns": ncol,
            "expanded_from_summit_bed": expanded}


def write_peaks(parsed: dict, dst) -> dict:
    """Write what ``normalize_peaks_text`` parsed."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(parsed["text"])
    return {k: v for k, v in parsed.items() if k != "text"}


def normalize_peaks(src, dst) -> dict:
    """Validate and write in one call (the shape every existing caller expects)."""
    return write_peaks(normalize_peaks_text(src), dst)


def adopt_attr(src_dir, out_dir, expected_rows: int, expected_width: int | None = None,
               peaks_name: str = "peaks.bed", attr_name: str = "attr.npy",
               ohe_name: str = "ohe.npy") -> dict:
    """Link an archived attr/ohe pair into ``out_dir`` and normalize its peaks bed."""
    src, out = Path(src_dir), Path(out_dir)
    attr_src, ohe_src = (src / attr_name).resolve(), (src / ohe_name).resolve()
    peaks_src = (src / peaks_name).resolve()
    for p in (attr_src, ohe_src, peaks_src):
        if not p.exists():
            raise FileNotFoundError(f"{p}: archived input missing, nothing adopted")

    sa, so = _npy_shape(attr_src), _npy_shape(ohe_src)
    n_bed = len(_read_bed(peaks_src))
    if not (sa[0] == so[0] == n_bed):
        raise ValueError(f"row drift: attr={sa[0]} ohe={so[0]} peaks={n_bed} in {src}")
    if sa[0] != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, archive has {sa[0]} in {src}")
    if sa[-1] != so[-1]:
        raise ValueError(f"width drift: attr={sa[-1]} ohe={so[-1]} in {src}")
    if expected_width is not None and sa[-1] != expected_width:
        raise ValueError(f"width drift: attr={sa[-1]} ohe={so[-1]} expected={expected_width}")

    out.mkdir(parents=True, exist_ok=True)
    # Normalise the bed FIRST. It is the step that can still refuse (a ragged bed, an
    # unsupported column count), and doing it after the symlinks left attr.npy and
    # ohe.npy linked with no regions.bed beside them -- a half-adopted directory the
    # docstring promises cannot exist. Inside the DAG snakemake cleans up after a failed
    # job; the documented direct CLI has no such backstop.
    bed_text = normalize_peaks_text(peaks_src)
    for name, target in ((attr_name, attr_src), (ohe_name, ohe_src)):
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    bed_info = write_peaks(bed_text, out / "regions.bed")
    return {"rows": int(sa[0]), "width": int(sa[-1]), "peaks": bed_info,
            "sources": {"attr": str(attr_src), "ohe": str(ohe_src), "peaks": str(peaks_src)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src_dir", required=True, help="archived interpret_*_par_hyp directory")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--expected_rows", type=int, required=True)
    ap.add_argument("--expected_width", type=int, default=None,
                    help="optional; the array's own width is the width (geometry.py removed 2026-09-07)")
    ap.add_argument("--peaks_name", default="peaks.bed")
    a = ap.parse_args()
    info = adopt_attr(a.src_dir, a.out_dir, a.expected_rows, a.expected_width, a.peaks_name)
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
