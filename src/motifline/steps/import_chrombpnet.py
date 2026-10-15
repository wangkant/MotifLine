#!/usr/bin/env python
"""Import official (TensorFlow) ChromBPNet contribution scores as a run's attr/.

Reads the `<prefix>.counts_scores.h5` (or `.profile_scores.h5`) that chrombpnet's
interpret step writes with deepdish, using PyTables: /raw/seq is the one-hot input and
/shap/seq the hypothetical scores, both (N, 4, L). Rows follow
`<prefix>.interpreted_regions.bed`.

Before anything is written: /shap must be nonzero off the observed base (hypothetical,
not projected), /projected_shap must equal raw * shap, and every row's one-hot must
equal the genome at start + summit ± L/2. chrombpnet keeps peaks with N (all-zero columns); they are kept
here too and counted. Scores are float16 in the file; attr.npy is float32.

Runs in the compute env and must not import motifline.
"""
import argparse
import json
import os

import numpy as np


def encode(seq):
    """(4, L) int8 like chrombpnet's dna_to_one_hot: upper-cased, non-ACGT all-zero."""
    s = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    out = np.zeros((4, len(s)), np.int8)
    for i, b in enumerate(b"ACGT"):
        out[i, s == b] = 1
    return out


def read_regions(path):
    rows = []
    for i, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) != 10:
            raise SystemExit(f"{path} line {i}: {len(cols)} columns, expected the "
                             f"10-column interpreted_regions.bed")
        rows.append((cols[0], int(cols[1]), int(cols[9]), line.rstrip("\n")))
    return rows


def mismatches(raw, rows, fetch, width):
    """Rows whose one-hot is not the genome at start + summit ± width/2."""
    bad = []
    for i, (chrom, start, summit, _) in enumerate(rows):
        c = start + summit
        if not np.array_equal(raw[i], encode(fetch(chrom, c - width // 2, c + width // 2))):
            bad.append(i)
    return bad


def check(raw, shap, projected, rows, expected_rows, expected_width):
    if raw.ndim != 3 or raw.shape[1] != 4 or raw.shape != shap.shape:
        raise SystemExit(f"raw/seq {raw.shape} and shap/seq {shap.shape} are not matching (N, 4, L)")
    n, _, width = raw.shape
    if len(rows) != n:
        raise SystemExit(f"{n} score rows but {len(rows)} regions")
    if n != expected_rows:
        raise SystemExit(f"expected {expected_rows} rows, the scores have {n}")
    if expected_width is not None and width != expected_width:
        raise SystemExit(f"expected width {expected_width}, the scores have {width}")
    if not np.isin(raw, (0, 1)).all() or (raw.sum(1) > 1).any():
        raise SystemExit("raw/seq is not one-hot")
    k = min(n, 256)
    if not (shap[:k][raw[:k] == 0] != 0).any():
        raise SystemExit("shap/seq is zero wherever a base is absent: these are projected "
                         "scores, not hypothetical ones")
    if projected is not None and             not np.array_equal(projected[:k], (raw[:k] * shap[:k]).astype(projected.dtype)):
        raise SystemExit("projected_shap is not raw * shap: the file is inconsistent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="<prefix>.counts_scores.h5")
    ap.add_argument("--regions", required=True, help="<prefix>.interpreted_regions.bed")
    ap.add_argument("--fasta", required=True, help="the genome chrombpnet read")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--expected_rows", type=int, required=True)
    ap.add_argument("--expected_width", type=int, default=None)
    a = ap.parse_args()
    import pyfaidx
    import tables

    with tables.open_file(a.scores) as f:
        raw = f.get_node("/raw/seq").read()
        shap = f.get_node("/shap/seq").read()
        projected = f.get_node("/projected_shap/seq").read() if "/projected_shap/seq" in f else None
    rows = read_regions(a.regions)
    check(raw, shap, projected, rows, a.expected_rows, a.expected_width)
    width = raw.shape[-1]

    fasta = pyfaidx.Fasta(a.fasta)
    bad = mismatches(raw, rows, lambda c, s, e: fasta[c][s:e].seq, width)
    if bad:
        raise SystemExit(f"{len(bad)} rows do not match {a.fasta}; first is row {bad[0]}: "
                         f"{rows[bad[0]][3]}")

    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "attr.npy"), shap.astype(np.float32))
    np.save(os.path.join(a.out_dir, "ohe.npy"), raw.astype(np.int8))
    with open(os.path.join(a.out_dir, "regions.bed"), "w") as fh:
        fh.writelines(r[3] + "\n" for r in rows)
    print(json.dumps({"rows": int(raw.shape[0]), "width": int(width),
                      "scores": os.path.basename(a.scores), "source_dtype": str(shap.dtype),
                      "rows_with_non_acgt": int((raw.sum(1) == 0).any(axis=1).sum()),
                      "checked_against_genome": True}), flush=True)


if __name__ == "__main__":
    main()
