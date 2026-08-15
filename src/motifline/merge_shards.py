"""Concatenate sharded attribution outputs back into one attr/ohe/regions triple.

Shard i is expected in ``shard_dirs[i]`` under the names the attribution steps
write (``attr_<i>.npy``/``ohe_<i>.npy``/``regions_<i>.bed``), and the shards are
concatenated in that order -- shard order IS row order, because the shard bounds
are contiguous slices of the peak list. Passing the directories out of order, or
one directory twice, would produce a matrix whose rows no longer correspond to
regions.bed, so the index in the filename is checked against the position in the
list rather than trusted.

Runs in the orchestration env (numpy only), like every other stage body it is
invoked as a subprocess by the Snakefile.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def merge_shards(shard_dirs, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    attrs, ohes, beds, per_shard = [], [], [], []
    for i, d in enumerate(shard_dirs):
        d = Path(d)
        a, o, b = d / f"attr_{i}.npy", d / f"ohe_{i}.npy", d / f"regions_{i}.bed"
        for p in (a, o, b):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} missing: shard {i} must be the {i}-th directory given "
                    f"(got {d}); shard order is row order and is not inferred")
        A, O = np.load(a), np.load(o)
        rows = [ln for ln in b.read_text().splitlines() if ln.strip()]
        if not (len(A) == len(O) == len(rows)):
            raise ValueError(f"shard {i}: attr={len(A)} ohe={len(O)} regions={len(rows)} rows disagree")
        attrs.append(A)
        ohes.append(O)
        beds.extend(rows)
        per_shard.append({"shard": i, "dir": str(d), "rows": int(len(A))})
    attr = np.concatenate(attrs, 0)
    ohe = np.concatenate(ohes, 0)
    np.save(out / "attr.npy", attr)
    np.save(out / "ohe.npy", ohe)
    (out / "regions.bed").write_text("".join(r + "\n" for r in beds))
    return {"n_shards": len(shard_dirs), "rows": int(len(attr)),
            "width": int(attr.shape[-1]), "per_shard": per_shard}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", action="append", required=True,
                    help="repeat once per shard, in shard order 0..n-1")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--expected_rows", type=int, default=None,
                    help="refuse to write a merge whose row count is not this")
    a = ap.parse_args()
    rep = merge_shards(a.shard_dir, a.out_dir)
    if a.expected_rows is not None and rep["rows"] != a.expected_rows:
        raise SystemExit(f"merged {rep['rows']} rows, expected {a.expected_rows}")
    print(f"MERGED {rep['rows']} rows x {rep['width']}bp from {rep['n_shards']} shards", flush=True)


if __name__ == "__main__":
    main()
