"""Concatenate attribution shards, in shard order, into one attr/ohe/regions triple.

Shard i must be the i-th directory given: shard order is row order.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def _side_files(shard_dirs, name, ext):
    """Every shard's <name>_<i>.<ext>, or [] when no shard has one."""
    paths = [Path(d) / f"{name}_{i}.{ext}" for i, d in enumerate(shard_dirs)]
    have = [p.exists() for p in paths]
    if any(have) and not all(have):
        raise FileNotFoundError(f"{name}_<i>.{ext} is missing from some shards: "
                                f"{[str(p) for p, h in zip(paths, have, strict=True) if not h]}")
    return paths if all(have) else []


def merge_shards(shard_dirs, out_dir, expected_rows: int | None = None) -> dict:
    attrs, ohes, beds, per_shard = [], [], [], []
    for i, d in enumerate(shard_dirs):
        d = Path(d)
        a, o, b = d / f"attr_{i}.npy", d / f"ohe_{i}.npy", d / f"regions_{i}.bed"
        for p in (a, o, b):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} missing: shard {i} must be the {i}-th directory given "
                    f"(got {d}); shard order is row order")
        A, O = np.load(a), np.load(o)
        rows = [ln for ln in b.read_text().splitlines() if ln.strip()]
        if not (len(A) == len(O) == len(rows)):
            raise ValueError(f"shard {i}: attr={len(A)} ohe={len(O)} regions={len(rows)} rows disagree")
        if A.shape[1:] != O.shape[1:] or (attrs and A.shape[1:] != attrs[0].shape[1:]):
            raise ValueError(f"shard {i}: shape {A.shape} / {O.shape} does not match the other shards")
        attrs.append(A)
        ohes.append(O)
        beds.extend(rows)
        per_shard.append({"shard": i, "dir": str(d), "rows": int(len(A))})
    n = sum(len(A) for A in attrs)
    if expected_rows is not None and n != expected_rows:
        raise ValueError(f"shards hold {n} rows, expected {expected_rows}")
    filled = _side_files(shard_dirs, "n_filled", "tsv")
    conv = [json.loads(p.read_text()) for p in _side_files(shard_dirs, "convergence", "json")]
    if len({c["threshold"] for c in conv}) > 1:
        raise ValueError("shards used different convergence thresholds")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    attr = np.concatenate(attrs, 0)
    np.save(out / "attr.npy", attr)
    np.save(out / "ohe.npy", np.concatenate(ohes, 0))
    (out / "regions.bed").write_text("".join(r + "\n" for r in beds))
    if filled:
        lines = [p.read_text().splitlines() for p in filled]
        (out / "n_filled.tsv").write_text(
            "\n".join([lines[0][0]] + [ln for f in lines for ln in f[1:] if ln]) + "\n")
    if conv:
        (out / "convergence.json").write_text(json.dumps({
            "threshold": conv[0]["threshold"],
            "pairs": sum(c["pairs"] for c in conv),
            "pairs_over": sum(c["pairs_over"] for c in conv),
            "max_delta": max(c["max_delta"] for c in conv)}, indent=2))
    return {"n_shards": len(shard_dirs), "rows": int(n),
            "width": int(attr.shape[-1]), "per_shard": per_shard}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_dir", action="append", required=True,
                    help="once per shard, in order 0..n-1")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--expected_rows", type=int, default=None)
    a = ap.parse_args()
    rep = merge_shards(a.shard_dir, a.out_dir, a.expected_rows)
    print(f"MERGED {rep['rows']} rows x {rep['width']}bp from {rep['n_shards']} shards", flush=True)


if __name__ == "__main__":
    main()
