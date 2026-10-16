#!/usr/bin/env python
"""Stack chrombpnet interpret shards into one <prefix>.counts_scores.h5 + .interpreted_regions.bed.

`adopt_format: chrombpnet_h5` reads one prefix. For an official interpret run that was
split into contiguous shards, this stacks /raw/seq, /shap/seq and /projected_shap/seq in
shard order, values and dtypes untouched, in the layout deepdish writes (blosc level 9
with shuffle, DEEPDISH_IO_VERSION on the root), and joins the shard beds line for line.
It writes nothing unless every shard has the same nodes, each shard's bed has as many
rows as its scores, and (with --check_bed) the joined bed's chr/start/end/summit equal
the peak bed the shards were cut from, row for row. An existing output is never replaced.
The written file is read back and compared array by array.

Runs in the compute env (PyTables) and must not import motifline.
"""
import argparse
import json
import os

import numpy as np

NODES = ("/raw/seq", "/shap/seq", "/projected_shap/seq")


def key_cols(line):
    c = line.rstrip("\n").split("\t")
    return (c[0], int(c[1]), int(c[2]), int(c[9]))


def merge(shard_prefixes, out_prefix, check_bed=None):
    import tables
    arrays = {n: [] for n in NODES}
    bed_lines = []
    for p in shard_prefixes:
        with tables.open_file(f"{p}.counts_scores.h5") as f:
            have = sorted(x._v_pathname for x in f.walk_nodes("/", "Array"))
            if have != sorted(NODES):
                raise SystemExit(f"{p}: nodes {have}, expected {sorted(NODES)}")
            for n in NODES:
                arrays[n].append(f.get_node(n).read())
        lines = [ln if ln.endswith("\n") else ln + "\n"
                 for ln in open(f"{p}.interpreted_regions.bed") if ln.strip()]
        n_rows = arrays[NODES[0]][-1].shape[0]
        if len(lines) != n_rows:
            raise SystemExit(f"{p}: {len(lines)} bed lines, {n_rows} score rows")
        bed_lines += lines
        print(f"{p}: {arrays[NODES[0]][-1].shape}", flush=True)

    merged = {n: np.concatenate(v) for n, v in arrays.items()}
    shapes = {merged[n].shape for n in NODES}
    if len(shapes) != 1:
        raise SystemExit(f"node shapes differ: {shapes}")
    if check_bed:
        ref = [key_cols(ln) for ln in open(check_bed) if ln.strip()]
        got = [key_cols(ln) for ln in bed_lines]
        if ref != got:
            first = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y),
                         min(len(ref), len(got)))
            raise SystemExit(f"joined bed differs from {check_bed} (rows {len(got)} vs "
                             f"{len(ref)}; first difference at row {first})")

    out_h5 = f"{out_prefix}.counts_scores.h5"
    out_bed = f"{out_prefix}.interpreted_regions.bed"
    for path in (out_h5, out_bed):
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; move it aside first")
    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)
    filters = tables.Filters(complevel=9, complib="blosc", shuffle=True)
    with tables.open_file(out_h5, "w") as f:
        f.root._v_attrs.DEEPDISH_IO_VERSION = np.int64(12)
        for n in NODES:
            grp, leaf = n.strip("/").split("/")
            f.create_carray(f.create_group("/", grp), leaf, obj=merged[n], filters=filters)
    with open(out_bed, "w") as fh:
        fh.writelines(bed_lines)

    with tables.open_file(out_h5) as f:
        for n in NODES:
            back = f.get_node(n).read()
            if back.dtype != merged[n].dtype or not np.array_equal(back, merged[n]):
                raise SystemExit(f"{n}: read-back differs from the stacked shards")
    return {"out": out_h5, "rows": int(merged[NODES[0]].shape[0]),
            "width": int(merged[NODES[0]].shape[-1]),
            "dtypes": {n: str(merged[n].dtype) for n in NODES},
            "shards": len(shard_prefixes), "bed_checked": bool(check_bed),
            "read_back_equal": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_prefix", action="append", required=True,
                    help="<dir>/<prefix> of one shard; once per shard, in shard order")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--check_bed", default=None, help="the peak bed the shards were cut from")
    a = ap.parse_args()
    print(json.dumps(merge(a.shard_prefix, a.out_prefix, a.check_bed)), flush=True)


if __name__ == "__main__":
    main()
