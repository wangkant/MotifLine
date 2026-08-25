#!/usr/bin/env python
"""The union library with each entry's cluster id prefixed onto its name.

``MOTIF MA0139.2_CTCF`` -> ``MOTIF cl181_CTCFL-CTCF_n3__MA0139.2_CTCF``. Not one
matrix value changes; only the ``MOTIF`` line does.

This is the TomTom target for pattern naming, and it exists because the two things a
target library has to do pull in opposite directions:

* **Sensitivity** wants the full set. A pattern's best match is usually NOT its
  cluster's medoid, so matching against the medoid library alone throws away the member
  that actually fits and inflates the q-value of a correct assignment.
* **Naming** wants the cluster, because individual library entries are redundant and
  a per-entry label claims a resolution the matrices do not have.

Tagging the full set satisfies both: TomTom ranks against all 2614 matrices, and the
winning id already carries the cluster the name is reported at. The cost is that the
top-3 can be three members of one cluster -- visible in the id, not hidden.
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--union", required=True)
    p.add_argument("--clusters", required=True, help="clusters_rsat.tsv")
    p.add_argument("--metadata", required=True, help="lexicon_rsat.metadata.tsv")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cl = pd.read_csv(a.clusters, sep="\t")
    meta = pd.read_csv(a.metadata, sep="\t")
    lexid = dict(zip(meta.cluster, meta.lexicon_id))
    tag_of = {r.motif: lexid[int(r.cluster.split("_")[1])] for r in cl.itertuples()}

    lines, tagged = [], 0
    for line in open(a.union):
        if line.startswith("MOTIF "):
            parts = line.rstrip("\n").split(" ")
            tag = tag_of.get(parts[1])
            assert tag is not None, f"union entry is in no cluster: {parts[1]}"
            parts[1] = f"{tag}__{parts[1]}"
            tagged += 1
            line = " ".join(parts) + "\n"
        lines.append(line)
    assert tagged == len(cl), f"tagged {tagged} of {len(cl)} clustered motifs"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    open(a.out, "w").writelines(lines)
    print(f"{a.out}: {tagged} motifs tagged with {meta.shape[0]} cluster ids")


if __name__ == "__main__":
    main()
