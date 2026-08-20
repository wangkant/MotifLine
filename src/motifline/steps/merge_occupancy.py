#!/usr/bin/env python
"""Cross-call occupancy: the ONE compiled lexicon scanned over each leaf's regions,
reduced to per-motif occupancy and cross-leaf agreement tables.

Counting caliber, stated once:

* The leaves of a merge are the SAME peak rows under different models; alignment
  across leaves is BY ROW INDEX into regions.bed / regions.npz. The script refuses
  to run if any two leaves' beds differ on (chrom, start, end) anywhere -- refusal,
  not reordering, because a reorder here would silently compare different peaks.
* ``peak_by_motif_<leaf>.tsv.gz`` -- hit COUNTS per (peak row, lexicon entry), one
  column per entry named by its variant_id. A peak "carries" a motif when its
  count is >= 1; that binarisation is what the other two tables use.
* ``occupancy.tsv`` -- one row per (motif, leaf): n_hits (all hits), n_peaks
  (peaks with >= 1 hit), frac_peaks = n_peaks / n_total_peaks.
* ``agreement.tsv`` -- one row per (motif, ordered leaf pair): n_both /
  n_only_<first> / n_only_<second> / n_neither on the binarised columns, and
  jaccard = both / (both + only_first + only_second), empty when that
  denominator is 0.

The motif universe is the compiled lexicon's manifest (tier=core), NOT the union
of motifs seen in the hit tables, so an entry one model never calls still gets
its zero row -- those zeros are the comparison's point.

stdlib + numpy + pandas only; runs in the compute env; must never import motifline.
"""
import argparse
import itertools
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", required=True,
                   help="lexicons/manifest.tsv written by `mlex compile`")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hits", action="append", required=True, metavar="LABEL=PATH",
                   help="one leaf's cross-call hits.tsv; repeat per leaf")
    p.add_argument("--bed", action="append", required=True, metavar="LABEL=PATH",
                   help="the regions.bed the leaf's regions.npz was extracted over; "
                        "repeat per leaf, same labels as --hits")
    return p.parse_args(argv)


def _pairs(items, what):
    out = {}
    for s in items:
        label, _, path = s.partition("=")
        if not label or not path:
            raise SystemExit(f"--{what} {s!r} is not LABEL=PATH")
        if label in out:
            raise SystemExit(f"--{what} label {label!r} given twice")
        out[label] = path
    return out


def check_beds_identical(beds):
    """{label: DataFrame(chrom,start,end)} -> row count, or die on any mismatch."""
    labels = sorted(beds)
    first = beds[labels[0]]
    for label in labels[1:]:
        b = beds[label]
        if len(b) != len(first):
            raise SystemExit(
                f"leaf {label!r} has {len(b)} bed rows, {labels[0]!r} has {len(first)}; "
                f"the cross-call aligns by row index and cannot proceed")
        neq = (b.values != first.values).any(axis=1)
        if neq.any():
            i = int(neq.argmax())
            raise SystemExit(
                f"leaf {label!r} bed row {i} is {tuple(b.iloc[i])}, "
                f"{labels[0]!r} has {tuple(first.iloc[i])}; refusing to align by index")
    return len(first)


def count_matrix(hits, motifs, n_peaks):
    """hits(motif_name, peak_id) -> (n_peaks, len(motifs)) int hit-count matrix."""
    import numpy as np
    mat = np.zeros((n_peaks, len(motifs)), dtype=int)
    col = {m: j for j, m in enumerate(motifs)}
    unknown = set(hits["motif_name"]) - set(motifs)
    if unknown:
        raise SystemExit(f"hit table names motifs absent from the manifest: {sorted(unknown)}")
    bad = (hits["peak_id"] < 0) | (hits["peak_id"] >= n_peaks)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} hits have peak_id outside [0, {n_peaks})")
    for m, sub in hits.groupby("motif_name"):
        cnt = sub["peak_id"].value_counts()
        mat[cnt.index.to_numpy(), col[m]] = cnt.to_numpy()
    return mat


def occupancy_table(mats, motifs, n_peaks):
    import pandas as pd
    rows = []
    for label in sorted(mats):
        m = mats[label]
        for j, motif in enumerate(motifs):
            rows.append({"motif": motif, "leaf": label,
                         "n_hits": int(m[:, j].sum()),
                         "n_peaks": int((m[:, j] >= 1).sum()),
                         "frac_peaks": (m[:, j] >= 1).sum() / n_peaks})
    return pd.DataFrame(rows)


def agreement_table(mats, motifs):
    import pandas as pd
    rows = []
    for a, b in itertools.combinations(sorted(mats), 2):
        ba, bb = mats[a] >= 1, mats[b] >= 1
        for j, motif in enumerate(motifs):
            both = int((ba[:, j] & bb[:, j]).sum())
            only_a = int((ba[:, j] & ~bb[:, j]).sum())
            only_b = int((~ba[:, j] & bb[:, j]).sum())
            denom = both + only_a + only_b
            rows.append({"motif": motif, "leaf_a": a, "leaf_b": b,
                         "n_both": both, "n_only_a": only_a, "n_only_b": only_b,
                         "n_neither": int(len(ba) - denom),
                         "jaccard": both / denom if denom else None})
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse_args(argv)
    import pandas as pd
    from pathlib import Path

    hits_paths = _pairs(args.hits, "hits")
    bed_paths = _pairs(args.bed, "bed")
    if set(hits_paths) != set(bed_paths):
        raise SystemExit(f"--hits labels {sorted(hits_paths)} != --bed labels {sorted(bed_paths)}")

    man = pd.read_csv(args.manifest, sep="\t")
    man = man[man["tier"] == "core"]
    # pattern_tag is the entry's name INSIDE the compiled modisco-format file,
    # which is exactly what finemo writes into hits.tsv:motif_name.
    motifs = man["pattern_tag"].tolist()
    if len(set(motifs)) != len(motifs) or man["variant_id"].duplicated().any():
        raise SystemExit("manifest pattern_tag/variant_id are not unique; "
                         "column naming would silently collapse entries")

    beds = {label: pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2],
                               names=["chrom", "start", "end"])
            for label, p in bed_paths.items()}
    n_peaks = check_beds_identical(beds)

    mats = {}
    for label, p in hits_paths.items():
        h = pd.read_csv(p, sep="\t", usecols=["motif_name", "peak_id"])
        mats[label] = count_matrix(h, motifs, n_peaks)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name_by_motif = dict(zip(man["pattern_tag"], man["variant_id"]))
    ref = beds[sorted(beds)[0]]
    for label, m in mats.items():
        df = pd.DataFrame(m, columns=[name_by_motif[x] for x in motifs])
        pd.concat([ref.reset_index(names="peak_id"), df], axis=1).to_csv(
            out / f"peak_by_motif_{label}.tsv.gz", sep="\t", index=False)

    occ = occupancy_table(mats, motifs, n_peaks)
    agr = agreement_table(mats, motifs)
    for df in (occ, agr):
        df.insert(1, "variant_id", df["motif"].map(name_by_motif))
    occ.to_csv(out / "occupancy.tsv", sep="\t", index=False)
    agr.to_csv(out / "agreement.tsv", sep="\t", index=False)
    print(f"written: {out} ({n_peaks} peaks x {len(motifs)} motifs x {len(mats)} leaves)")


if __name__ == "__main__":
    main()
