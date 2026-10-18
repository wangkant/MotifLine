"""Stacking sharded chrombpnet interpret output into one prefix (motifline merge-h5)."""
import importlib.util

import numpy as np
import pytest

from conftest import STEPS

_spec = importlib.util.spec_from_file_location("merge_chrombpnet_h5_step",
                                               STEPS / "merge_chrombpnet_h5.py")
step = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step)


def write_shard(prefix, rows, first, width=6):
    """A deepdish-style counts_scores.h5 and its interpreted_regions.bed; peak i at 100*i."""
    tables = pytest.importorskip("tables")
    rng = np.random.default_rng(first)
    raw = np.eye(4, dtype=np.int8)[rng.integers(0, 4, (rows, width))].transpose(0, 2, 1)
    shap = rng.normal(size=raw.shape).astype(np.float16)
    with tables.open_file(f"{prefix}.counts_scores.h5", "w") as f:
        for grp, arr in (("raw", raw), ("shap", shap),
                         ("projected_shap", (raw * shap).astype(np.float16))):
            f.create_carray(f.create_group("/", grp), "seq", obj=arr)
    with open(f"{prefix}.interpreted_regions.bed", "w") as fh:
        fh.writelines(f"chr1\t{100 * i}\t{100 * i + 50}\t.\t0\t.\t0\t0\t0\t25\n"
                      for i in range(first, first + rows))
    return raw, shap


def _read(path, node):
    import tables
    with tables.open_file(path) as f:
        return f.get_node(node).read(), f.root._v_attrs.DEEPDISH_IO_VERSION


def test_shards_are_stacked_in_the_order_given(tmp_path):
    a = write_shard(tmp_path / "s0", rows=3, first=0)
    b = write_shard(tmp_path / "s1", rows=2, first=3)
    rep = step.merge([str(tmp_path / "s0"), str(tmp_path / "s1")], str(tmp_path / "all"))
    assert rep["rows"] == 5 and rep["read_back_equal"]
    shap, version = _read(tmp_path / "all.counts_scores.h5", "/shap/seq")
    assert shap.dtype == np.float16 and np.array_equal(shap, np.concatenate([a[1], b[1]]))
    assert version == 12
    starts = [int(ln.split("\t")[1]) for ln in
              (tmp_path / "all.interpreted_regions.bed").read_text().splitlines()]
    assert starts == [0, 100, 200, 300, 400]


def test_the_joined_regions_must_follow_the_peak_bed(tmp_path):
    write_shard(tmp_path / "s0", rows=2, first=0)
    write_shard(tmp_path / "s1", rows=2, first=2)
    bed = tmp_path / "peaks.bed"
    bed.write_text((tmp_path / "s1.interpreted_regions.bed").read_text()
                   + (tmp_path / "s0.interpreted_regions.bed").read_text())
    with pytest.raises(SystemExit, match="missing or out of order"):
        step.merge([str(tmp_path / "s0"), str(tmp_path / "s1")], str(tmp_path / "all"),
                   check_bed=str(bed))
    assert not (tmp_path / "all.counts_scores.h5").exists()


def _bed_with_extra_peak(tmp_path):
    """The two shards' rows plus one peak (start 150, window centre 175) they lack."""
    write_shard(tmp_path / "s0", rows=2, first=0)
    write_shard(tmp_path / "s1", rows=2, first=2)
    lines = ((tmp_path / "s0.interpreted_regions.bed").read_text()
             + (tmp_path / "s1.interpreted_regions.bed").read_text()).splitlines(keepends=True)
    extra = lines[1].replace("\t100\t", "\t150\t", 1)
    bed = tmp_path / "peaks.bed"
    bed.write_text("".join(lines[:2] + [extra] + lines[2:]))
    return bed


def test_a_peak_whose_window_runs_off_the_chromosome_may_be_missing(tmp_path):
    bed = _bed_with_extra_peak(tmp_path)
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chr1\t177\n")          # centre 175 + width 6 // 2 > 177
    rep = step.merge([str(tmp_path / "s0"), str(tmp_path / "s1")], str(tmp_path / "all"),
                     check_bed=str(bed), chrom_sizes=str(sizes))
    assert rep["bed_dropped"] == 1


def test_a_missing_peak_inside_the_chromosome_is_refused(tmp_path):
    bed = _bed_with_extra_peak(tmp_path)
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chr1\t100000\n")
    for kw in ({}, {"chrom_sizes": str(sizes)}):
        with pytest.raises(SystemExit, match="missing or out of order"):
            step.merge([str(tmp_path / "s0"), str(tmp_path / "s1")], str(tmp_path / "all"),
                       check_bed=str(bed), **kw)
    assert not (tmp_path / "all.counts_scores.h5").exists()


def test_a_shard_left_out_is_refused(tmp_path):
    write_shard(tmp_path / "s0", rows=2, first=0)
    write_shard(tmp_path / "s1", rows=2, first=2)
    write_shard(tmp_path / "s2", rows=2, first=4)
    bed = tmp_path / "peaks.bed"
    bed.write_text("".join((tmp_path / f"s{i}.interpreted_regions.bed").read_text()
                           for i in range(3)))
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chr1\t100000\n")
    for shards in (["s0", "s1"], ["s0", "s2"], ["s1", "s2"]):
        with pytest.raises(SystemExit, match="missing or out of order"):
            step.merge([str(tmp_path / s) for s in shards], str(tmp_path / "all"),
                       check_bed=str(bed), chrom_sizes=str(sizes))


def test_an_existing_output_is_never_replaced(tmp_path):
    write_shard(tmp_path / "s0", rows=2, first=0)
    (tmp_path / "all.counts_scores.h5").write_text("keep me")
    with pytest.raises(SystemExit, match="exists"):
        step.merge([str(tmp_path / "s0")], str(tmp_path / "all"))
    assert (tmp_path / "all.counts_scores.h5").read_text() == "keep me"


def test_a_shard_whose_bed_and_scores_disagree_is_refused(tmp_path):
    write_shard(tmp_path / "s0", rows=3, first=0)
    bed = tmp_path / "s0.interpreted_regions.bed"
    bed.write_text("".join(bed.read_text().splitlines(keepends=True)[:2]))
    with pytest.raises(SystemExit, match="2 bed lines, 3 score rows"):
        step.merge([str(tmp_path / "s0")], str(tmp_path / "all"))
