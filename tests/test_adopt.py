import numpy as np
import pytest

from motifline.adopt import adopt_attr, normalize_peaks


def _write(p, text):
    p.write_text(text)
    return p


def test_normalize_passes_through_eleven_column_bed(tmp_path):
    src = _write(tmp_path / "peaks.bed", "chr1\t0\t2114\tPeak_1\t9\t.\t1.0\t2.0\t3.0\t1057\tTrue\n" * 3)
    dst = tmp_path / "regions.bed"
    info = normalize_peaks(src, dst)
    assert info == {"rows": 3, "src_columns": 11, "expanded_from_summit_bed": False}
    assert dst.read_text() == src.read_text()


def test_normalize_expands_four_column_summit_bed(tmp_path):
    src = _write(tmp_path / "peaks.bed", "chr1\t100\t2214\t1057\nchr2\t5\t2119\t300\n")
    dst = tmp_path / "regions.bed"
    info = normalize_peaks(src, dst)
    assert info == {"rows": 2, "src_columns": 4, "expanded_from_summit_bed": True}
    assert dst.read_text() == ("chr1\t100\t2214\t.\t.\t.\t.\t.\t.\t1057\tTrue\n"
                               "chr2\t5\t2119\t.\t.\t.\t.\t.\t.\t300\tTrue\n")


def test_normalize_rejects_unknown_column_count(tmp_path):
    src = _write(tmp_path / "peaks.bed", "chr1\t0\t10\tx\ty\n")
    with pytest.raises(ValueError, match="columns"):
        normalize_peaks(src, tmp_path / "regions.bed")


def test_normalize_rejects_ragged_bed(tmp_path):
    src = _write(tmp_path / "peaks.bed", "chr1\t0\t10\t5\nchr1\t0\t10\t5\t.\t.\t.\t.\t.\t5\tTrue\n")
    with pytest.raises(ValueError, match="columns"):
        normalize_peaks(src, tmp_path / "regions.bed")


def _archive(tmp_path, n_rows=3, width=2114, bed_rows=None):
    src = tmp_path / "archive"
    src.mkdir()
    np.save(src / "attr.npy", np.zeros((n_rows, 4, width), np.float32))
    np.save(src / "ohe.npy", np.zeros((n_rows, 4, width), np.int8))
    (src / "peaks.bed").write_text("chr1\t0\t2114\t1057\n" * (n_rows if bed_rows is None else bed_rows))
    return src


def test_adopt_symlinks_arrays_and_writes_regions(tmp_path):
    src = _archive(tmp_path)
    out = tmp_path / "attr"
    info = adopt_attr(src, out, expected_rows=3, expected_width=2114)
    assert info["rows"] == 3 and info["width"] == 2114
    assert (out / "attr.npy").is_symlink() and (out / "ohe.npy").is_symlink()
    assert (out / "attr.npy").resolve() == (src / "attr.npy").resolve()
    assert np.load(out / "attr.npy").shape == (3, 4, 2114)
    assert (out / "regions.bed").read_text().count("\n") == 3
    assert info["sources"]["attr"] == str((src / "attr.npy").resolve())


def test_adopt_is_idempotent(tmp_path):
    src = _archive(tmp_path)
    out = tmp_path / "attr"
    adopt_attr(src, out, expected_rows=3, expected_width=2114)
    adopt_attr(src, out, expected_rows=3, expected_width=2114)
    assert (out / "attr.npy").resolve() == (src / "attr.npy").resolve()


def test_adopt_refuses_row_drift_between_bed_and_arrays(tmp_path):
    src = _archive(tmp_path, n_rows=3, bed_rows=2)
    with pytest.raises(ValueError, match="row"):
        adopt_attr(src, tmp_path / "attr", expected_rows=3, expected_width=2114)


def test_adopt_refuses_expected_row_drift(tmp_path):
    src = _archive(tmp_path, n_rows=3)
    with pytest.raises(ValueError, match="expected 4 rows"):
        adopt_attr(src, tmp_path / "attr", expected_rows=4, expected_width=2114)


def test_adopt_refuses_width_drift(tmp_path):
    src = _archive(tmp_path, n_rows=3, width=1000)
    with pytest.raises(ValueError, match="width"):
        adopt_attr(src, tmp_path / "attr", expected_rows=3, expected_width=2114)


def test_adopt_writes_nothing_when_it_refuses(tmp_path):
    src = _archive(tmp_path, n_rows=3, bed_rows=2)
    out = tmp_path / "attr"
    with pytest.raises(ValueError):
        adopt_attr(src, out, expected_rows=3, expected_width=2114)
    assert not (out / "attr.npy").exists() and not (out / "regions.bed").exists()


def test_adopt_takes_an_alternative_peaks_file(tmp_path):
    src = _archive(tmp_path)
    (src / "peaks_np.bed").write_text("chrX\t0\t2114\t.\t.\t.\t.\t.\t.\t1057\tTrue\n" * 3)
    out = tmp_path / "attr"
    adopt_attr(src, out, expected_rows=3, expected_width=2114, peaks_name="peaks_np.bed")
    assert (out / "regions.bed").read_text().startswith("chrX")
