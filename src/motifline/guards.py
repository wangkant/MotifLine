"""motifline's refusal rules, expressed as motiflex GuardResults.

Each rule exists because the corresponding silent failure has already been
observed or is undetectable downstream: act-vs-hyp attributions look alike,
finemo 0.30 and 0.41 hitcallers differ, and a row-shifted attr matrix still
feeds modisco without complaint.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from motiflex.guards import GuardResult
from motifline.contracts import params_of_success
from motifline.versions import dist_info_version
from motiflex.run_status import read_run_status


def _ok(gid, detail="ok"):
    return GuardResult(gid, True, detail)


def _fail(gid, detail):
    return GuardResult(gid, False, detail)


def upstream_success(stage_dir) -> GuardResult:
    gid = "upstream_success"
    rs = read_run_status(stage_dir)
    if rs is None:
        return _fail(gid, f"{stage_dir}: no readable run_status.json")
    if rs.get("status") != "SUCCESS":
        return _fail(gid, f"{stage_dir}: last run status is {rs.get('status')}, not SUCCESS")
    return _ok(gid)


def _success_params(stage_dir):
    up = upstream_success(stage_dir)
    if not up.passed:
        return None, up.detail
    p = params_of_success(stage_dir)
    if p is None:
        return None, f"{stage_dir}: cannot join params.json to the successful run"
    return p, None


def attr_is_hypothetical(attr_dir) -> GuardResult:
    gid = "attr_is_hypothetical"
    p, err = _success_params(attr_dir)
    if err:
        return _fail(gid, err)
    if p.get("hypothetical") is not True:
        return _fail(gid, f"{attr_dir}: params.hypothetical is {p.get('hypothetical')!r}, "
                          "hypothetical attributions are the only accepted input")
    return _ok(gid)


def hyp_flags_match(attr_dir_hits, attr_dir_lexicon) -> GuardResult:
    gid = "hyp_flags_match"
    a, err = _success_params(attr_dir_hits)
    if err:
        return _fail(gid, err)
    b, err = _success_params(attr_dir_lexicon)
    if err:
        return _fail(gid, err)
    if a.get("hypothetical") != b.get("hypothetical"):
        return _fail(gid, f"hits attr hypothetical={a.get('hypothetical')!r} but lexicon attr "
                          f"hypothetical={b.get('hypothetical')!r}; mixed scores are not comparable")
    return _ok(gid)


def _npy_shape(path):
    """Shape of a .npy without reading the data.

    ``np.lib.format._read_array_header`` is private and absent from the numpy
    installed here, so the shape comes from a memory-mapped handle instead --
    which also touches only the header.
    """
    return np.load(path, mmap_mode="r").shape


def rows_and_width_match(attr_npy, ohe_npy, regions_bed, expected_width) -> GuardResult:
    gid = "rows_and_width_match"
    sa, so = _npy_shape(attr_npy), _npy_shape(ohe_npy)
    n_bed = sum(1 for line in open(regions_bed) if line.strip())
    if not (sa[0] == so[0] == n_bed):
        return _fail(gid, f"row mismatch: attr={sa[0]} ohe={so[0]} regions={n_bed}")
    if not (sa[-1] == so[-1] == expected_width):
        return _fail(gid, f"width mismatch: attr={sa[-1]} ohe={so[-1]} expected={expected_width}")
    return _ok(gid)


def finemo_version_matches(recorded_dir, site_packages) -> GuardResult:
    gid = "finemo_version_matches"
    p, err = _success_params(recorded_dir)
    if err:
        return _fail(gid, err)
    recorded = (p.get("software") or {}).get("finemo")
    if recorded is None:
        return _fail(gid, f"{recorded_dir}: no software.finemo recorded in params")
    try:
        installed = dist_info_version(site_packages, "finemo")
    except LookupError as e:
        return _fail(gid, str(e))
    if installed != recorded:
        return _fail(gid, f"installed finemo {installed} != recorded {recorded}; "
                          "hitcaller.py differs between versions, hits would not be comparable")
    return _ok(gid)


def finemo_version_declared(declared, site_packages) -> GuardResult:
    """The finemo the stage says it is running is the finemo installed there.

    The cross-version comparison stage deliberately feeds 0.30-extracted regions
    to the 0.41 hit caller, so ``finemo_version_matches`` (regions vs runtime)
    would refuse it for the very property it exists to measure. What still has
    to hold is that the stage's recorded ``software.finemo`` is not a label
    someone typed: it must equal the version of the interpreter the argv points
    at, or the resulting table is attributed to the wrong hitcaller.
    """
    gid = "finemo_version_declared"
    try:
        installed = dist_info_version(site_packages, "finemo")
    except LookupError as e:
        return _fail(gid, str(e))
    if installed != str(declared):
        return _fail(gid, f"stage declares finemo {declared} but {site_packages} "
                          f"has {installed}; the run would be filed under the wrong version")
    return _ok(gid)


def recorded_finemo_version_is(recorded_dir, expected) -> GuardResult:
    """The upstream stage really was produced by the version this stage claims.

    ``regions_extracted_with`` is the other half of the cross-version record;
    if the regions were silently re-extracted the experiment stops being a
    comparison of hit callers alone.
    """
    gid = "recorded_finemo_version_is"
    p, err = _success_params(recorded_dir)
    if err:
        return _fail(gid, err)
    recorded = (p.get("software") or {}).get("finemo")
    if recorded is None:
        return _fail(gid, f"{recorded_dir}: no software.finemo recorded in params")
    if recorded != str(expected):
        return _fail(gid, f"{recorded_dir} was produced by finemo {recorded}, "
                          f"but this stage records regions_extracted_with={expected}")
    return _ok(gid)


def lexicon_exists(lexicon_h5) -> GuardResult:
    """An external lexicon path resolves to a file.

    ``finemo call-hits -m`` on a missing path is a crash, but a path pointing at
    a directory, or at a route that resolved to something unexpected, is the
    case worth catching before a GPU is leased.
    """
    gid = "lexicon_exists"
    p = Path(lexicon_h5)
    if not p.is_file():
        return _fail(gid, f"{p}: external lexicon is not a file "
                          f"({'directory' if p.is_dir() else 'missing'})")
    return _ok(gid)
