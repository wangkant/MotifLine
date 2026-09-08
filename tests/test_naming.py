"""The naming chain's load-bearing claims, each testable in isolation.

Not a unit-test suite for its own sake: every case here is a way the chain has been
seen to fail silently, or a parameter whose value is a decision rather than a default.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motifline import naming as nm

STEPS = Path(__file__).resolve().parents[1] / "src/motifline/steps"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_step", STEPS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


union_step = _load("lexicon_union")
medoid_step = _load("lexicon_medoid")
name_step = _load("name_patterns")

CFG = {"reference_lexicon": {
    "sources": {"H14": "/lib/h14.meme", "JAS": "/lib/jaspar.meme"},
    "rsat": {"bin": "/env/rsat/bin/rsat", "hclust_method": "average",
             "metric": "Ncor", "lth": {"cor": 0.75, "Ncor": 0.55}, "quick": True},
    "tomtom": "/env/meme/bin/tomtom",
    "w_th": 5,
    "gate": {"qval": 0.05, "ncor": 0.55}}}

HOCOMOCO_STYLE = """MEME version 4

ALPHABET= ACGT

MOTIF AHR.H14CORE.0.P.B
letter-probability matrix: alength= 4 w= 2 nsites= 1003
0.1754735792622134\t0.3060817547357926\t0.390827517447657\t0.127617148554337
0.0019940179461615\t0.0\t0.0\t0.9980059820538385
URL https://hocomoco14.autosome.org/motif/AHR.H14CORE.0.P.B
"""

JASPAR_STYLE = """MEME version 4

ALPHABET= ACGT

MOTIF MA0004.1 Arnt
letter-probability matrix: alength= 4 w= 1 nsites= 20 E= 0
 0.200000  0.800000  0.000000  0.000000
"""


def test_union_adds_the_E_field_hocomoco_omits(tmp_path):
    """Without ``E= 0`` RSAT drops the whole HOCOMOCO half and says nothing."""
    p = tmp_path / "h14.meme"
    p.write_text(HOCOMOCO_STYLE)
    lines = union_step.blocks(p)
    matrix_line = next(l for l in lines if l.startswith("letter-probability"))
    assert matrix_line.rstrip("\n").endswith("E= 0")
    # and the probabilities are copied through, not reformatted
    assert "0.1754735792622134\t0.3060817547357926\t0.390827517447657\t0.127617148554337\n" in lines


def test_union_does_not_double_the_E_field_or_split_jaspar_names(tmp_path):
    p = tmp_path / "jaspar.meme"
    p.write_text(JASPAR_STYLE)
    lines = union_step.blocks(p)
    assert lines[0] == "MOTIF MA0004.1_Arnt\n"          # one id, not id + symbol
    matrix_line = next(l for l in lines if l.startswith("letter-probability"))
    assert matrix_line.count("E=") == 1


def test_cluster_label_names_the_medoid_it_represents():
    """A label whose representative belongs to a TF the label omits is unreadable."""
    label = medoid_step.cluster_label(
        181, "MA0139.2_CTCF", ["CTCF", "CTCFL", "CTCFL2"], 3)
    assert label.startswith("cl181_CTCF")
    assert label.endswith("_n3")


def test_gate_is_an_AND_so_a_significant_repeat_match_fails():
    """The failure this arm exists for: an Alu B-box matching HSF1 at q ~ 1e-7 because
    TTCGAA is an HSF1 half-site palindrome. Significant, and the wrong label."""
    assert name_step.gated(1e-7, 0.90, 0.05, 0.55) is True
    assert name_step.gated(1e-7, 0.31, 0.05, 0.55) is False     # Ncor arm
    assert name_step.gated(0.63, 0.90, 0.05, 0.55) is False     # q arm
    assert name_step.gated(1e-7, float("nan"), 0.05, 0.55) is False   # no comparable core


def test_tomtom_line_omits_mi_1():
    """-mi 1 against a batch query file returns matches for pattern 0 and nothing else."""
    argv = name_step.tomtom_argv("/bin/tomtom", "q.meme", "t.meme", "out")
    assert "-mi" not in argv
    assert argv[argv.index("-dist") + 1] == "pearson"


def test_modisco_trim_is_the_thirty_percent_of_max_span():
    ppm = np.tile([0.25, 0.25, 0.25, 0.25], (5, 1))
    cwm = np.zeros((5, 4))
    cwm[1] = [1.0, 0, 0, 0]          # max
    cwm[2] = [0.1, 0, 0, 0]          # inside the span, below threshold: kept anyway
    cwm[3] = [0.5, 0, 0, 0]
    trimmed, start, end = name_step.modisco_trim(ppm, cwm, threshold=0.3)
    assert (start, end) == (1, 4) and len(trimmed) == 3


def test_rsat_thresholds_are_passed_explicitly():
    """RSAT standalone defaults these to empty; the web server's values are not inherited."""
    argv = nm.rsat_argv(CFG, "union.meme", "out/rsat")
    for field, value in (("cor", "0.75"), ("Ncor", "0.55")):
        i = [k for k, a in enumerate(argv) if a == "-lth" and argv[k + 1] == field]
        assert i, f"-lth {field} missing"
        assert argv[i[0] + 2] == value
    assert argv[argv.index("-matrix") + 1] == "union"          # the union_m<N>_ id prefix


def test_a_threshold_quick_would_discard_is_refused():
    """RSAT accepts `-lth w 5 ... -quick` and then drops the w arm: the run reaches R as
    `lthsp = 'Ncor_0.55_cor_0.75_'`. Config must state the thresholds in force."""
    cfg = {"reference_lexicon": {**CFG["reference_lexicon"],
                                 "rsat": {**CFG["reference_lexicon"]["rsat"],
                                          "lth": {"w": 5, "cor": 0.75, "Ncor": 0.55}}}}
    with pytest.raises(nm.NamingError, match="quick"):
        nm.rsat_argv(cfg, "union.meme", "out/rsat")
    # ...unless quick is off, where RSAT does honour it
    cfg["reference_lexicon"]["rsat"]["quick"] = False
    assert "-quick" not in nm.rsat_argv(cfg, "union.meme", "out/rsat")


