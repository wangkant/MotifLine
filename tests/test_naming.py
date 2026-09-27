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
name_step = _load("name_patterns")

CFG = {"reference_lexicon": {
    "sources": {"H14": "/lib/h14.meme", "JAS": "/lib/jaspar.meme"},
    "tomtom": "/env/meme/bin/tomtom",
    "h14_annotation": "/lib/H14CORE_annotation.jsonl"}}

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
    """The union is kept byte-identical to the one the existing names came from."""
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


ANNOTATION = [
    {"name": "TYY1.H14CORE.0.P.B",
     "masterlist_info": {"species": {"HUMAN": {"gene_symbol": "YY1"}}}},
    {"name": "CTCF.H14CORE.0.P.B",
     "masterlist_info": {"species": {"HUMAN": {"gene_symbol": "CTCF"}}}},
]


def test_names_are_gene_symbols_for_both_id_styles(tmp_path):
    """An H14 prefix is the UniProt entry name (TYY1 is YY1), so the symbol comes from
    HOCOMOCO's annotation; JASPAR's symbol is upper-cased, and a dimer keeps its '::'."""
    p = tmp_path / "h14.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in ANNOTATION))
    h14 = name_step.h14_symbols(p)
    assert name_step.tf_name("TYY1.H14CORE.0.P.B", h14) == "YY1"
    assert name_step.tf_name("CTCF.H14CORE.0.P.B", h14) == "CTCF"
    assert name_step.tf_name("MA0140.3_GATA1::TAL1", h14) == "GATA1::TAL1"
    assert name_step.tf_name("MA2503.1_Banp", h14) == "BANP"
    assert name_step.tf_name("", h14) == ""


def test_an_h14_id_missing_from_the_annotation_is_refused_not_guessed():
    with pytest.raises(KeyError):
        name_step.tf_name("ZBT14.H14CORE.0.P.B", {})


def test_naming_requires_the_h14_annotation():
    cfg = {"reference_lexicon": {k: v for k, v in CFG["reference_lexicon"].items()
                                 if k != "h14_annotation"}}
    with pytest.raises(nm.NamingError, match="h14_annotation"):
        nm.params(cfg)


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
