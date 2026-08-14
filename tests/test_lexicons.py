"""The gw/mmv lexicon routing table, ported from run_finemo.sh.

The shell original resolved `-m` with a `case "$ROUTE/$1"` and aborted on an
empty result; the port has to keep both halves -- an unroutable (route, model)
pair must raise, not silently fall through to "no lexicon" and let finemo be
called without `-m`.
"""
import pytest
from motifline.lexicons import LexiconRouteError, attr_width, lexicon_batch, lexicon_path

CFG = {
    "lexicons": {
        "batch": 1000,
        "routes": {
            "gw": {"cbp": "/arch/cbp_gw.h5", "ag2048": "/arch/ag_gw.h5"},
            "mmv": {"cbp": "/arch/core.h5", "ag2048": "/arch/core.h5"},
        },
    }
}


def test_routes_match_the_shell_table():
    assert lexicon_path(CFG, {"lexicon_model": "cbp"}, "gw") == "/arch/cbp_gw.h5"
    assert lexicon_path(CFG, {"lexicon_model": "ag2048"}, "gw") == "/arch/ag_gw.h5"
    # mmv deliberately routes BOTH models to one shared vocabulary (run_finemo.sh:43-49)
    assert (lexicon_path(CFG, {"lexicon_model": "cbp"}, "mmv")
            == lexicon_path(CFG, {"lexicon_model": "ag2048"}, "mmv") == "/arch/core.h5")


def test_unknown_route_or_model_raises_instead_of_returning_empty():
    with pytest.raises(LexiconRouteError):
        lexicon_path(CFG, {"lexicon_model": "cbp"}, "nosuchroute")
    with pytest.raises(LexiconRouteError):
        lexicon_path(CFG, {"lexicon_model": "enformer"}, "gw")
    with pytest.raises(LexiconRouteError):
        lexicon_path(CFG, {}, "gw")          # run never declared a lexicon_model


def test_batch_is_the_documented_deviation_not_the_official_default():
    assert lexicon_batch(CFG) == 1000
    with pytest.raises(LexiconRouteError):
        lexicon_batch({"lexicons": {"routes": {}}})


def test_width_follows_the_model_not_the_route():
    # width_for() in the shell: cbp 2114, ag2048 2048 -- the attribution window,
    # which in motifline is the run's crop (AG) or in_window (CBP).
    assert attr_width({"model": "cbp", "in_window": 2114}) == 2114
    assert attr_width({"model": "ag", "in_window": 16384, "crop": 2048}) == 2048
