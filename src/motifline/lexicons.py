"""Hit calling against an EXTERNAL motif vocabulary: the gw/mmv route table.

Ported (read-only, 2026-08-14) from
``new_experiment/region_v4/05_allcluster_lexicon/run_finemo.sh:39-51`` --  its
``lexicon_for()`` and ``width_for()`` shell functions, which are the only two
things that vary across the 15 x 2 runs that script performs:

* ``gw``  -- each model scores against its OWN genome-wide MoDISco lexicon.
* ``mmv`` -- both models score against ONE shared motifmultiverse union lexicon,
  deliberately, so that per-instance contributions are comparable at the variant
  level rather than only at a harmonised family label (run_finemo.sh:43-47).

Two properties of the original are kept rather than reimplemented:

1. An unroutable ``(route, model)`` pair is fatal. The shell fell through to
   ``echo ""`` and the caller aborted on the empty string; here it raises. The
   failure mode being prevented is calling ``finemo call-hits`` with no ``-m``.
2. The batch size is a *deviation*, not a default. ``-b 1000`` instead of the
   official 2000 because ``(F.conv1d(contribs**2, cwm_mask) + eps)**-0.5``
   materialises ``(b, n_motifs, 2065)`` three times and a 289-motif union at
   b=2000 exceeds a 49 GiB card (run_finemo.sh:31-36, docs/PARAMS_AUDIT.md).
   It therefore lives in config under ``lexicons.batch``, separate from the
   canonical ``finemo.batch``, and is required rather than defaulted.
"""
from __future__ import annotations


class LexiconRouteError(KeyError):
    """No lexicon is defined for this (route, model) -- refuse rather than guess."""


def lexicon_path(cfg: dict, run_cfg: dict, route: str) -> str:
    """Absolute path of the external lexicon for ``route`` and the run's model."""
    routes = (cfg.get("lexicons") or {}).get("routes") or {}
    if route not in routes:
        raise LexiconRouteError(
            f"unknown lexicon route {route!r}; config defines {sorted(routes)}")
    model = run_cfg.get("lexicon_model")
    if not model:
        raise LexiconRouteError(
            "run declares no 'lexicon_model'; the route table is keyed on the "
            "archive's model names (cbp, ag2048), not on the run's own 'model'")
    table = routes[route]
    if model not in table:
        raise LexiconRouteError(
            f"no {route!r} lexicon for model {model!r}; route defines {sorted(table)}")
    return table[model]


def lexicon_batch(cfg: dict) -> int:
    """``-b`` for external-lexicon hit calling; no default, see module docstring."""
    lex = cfg.get("lexicons") or {}
    if "batch" not in lex:
        raise LexiconRouteError(
            "config.lexicons.batch is unset; the union-lexicon batch size is a "
            "documented deviation from finemo's official default and must be explicit")
    return int(lex["batch"])


def attr_width(run_cfg: dict) -> int:
    """Width of the saved attributions -- the shell's ``width_for()``.

    AG attributes a 16384bp window but stores only the central crop, so the
    width finemo must be told about is the crop; ChromBPNet stores its whole
    input window.
    """
    return run_cfg["crop"] if run_cfg["model"] == "ag" else run_cfg["in_window"]
