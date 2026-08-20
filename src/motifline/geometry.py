"""How wide each model attributes, and how much of that it stores.

Canonical since 2026-08-16 (``config.geometry``):

* ``cbp`` -- 2114bp in, 2114bp stored. ChromBPNet's input window IS its readout;
  there is no crop and a run that asks for one is a different model, not a
  different setting.
* ``ag``  -- 2048bp in, 2048bp stored. AlphaGenome's input floor is 2048bp
  (16 x 128bp trunk bins) and at that width there is nothing to crop, so
  ``crop == in_window``. Precedent for the pair:
  ``02_motif_grammar/interpret_ag2048_par_hyp/run_ag2048_par_hyp_k562.sh:33-34``.

AG cannot simply be run at CBP's 2114 to make the two equal. Tried 2026-08-16 and
it raises: 2114/128 = 17 trunk bins against an attention-bias table built for 32
(``alphagenome_pytorch/attention.py:188``, ``The size of tensor a (17) must match
the size of tensor b (32)``) -- the bin count has to be a power of two. A 2114bp
AG *output* is only reachable by attributing 4096bp and cropping, which buys equal
stored widths at the price of ~991bp of extra context per side that CBP never sees.
The 2048 pair is the one that keeps each model on its own native window; the choice
is the user's, 2026-08-16.

The two models therefore enter MoDISco 66bp apart. That is deliberate: each one
attributes its OWN native window, so neither is being run off-distribution to
match the other. Anything comparing them has to be robust to the 66bp, which
motif discovery is (patterns are tens of bp) and per-position profiles are not.

A run may still override either value, and the adopted runs do -- their archived
attributions were computed at AG 16384 -> crop 2114, and the geometry recorded
for a run has to be the geometry its arrays are actually at, not the one the
pipeline would pick today.
"""
from __future__ import annotations


class GeometryError(KeyError):
    """The run's attribution geometry cannot be resolved -- refuse rather than guess."""


def geometry(cfg: dict, run_cfg: dict) -> dict:
    """``{"in_window": int, "crop": int}`` for a run: config canon + the run's overrides.

    ``crop`` is returned for both models; for ``cbp`` it equals ``in_window``,
    which is what "ChromBPNet stores its whole input" means arithmetically.
    """
    model = run_cfg.get("model")
    if model not in ("cbp", "ag"):
        raise GeometryError(f"unknown model {model!r}; expected 'cbp' or 'ag'")
    canon = (cfg.get("geometry") or {}).get(model)
    if not canon or "in_window" not in canon:
        raise GeometryError(
            f"config.geometry.{model}.in_window is unset; the canonical attribution "
            f"window is a pinned parameter of this pipeline, not a default to invent")
    in_window = int(run_cfg.get("in_window", canon["in_window"]))
    if model == "cbp":
        if run_cfg.get("crop") not in (None, in_window):
            raise GeometryError(
                f"cbp run declares crop={run_cfg['crop']} against in_window={in_window}; "
                f"the ChromBPNet path stores its whole input and never crops")
        crop = in_window
    else:
        crop = int(run_cfg.get("crop", canon.get("crop", in_window)))
        if crop > in_window:
            raise GeometryError(
                f"crop={crop} exceeds in_window={in_window}: the stored array would be "
                f"{in_window}bp while every downstream stage is told it is {crop}bp")
    return {"in_window": in_window, "crop": crop}


def attr_width(cfg: dict, run_cfg: dict) -> int:
    """Width of the saved attributions -- the width finemo and adopt must be told.

    AG stores its central crop, ChromBPNet its whole input window; at the canonical
    geometry AG's crop is its whole window too, so this is ``crop`` either way.
    """
    return geometry(cfg, run_cfg)["crop"]
