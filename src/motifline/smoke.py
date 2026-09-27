"""Does this installation work? Plant a motif, discover it, call it back.

``motifline smoke`` answers the one question a new install cannot otherwise answer:
are modisco and finemo present, do they agree on the array layout this pipeline writes,
and does a motif that is definitely in the input come back out. It needs no genome, no
model weights and no GPU -- the attributions are synthetic, so the only thing under test
is the downstream half of the pipeline and the environment it runs in.

It does not cover the attribution steps (they need real model weights) or naming
(it needs TomTom and the source libraries).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASES = "ACGT"

#: AP-1. Short, high information content, and not palindromic, so a recovered pattern
#: can be told apart from its own reverse complement.
MOTIF = "TGACTCA"


def decode(block: np.ndarray) -> str:
    """One-hot ``(4, W)`` back to sequence."""
    return "".join(BASES[i] for i in block.argmax(axis=0))


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def synthetic(n: int = 300, width: int = 500, motif: str = MOTIF,
              frac: float = 0.6, seed: int = 0):
    """Attributions with ``motif`` planted in ``frac`` of the regions.

    Returns ``(attr, ohe, planted, bed)`` in the layout every attribution stage writes:
    ``attr`` and ``ohe`` are ``(N, 4, W)``, ``planted[i]`` is where the motif starts in
    region *i* or None, and ``bed`` is the 10-column narrowPeak text whose column 10
    carries the summit finemo reads.

    The attribution is hypothetical-shaped on purpose: every base of every position
    carries a score, as ``--hypothetical`` produces, not just the observed one.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(n, width))
    ohe = np.zeros((n, 4, width), dtype=np.int8)
    np.put_along_axis(ohe, idx[:, None, :], 1, axis=1)

    n_planted = round(frac * n)
    which = set(rng.choice(n, size=n_planted, replace=False).tolist())
    m = len(motif)
    if width < 4 * m:
        raise ValueError(f"width {width} is too small to plant a {m}bp motif centrally")
    # Planted in the central half, the way a real summit-centred peak set concentrates.
    half = width // 4
    lo, hi = width // 2 - half, min(width - m, width // 2 + half)
    planted: list[int | None] = []
    for i in range(n):
        if i not in which:
            planted.append(None)
            continue
        start = int(rng.integers(lo, hi))
        for j, base in enumerate(motif):
            ohe[i, :, start + j] = 0
            ohe[i, BASES.index(base), start + j] = 1
        planted.append(start)

    # Background: small hypothetical scores everywhere, on all four bases.
    attr = rng.normal(0.0, 0.01, size=(n, 4, width)).astype(np.float32)
    for i, start in enumerate(planted):
        if start is None:
            continue
        block = slice(start, start + m)
        attr[i, :, block] = rng.normal(-0.25, 0.01, size=(4, m)).astype(np.float32)
        observed = ohe[i, :, block].argmax(axis=0)
        attr[i, observed, np.arange(start, start + m)] = \
            rng.normal(1.0, 0.02, size=m).astype(np.float32)

    bed = "\n".join(
        "\t".join([f"chr1", str(i * width * 2), str(i * width * 2 + width),
                   f"smoke_{i}", "0", ".", "1.0", "-1", "-1", str(width // 2)])
        for i in range(n)) + "\n"
    return attr, ohe, planted, bed


# --- the end-to-end check --------------------------------------------------

def _run(argv, cwd) -> None:
    print("  $", " ".join(str(a) for a in argv), flush=True)
    # CPU unless the caller chose a card: nothing here needs a GPU, and a smoke test
    # must not land on a card somebody else is computing on.
    env = {**os.environ}
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    subprocess.run([str(a) for a in argv], cwd=cwd, check=True, env=env)


def _compute_env(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    from motifline.cli import DEFAULT_CONFIG, find_workflow
    from motifline.config import load
    workflow = find_workflow()
    if workflow and (workflow / DEFAULT_CONFIG).is_file():
        return Path(load(workflow / DEFAULT_CONFIG)["envs"]["compute"])
    raise SystemExit("no --compute-env given and no config/default.yaml to read it from")


def discovered_patterns(h5_path) -> list[str]:
    """Consensus of every MoDISco pattern, read off its contribution weight matrix."""
    import h5py
    out = []
    with h5py.File(h5_path, "r") as f:
        for group in ("pos_patterns", "neg_patterns"):
            if group not in f:
                continue
            for name in f[group]:
                cwm = np.asarray(f[group][name]["contrib_scores"])
                out.append("".join(BASES[i] for i in np.abs(cwm).argmax(axis=1)))
    return out


def run_smoke(compute_env=None, workdir=None, keep: bool = False,
              n: int = 300, width: int = 500) -> int:
    """Generate, discover, call hits, and report whether the planted motif came back."""
    env = _compute_env(compute_env)
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="motifline_smoke_"))
    attr_dir = tmp / "attr"
    attr_dir.mkdir(parents=True, exist_ok=True)
    print(f"smoke: {tmp}\nsmoke: compute env {env}")

    attr, ohe, planted, bed = synthetic(n=n, width=width)
    np.save(attr_dir / "attr.npy", attr)
    np.save(attr_dir / "ohe.npy", ohe)
    (attr_dir / "regions.bed").write_text(bed)
    print(f"smoke: {n} regions x {width}bp, {MOTIF} planted in "
          f"{sum(p is not None for p in planted)}")

    h5 = tmp / "modisco" / "modisco.h5"
    h5.parent.mkdir(exist_ok=True)
    _run([env / "bin" / "modisco", "motifs", "-s", attr_dir / "ohe.npy",
          "-a", attr_dir / "attr.npy", "-n", "50000", "-w", "200", "-o", h5], tmp)

    npz = tmp / "finemo_regions" / "regions.npz"
    npz.parent.mkdir(exist_ok=True)
    _run([env / "bin" / "finemo", "extract-regions-modisco-fmt",
          "-s", attr_dir / "ohe.npy", "-a", attr_dir / "attr.npy",
          "-p", attr_dir / "regions.bed", "-o", npz, "-w", str(width)], tmp)

    hits_dir = tmp / "hits"
    hits_dir.mkdir(exist_ok=True)
    _run([env / "bin" / "finemo", "call-hits", "-M", "pp", "-r", npz, "-m", h5,
          "-o", hits_dir, "-t", "0.3", "-l", "0.7", "-b", "2000"], tmp)

    patterns = discovered_patterns(h5)
    wanted = (MOTIF, revcomp(MOTIF))
    found = [p for p in patterns if any(w in p for w in wanted)]
    n_hits = sum(1 for _ in open(hits_dir / "hits.tsv")) - 1

    print(json.dumps({"patterns": patterns, "recovered": found, "hits": n_hits},
                     indent=2))
    ok = bool(found) and n_hits > 0
    if ok:
        print(f"\nsmoke PASSED: {MOTIF} recovered in {len(found)} of "
              f"{len(patterns)} pattern(s); {n_hits} hits called")
    else:
        print(f"\nsmoke FAILED: planted {MOTIF}, discovered {patterns}, "
              f"{n_hits} hits", file=sys.stderr)
    if keep or not ok:
        print(f"working directory kept at {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1
