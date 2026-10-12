"""motifline smoke: plant a motif in synthetic attributions and get it back.

Checks the compute env and the downstream half of the pipeline (MoDISco, region
packaging, FiNeMo), including that hits land on the planted coordinates. Attribution
and naming are not covered.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASES = "ACGT"

# AP-1; not palindromic, so a recovered pattern is distinguishable from its reverse complement
MOTIF = "TGACTCA"


def decode(block: np.ndarray) -> str:
    return "".join(BASES[i] for i in block.argmax(axis=0))


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def synthetic(n: int = 300, width: int = 500, motif: str = MOTIF,
              frac: float = 0.6, seed: int = 0):
    """(attr, ohe, planted, bed): `motif` planted in the central half of `frac` of the
    regions, hypothetical-style scores on all four bases, region i at chr1:i*2*width."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(n, width))
    ohe = np.zeros((n, 4, width), dtype=np.int8)
    np.put_along_axis(ohe, idx[:, None, :], 1, axis=1)

    n_planted = round(frac * n)
    which = set(rng.choice(n, size=n_planted, replace=False).tolist())
    m = len(motif)
    if width < 4 * m:
        raise ValueError(f"width {width} is too small to plant a {m}bp motif centrally")
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
        "\t".join(["chr1", str(i * width * 2), str(i * width * 2 + width),
                   f"smoke_{i}", "0", ".", "1.0", "-1", "-1", str(width // 2)])
        for i in range(n)) + "\n"
    return attr, ohe, planted, bed


def _run(argv, cwd) -> None:
    print("  $", " ".join(str(a) for a in argv), flush=True)
    env = {**os.environ}
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    subprocess.run([str(a) for a in argv], cwd=cwd, check=True, env=env)


def _compute_env(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    from motifline.cli import DEFAULT_CONFIG, find_workflow
    from motifline.config import load
    workflow = find_workflow()
    if workflow and (workflow / DEFAULT_CONFIG).is_file():
        env = Path(load(workflow / DEFAULT_CONFIG)["envs"]["compute"]).expanduser()
        return env if env.is_absolute() else (workflow / env).resolve()
    raise SystemExit("no --compute-env given and no config/default.yaml to read it from")


def discovered_patterns(h5_path) -> dict[str, str]:
    """{group.pattern: consensus of |CWM|}."""
    import h5py
    out = {}
    with h5py.File(h5_path, "r") as f:
        for group in ("pos_patterns", "neg_patterns"):
            if group not in f:
                continue
            for name in f[group]:
                cwm = np.asarray(f[group][name]["contrib_scores"])
                out[f"{group}.{name}"] = "".join(BASES[i] for i in np.abs(cwm).argmax(axis=1))
    return out


def hit_agreement(hits_tsv, motif_names, planted, width) -> dict:
    """Precision: share of the motif's hits overlapping a planted site. Recall: share of
    planted sites overlapped by a hit."""
    sites = {i: i * width * 2 + p for i, p in enumerate(planted) if p is not None}
    m = len(MOTIF)
    on, total, found = 0, 0, set()
    with open(hits_tsv) as fh:
        for h in csv.DictReader(fh, delimiter="\t"):
            if h["motif_name"] not in motif_names:
                continue
            total += 1
            s, e = int(h["start"]), int(h["end"])
            i = s // (width * 2)
            if i in sites and s < sites[i] + m and e > sites[i]:
                on += 1
                found.add(i)
    return {"hits": total, "precision": on / total if total else 0.0,
            "recall": len(found) / len(sites) if sites else 0.0}


def run_smoke(compute_env=None, workdir=None, keep: bool = False,
              n: int = 300, width: int = 500) -> int:
    env = _compute_env(compute_env)
    tmp = Path(workdir).resolve() if workdir else Path(tempfile.mkdtemp(prefix="motifline_smoke_"))
    attr_dir = tmp / "attr"
    attr_dir.mkdir(parents=True, exist_ok=True)
    print(f"smoke: {tmp}\nsmoke: compute env {env}")

    attr, ohe, planted, bed = synthetic(n=n, width=width)
    np.save(attr_dir / "attr.npy", attr)
    np.save(attr_dir / "ohe.npy", ohe)
    (attr_dir / "regions.bed").write_text(bed)

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
    recovered = {k for k, p in patterns.items() if any(w in p for w in wanted)}
    agree = hit_agreement(hits_dir / "hits.tsv", recovered, planted, width)
    print(json.dumps({"patterns": patterns, "recovered": sorted(recovered), **agree}, indent=2))

    ok = bool(recovered) and agree["precision"] >= 0.8 and agree["recall"] >= 0.5
    if ok:
        print(f"\nsmoke PASSED: {MOTIF} recovered; hit precision {agree['precision']:.2f}, "
              f"recall {agree['recall']:.2f}")
    else:
        print(f"\nsmoke FAILED: recovered {sorted(recovered)}, {agree}", file=sys.stderr)
    if keep or not ok:
        print(f"working directory kept at {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1
