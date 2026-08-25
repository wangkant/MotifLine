"""MEME matrix I/O and the RSAT-shaped Ncor, shared by the lexicon and naming steps.

Imported, never run. Lives under steps/ because its consumers are stage bodies that
run in the compute env, not in the orchestration env that imports ``motifline``.

Ncor is the similarity RSAT ``matrix-clustering`` builds its tree on (Castro-Mondragon
et al., NAR 2017, e119): over every ungapped offset and both strands, Pearson ``cor``
on the overlapping columns whenever at least ``w_th`` columns overlap, length-normalised

    Ncor = cor * W_overlap / max(W_a, W_b)

and maximised over offsets/strands. This is the same *shape* of normalisation, computed
here in numpy rather than by calling RSAT -- it is what selects the medoid of a cluster
and what supplies ``best_ncor`` for the annotation gate. The partition itself is never
computed here: that comes from RSAT's own ``clusters.tab``.
"""
from __future__ import annotations

import re

import numpy as np

#: Fallback minimum overlapping columns for a pair to be comparable. Callers inside
#: the pipeline pass ``config.reference_lexicon.w_th`` explicitly; this default exists
#: so the module is usable standalone. It is NOT RSAT's ``-lth w`` (RSAT discards that
#: arm under ``-quick``) and NOT TomTom's ``-min-overlap``.
W_TH = 5


def read_meme(path):
    """``{motif_name: (ppm, nsites)}`` in file order.

    A JASPAR-style ``MOTIF MA0004.1 Arnt`` line becomes ``MA0004.1_Arnt``: the union
    library carries both naming styles and the two-token form must not collide.
    """
    out, name, mat, nsites = {}, None, None, None
    for line in open(path):
        s = line.strip()
        if s.startswith("MOTIF"):
            if name is not None and mat:
                out[name] = (np.array(mat, float), nsites)
            p = s.split()
            name = p[1] if len(p) == 2 else f"{p[1]}_{p[2]}"
            mat, nsites = None, None
        elif s.startswith("letter-probability"):
            mat = []
            m = re.search(r"nsites=\s*(\d+)", s)
            nsites = int(m.group(1)) if m else None
        elif mat is not None:
            v = s.split()
            if len(v) == 4:
                try:
                    mat.append([float(x) for x in v])
                except ValueError:
                    pass
    if name is not None and mat:
        out[name] = (np.array(mat, float), nsites)
    return out


MEME_HEADER = ("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
               "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")


def write_meme(path, blocks):
    """``blocks`` are ready-made ``MOTIF ...`` chunks; only the header is added."""
    with open(path, "w") as f:
        f.write(MEME_HEADER + "".join(blocks))


def meme_block(name, ppm, nsites=None):
    """One MEME record, probabilities renormalised per row.

    ``E= 0`` is on the matrix line unconditionally: RSAT's MEME reader
    (``_readFromMEMEFile_2015``) requires the field and drops -- silently -- every
    record that lacks it.
    """
    rows = []
    for r in ppm:
        c = np.clip(r, 1e-6, None)
        rows.append("  " + "  ".join(f"{x:.6f}" for x in c / c.sum()) + "\n")
    return (f"MOTIF {name}\nletter-probability matrix: alength= 4 w= {ppm.shape[0]} "
            f"nsites= {nsites or 100} E= 0\n" + "".join(rows) + "\n")


def ic(ppm):
    """Per-column information content against a uniform background."""
    return (ppm * np.log2(ppm / 0.25 + 1e-9)).sum(1)


def ic_core(ppm, threshold=0.3, w_th=W_TH):
    """The IC>threshold core of a PPM, or None if it is shorter than ``w_th``.

    This is the trim the Ncor side uses. It is NOT the trim the TomTom query uses
    (that one is modiscolite's |CWM| 30%-of-max rule, in ``name_patterns.py``); the
    two coexist on purpose and the gate reads one number from each.
    """
    k = np.where(ic(ppm) > threshold)[0]
    if len(k) < w_th:
        return None
    return ppm[k[0]:k[-1] + 1]


def _pad(mats, L):
    """Left-aligned zero pad, plus the left-aligned reverse complement, plus widths."""
    n = len(mats)
    A = np.zeros((n, L, 4))
    R = np.zeros((n, L, 4))
    W = np.zeros(n, dtype=int)
    for i, m in enumerate(mats):
        w = len(m)
        W[i] = w
        A[i, :w] = m
        R[i, :w] = m[::-1, ::-1]
    return A, R, W


def ncor_matrix(mats, w_th=W_TH):
    """All-pairs ``(Ncor, cor, w_overlap, widths)``; one matmul per offset does every pair.

    ``cor`` and ``w_overlap`` are read at the SAME alignment that maximised Ncor, not at
    their own optima -- RSAT reports one row per compared pair, so the three scores its
    thresholds are ANDed over describe one registration. ``w_overlap`` is the number of
    aligned columns there, which is what RSAT's ``-lth w`` is a floor on.
    """
    n = len(mats)
    L = max(len(m) for m in mats)
    A, R, W = _pad(mats, L)

    def prefix(X):
        s = np.concatenate([np.zeros((n, 1)), X.sum(2).cumsum(1)], 1)
        s2 = np.concatenate([np.zeros((n, 1)), (X ** 2).sum(2).cumsum(1)], 1)
        return s, s2

    SxA, SxxA = prefix(A)
    SxR, SxxR = prefix(R)
    best = np.full((n, n), -np.inf)
    bestcor = np.full((n, n), -np.inf)      # the cor at the Ncor-maximising alignment
    bestw = np.zeros((n, n))                # and the overlap width there
    Wmax = np.maximum.outer(W, W).astype(np.float64)

    for Bmat, SxB, SxxB in ((A, SxA, SxxA), (R, SxR, SxxR)):
        for d in range(-(L - 1), L):
            # d >= 0: B shifted right by d, so A contributes [d, L) and B [0, L-d).
            ai, bi, span = (d, 0, L - d) if d >= 0 else (0, -d, L + d)
            if span < w_th:
                continue
            Af = A[:, ai:ai + span, :].reshape(n, -1)
            Bf = Bmat[:, bi:bi + span, :].reshape(n, -1)
            Sxy = Af @ Bf.T
            # real overlap per pair: each motif's own length intersected with the window
            Wa = np.minimum(np.clip(W[:, None] - ai, 0, span),
                            np.clip(W[None, :] - bi, 0, span)).astype(np.float64)
            ok = Wa >= w_th
            if not ok.any():
                continue
            enA = (ai + Wa).astype(int)
            enB = (bi + Wa).astype(int)
            Sx = np.take_along_axis(SxA, enA, 1) - SxA[:, [ai]]
            Sxx = np.take_along_axis(SxxA, enA, 1) - SxxA[:, [ai]]
            # SxB is (n, L+1) and enB.T is (nb, na): the take gives [j, i], transpose back
            Sy = (np.take_along_axis(SxB, enB.T, 1) - SxB[:, [bi]]).T
            Syy = (np.take_along_axis(SxxB, enB.T, 1) - SxxB[:, [bi]]).T
            m = 4.0 * Wa
            num = m * Sxy - Sx * Sy
            va = m * Sxx - Sx ** 2
            vb = m * Syy - Sy ** 2
            den = np.sqrt(np.clip(va, 0, None) * np.clip(vb, 0, None))
            with np.errstate(invalid="ignore", divide="ignore"):
                cor = np.where(den > 0, num / den, 0.0)
            nc = np.where(ok, cor * Wa / Wmax, -np.inf)
            upd = nc > best
            best = np.where(upd, nc, best)
            bestcor = np.where(upd, np.where(ok, cor, -np.inf), bestcor)
            bestw = np.where(upd, np.where(ok, Wa, 0.0), bestw)
    # both strand passes are one-directional; symmetrise on Ncor and let cor/w follow
    take_t = best.T > best
    best = np.where(take_t, best.T, best)
    bestcor = np.where(take_t, bestcor.T, bestcor)
    bestw = np.where(take_t, bestw.T, bestw)
    np.fill_diagonal(best, 1.0)
    np.fill_diagonal(bestcor, 1.0)
    np.fill_diagonal(bestw, W.astype(np.float64))
    return np.clip(best, -1.0, 1.0), np.clip(bestcor, -1.0, 1.0), bestw, W
