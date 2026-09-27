#!/usr/bin/env python
"""Concatenate the reference source libraries into ONE union MEME file, TomTom's target.

Two edits, nothing else -- every matrix value is copied through verbatim:

* ``MOTIF MA0004.1 Arnt`` -> ``MOTIF MA0004.1_Arnt``. JASPAR puts the gene symbol in a
  third column; TomTom keys on a single id, and the two-token form would make the id
  ambiguous against HOCOMOCO's single-token names.
* ``letter-probability matrix: ...`` gains ``E= 0`` when it has no ``E=`` field. TomTom
  ignores it; it is kept so the union stays byte-identical to the one the existing
  names were made from.
"""
import argparse
import re
from pathlib import Path

MEME_HEADER = ("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
               "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")


def blocks(path):
    """Every line from the first ``MOTIF`` on, with the two edits applied."""
    out, started = [], False
    for line in open(path):
        if line.startswith("MOTIF"):
            started = True
            parts = line.rstrip("\n").split()
            name = parts[1] if len(parts) == 2 else f"{parts[1]}_{parts[2]}"
            line = f"MOTIF {name}\n"
        elif line.startswith("letter-probability") and not re.search(r"\bE=", line):
            line = line.rstrip("\n") + " E= 0\n"
        if started:
            out.append(line)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="append", required=True,
                   help="tag=path, repeatable; concatenated in the order given")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    lines, counts = [], []
    for spec in a.source:
        tag, path = spec.split("=", 1)
        b = blocks(path)
        counts.append((tag, sum(l.startswith("MOTIF") for l in b)))
        lines += b
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(MEME_HEADER)
        f.writelines(lines)
    total = sum(n for _, n in counts)
    print(f"{a.out}: {total} motifs (" + ", ".join(f"{t} {n}" for t, n in counts) + ")")


if __name__ == "__main__":
    main()
