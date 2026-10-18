#!/usr/bin/env python
"""Concatenate the reference MEME libraries into one TomTom target.

Matrices are copied verbatim. Two edits: JASPAR's `MOTIF MA0004.1 Arnt` becomes
`MOTIF MA0004.1_Arnt` (TomTom reports only the first token), and a matrix line without
an `E=` field gets `E= 0`.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

MEME_HEADER = ("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
               "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")


def blocks(path):
    out, started = [], False
    for line in open(path):
        if line.startswith("MOTIF"):
            started = True
            parts = line.split()
            if len(parts) == 2:
                name = parts[1]
            elif len(parts) == 3:
                name = f"{parts[1]}_{parts[2]}"
            else:
                raise ValueError(f"{path}: unexpected MOTIF line {line.rstrip()!r}")
            line = f"MOTIF {name}\n"
        elif line.startswith("letter-probability") and not re.search(r"\bE=", line):
            line = line.rstrip("\n") + " E= 0\n"
        if started:
            # a source without a trailing newline would fuse with the next MOTIF line
            out.append(line if line.endswith("\n") else line + "\n")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="append", required=True,
                   help="tag=path, repeatable; concatenated in this order")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    lines, counts = [], []
    for spec in a.source:
        tag, path = spec.split("=", 1)
        b = blocks(path)
        counts.append((tag, sum(l.startswith("MOTIF") for l in b)))
        lines += b
    ids = [l.split()[1] for l in lines if l.startswith("MOTIF")]
    dup = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dup:
        raise SystemExit(f"duplicate motif ids across sources: {dup[:5]}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(MEME_HEADER)
        f.writelines(lines)
    print(f"{a.out}: {len(ids)} motifs (" + ", ".join(f"{t} {n}" for t, n in counts) + ")")


if __name__ == "__main__":
    main()
