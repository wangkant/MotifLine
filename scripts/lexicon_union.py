#!/usr/bin/env python
"""Concatenate the reference MEME libraries into one TomTom target.

Two edits: JASPAR's `MOTIF MA0004.1 Arnt` becomes `MOTIF MA0004.1_Arnt` (TomTom reports
only the first token), and a matrix line without `E=` gets `E= 0`.
"""
import argparse

HEADER = ("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
          "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("sources", nargs="+")
a = ap.parse_args()

with open(a.out, "w") as out:
    out.write(HEADER)
    for path in a.sources:
        started = False
        for line in open(path):
            if line.startswith("MOTIF"):
                started = True
                line = "MOTIF " + "_".join(line.split()[1:]) + "\n"
            elif line.startswith("letter-probability") and "E=" not in line:
                line = line.rstrip("\n") + " E= 0\n"
            if started:
                out.write(line)
