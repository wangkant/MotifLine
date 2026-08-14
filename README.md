# motifline

Audited pipeline from attribution to MoDISco to FiNeMo to motiflex: every stage
directory carries its result alongside `provenance.json`, `params.json`,
`run_status.json` and `guard_outcomes.json`, the orchestration layer (Python
3.12) stays separate from the compute layer (chrombpnet_local, Python 3.10),
and a violated contract produces a REFUSED stage rather than a result.

Design spec: `docs/superpowers/specs/2026-08-14-motifline-design.md`.
