# Satisfiability encodings

Gyuyoung Park and OukJae Lee

This repository contains a small code release for the earlier SAT 2026 study and the numerical artifacts for
*Solution loss in auxiliary-free pairwise Ising encodings of random 3-SAT*.

- [demorgan-sat](demorgan-sat/) contains the earlier benchmark code with a portable input path and a small smoke test.
- [pre](pre/) contains checksum-pinned code and computed data for the PRE article. One command rebuilds a fresh
  Python environment, verifies the inputs, and regenerates the reference numbers and ten figures.

The PRE artifact reanalyses stored outputs. It does not rerun every original simulation or build the article PDF.
The source SATLIB instances are fetched from their original host rather than redistributed. Article prose,
cover letters, full internal reviews and author discussions are not included. Numerical history and code comments
retain provenance labels and references to some private review records; those records are not distributed.

The authors' code is MIT licensed. Their computed data are CC BY 4.0. Third-party notices are included in the artifact.
For an immutable citation, use the repository commit containing the files being cited.
