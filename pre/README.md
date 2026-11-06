# PRE numerical artifact

**Solution loss in auxiliary-free pairwise Ising encodings of random 3-SAT**

Gyuyoung Park and OukJae Lee. Version 2026-11-07.

With Python 3.12 installed, run from the repository root

```sh
python3 pre/reproduce.py
```

The command verifies and extracts the compressed artifact, creates an isolated environment with the recorded
package versions, obtains the required SATLIB inputs from their original host, and runs the numerical and figure
checks. The first run needs network access for Python packages and SATLIB. Outputs and logs are under
`pre/.reproduced/artifact/reproduced/`. A successful run ends with `PRE ARTIFACT VERIFIED`.

All 6,552 reference numerical definitions are regenerated and compared byte for byte, including the underlying
provenance gates. All ten figure PNGs must match their reference hashes. The public numeric-use fixture preserves
which keys the article prints without publishing its prose. The numerical change ledger's printed-use locations
therefore differ from those in the article source; its value and cause checks remain active.

This is reanalysis of stored results, including explicitly labeled archival sampling data. It is not a rerun of
all stochastic or exact-optimization experiments, a hardware benchmark, or a build of the article PDF. The immutable
historical inputs needed by the inherited numerical registry are kept even when they support no new PRE claim.

`RELEASE.json` gives the hashes and sizes of the compressed code and data. Archive pieces live under `artifacts/`
and are assembled automatically. Inside the artifact, `ARTIFACT_MANIFEST.sha256`, `DATA_CARD.md`, `DATA_INDEX.json`,
`FIGURE_REFERENCES.json` and the numerical history describe the files and checks. Missing or changed inputs fail.

The authors' code uses the MIT license. Their computed data use CC BY 4.0. Vendored material retains its upstream
notices. SATLIB CNF files are not redistributed. Cite this repository at the specific commit used for the analysis.
