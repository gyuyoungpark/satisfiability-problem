# Earlier SAT-to-Ising benchmark code

This directory publishes three scripts from the authors' earlier SAT study. It contains BP/SP initialization,
simulated annealing, p-bit sampling, and cross-encoding benchmarks. It is a small code release, not a PRE
artifact archive or independent verification of the PRE paper's performance claims. Results from a tiny
smoke test must not be read as research evidence.

No SATLIB instances, manuscript sources, reviewer correspondence, presentations, or research-result archives
are included. The scripts read local DIMACS files named `uf*.cnf`, recursively grouped by their `uf<number>`
filename prefix. Obtain any third-party benchmark data separately and follow its source terms.

## Setup

Use Python 3.12 and the pinned NumPy dependency. From this directory, create a local environment and install it.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Set `DEMORGAN_SAT_BENCHMARKS` to your local directory containing `uf*.cnf` files, then run a script.

```sh
export DEMORGAN_SAT_BENCHMARKS=/path/to/your/dimacs
python bp_sp_bench.py
python sa_pbit_bench.py
python cross_encoding_bench.py
```

Without the environment variable, the scripts look for a `benchmarks` directory beside the scripts.
A missing directory or one without matching files exits with an error. The scripts retain the original
batch sizes, seeds, initialization choices, schedules, iteration counts, and text output. Full benchmark
runs can take time. `bp_sp_bench.py` retains its original execution at module import; run it as a script.

## Small execution check

```sh
python smoke_test.py
```

The check creates one small synthetic satisfiable formula in a temporary directory, runs all three scripts
from another working directory, checks that they processed the formula and printed nonempty finite results,
and verifies failure on missing and empty input directories. The fixture is generated during the test and
is not copied from SATLIB. The test does not rerun the historical experiments, reproduce PRE figures or
numbers, test hardware, or establish a solver comparison.

## Source provenance

The table records SHA-256 hashes of the original scripts before portability changes.

| Script | Original SHA-256 |
|---|---|
| `bp_sp_bench.py` | `6baaecc22d3ce553d2362b4a3f2d9f0ddebd1b933b16b53a201ca307f61873a8` |
| `sa_pbit_bench.py` | `51e42ca36db61778c1017c2ae2b6174e947e67abaa8c0ead36f8346ed8f97755` |
| `cross_encoding_bench.py` | `54cee410fea068ed940d4772ba809a664fb89f17577eb6fca519fe0b1f869097` |

The public copies change only the `BENCH_ROOT` constant and add an import from `benchmark_paths.py`.
The shared helper selects the environment/default directory and rejects missing or empty inputs. All
original function and class definitions are unchanged, as checked by Python AST comparison. All other
original bytes, including line endings, are preserved. `SOURCE_PROVENANCE.json` also records the hashes
of the public copies. Original hardcoded session paths are omitted from the public copies.

The code is distributed under the existing authors' MIT license in `LICENSE`.
