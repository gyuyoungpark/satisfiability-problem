"""Portable input discovery for the earlier SAT benchmark scripts."""

import os
from pathlib import Path

ENV_NAME = "DEMORGAN_SAT_BENCHMARKS"


def benchmark_root():
    """Return an existing benchmark directory containing at least one uf*.cnf file."""
    supplied = os.environ.get(ENV_NAME)
    path = Path(supplied).expanduser() if supplied else Path(__file__).resolve().parent / "benchmarks"
    path = path.resolve()
    if not path.is_dir():
        raise SystemExit(
            f"Benchmark directory does not exist: {path}. "
            f"Set {ENV_NAME} to your local DIMACS directory."
        )
    if not any(candidate.is_file() for candidate in path.glob("**/uf*.cnf")):
        raise SystemExit(
            f"No uf*.cnf benchmark files found in {path}. "
            f"Set {ENV_NAME} to a nonempty DIMACS directory."
        )
    return str(path)
