"""Exercise each public entry point on a generated tiny formula and reject missing inputs.

This checks execution and input handling, not any reported research result.
"""

import itertools
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SCRIPTS = ("bp_sp_bench.py", "sa_pbit_bench.py", "cross_encoding_bench.py")
CLAUSES = ((1, 2, 3), (-1, -2, 3), (1, -2, -3))


def run(script, inputs, cwd):
    env = os.environ.copy()
    env["DEMORGAN_SAT_BENCHMARKS"] = str(inputs)
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(HERE / script)], cwd=cwd, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
    )


def main():
    # Independently enumerate this test fixture to ensure that it has SAT solutions.
    solutions = sum(
        all(any(bits[abs(lit) - 1] == (lit > 0) for lit in clause) for clause in CLAUSES)
        for bits in itertools.product((False, True), repeat=3)
    )
    assert 0 < solutions < 8
    with tempfile.TemporaryDirectory(prefix="demorgan-sat-smoke-") as temp:
        work = Path(temp)
        good, empty, missing = work / "good", work / "empty", work / "missing"
        good.mkdir()
        empty.mkdir()
        fixture = "c Generated test fixture, not a SATLIB instance.\np cnf 3 3\n"
        fixture += "".join(" ".join(map(str, clause)) + " 0\n" for clause in CLAUSES)
        (good / "uf3-01.cnf").write_text(fixture)
        expected = {
            "bp_sp_bench.py": ("Random", "Bias-only", "Seq. greedy", "BP", "SP", "SUMMARY TABLE"),
            "sa_pbit_bench.py": ("Rand", "Bias", "BP", "Ising-SA", "Ising-pb", "SAT-SA", "SUMMARY"),
            "cross_encoding_bench.py": ("DeMorgan", "Nusslein", "Chancellor", "ORchain", "SUMMARY"),
        }
        for script in SCRIPTS:
            result = run(script, good, work)
            assert result.returncode == 0, (script, result.returncode, result.stderr)
            assert re.search(r"uf3: n=3, m=3,(?: α=[^,]+,)? 1 instances", result.stdout), (script, result.stdout)
            assert all(label in result.stdout for label in expected[script]), script
            assert len(re.findall(r"^\s*uf3\s+", result.stdout, re.M)) == 1, (script, result.stdout)
            assert not re.search(r"\b(?:nan|inf)\b", result.stdout, re.I), (script, result.stdout)
            # The nonempty summary and observed formula count prevent a header-only success.
            if script != "bp_sp_bench.py":
                solved = re.findall(r"(?<![\d.])([01])\s*/\s*1(?![\d.])", result.stdout)
                assert len(solved) >= (9 if script == "sa_pbit_bench.py" else 8), (script, result.stdout)
            print(f"PASS {script}: one synthetic formula and a nonempty finite summary")
            for label, path, diagnostic in (
                ("missing", missing, "Benchmark directory does not exist"),
                ("empty", empty, "No uf*.cnf benchmark files found"),
            ):
                failure = run(script, path, work)
                assert failure.returncode != 0, (script, label)
                assert diagnostic in failure.stderr, (script, label, failure.stderr)
                assert "SUMMARY" not in failure.stdout, (script, label)
                print(f"PASS {script}: {label} input rejected")
    print("PASS synthetic smoke only; no paper-result reproduction was attempted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
