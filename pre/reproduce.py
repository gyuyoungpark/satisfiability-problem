#!/usr/bin/env python3
"""Extract the checksum-pinned artifact, create its environment and verify numbers and figures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local_file(name: str) -> Path:
    path = HERE / name
    if not path.resolve().is_relative_to(HERE) or not path.is_file():
        raise ValueError(f"missing or invalid release file {name}")
    return path


def extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"unsafe archive member {member.name}")
        tf.extractall(dest, filter="data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=HERE / ".reproduced")
    ap.add_argument("--check-archives-only", action="store_true")
    args = ap.parse_args()
    release = json.loads((HERE / "RELEASE.json").read_text())
    for name, expected in release["files"].items():
        if digest(local_file(name)) != expected:
            raise ValueError(f"release file checksum differs, {name}")
    if args.check_archives_only:
        print("Release files match their recorded checksums")
        return
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    artifact = work / "artifact"
    if not artifact.resolve().is_relative_to(work):
        raise ValueError("artifact directory escapes the selected work directory")
    artifact.mkdir(exist_ok=True)
    for label, info in release["archives"].items():
        archive = work / (label + ".tar.gz")
        if not archive.resolve().is_relative_to(work):
            raise ValueError("assembled archive escapes the selected work directory")
        with archive.open("wb") as output:
            for part in info["parts"]:
                with local_file(part).open("rb") as source:
                    shutil.copyfileobj(source, output)
        if digest(archive) != info["sha256"]:
            raise ValueError(f"assembled archive checksum differs, {label}")
        extract(archive, artifact)
    environment = work / "environment"
    if not environment.resolve().is_relative_to(work):
        raise ValueError("environment directory escapes the selected work directory")
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(environment)
    lock = local_file("requirements-lock.txt")
    installed = environment / ".requirements-sha256"
    expected_lock = digest(lock)
    env = {**os.environ, "DEMORGAN_ROOT": str(artifact), "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
    env.pop("PYTHONPATH", None)
    if not installed.exists() or installed.read_text().strip() != expected_lock:
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(lock)],
                       check=True, env=env)
        installed.write_text(expected_lock + "\n")
    report_path = artifact / "reproduced/verification.json"
    if not report_path.parent.resolve().is_relative_to(artifact.resolve()):
        raise ValueError("verification output escapes the artifact directory")
    report_path.unlink(missing_ok=True)
    subprocess.run([str(python), str(artifact / "scripts/reproduce_pre_artifacts.py")],
                   cwd=artifact, env=env, check=True)
    report = json.loads(report_path.read_text())
    if (report.get("input_manifest") is not True or report.get("numbers_byte_identical") is not True
            or report.get("figure_pngs_identical") != 10):
        raise ValueError("the artifact did not produce a complete verification report")
    if digest(artifact / "reproduced/numbers.tex") != release["numbers_sha256"]:
        raise ValueError("the verified numerical output differs from the release reference")
    print(f"Verified outputs are in {artifact / 'reproduced'}")


if __name__ == "__main__":
    main()
