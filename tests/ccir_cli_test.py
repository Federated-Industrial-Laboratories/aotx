#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check the CCIR file commands with distinct batches and invalid inputs.
# Input: executable path. Output: check counts. Exit: 0 on success, 1 on failure.
import pathlib
import subprocess
import sys
import tempfile

checks = 0


def check(value):
    global checks
    checks += 1
    if not value:
        raise AssertionError(f"check {checks} failed")


def run(*args, code=0):
    result = subprocess.run([sys.argv[1], *map(str, args)], capture_output=True,
                            text=True, timeout=20)
    check(result.returncode == code)
    return result.stdout


def cases(root, count):
    files = []
    for item in range(count):
        checkpoint = root / f"checkpoint-{item}.bin"
        tail = root / f"tail-{item}.bin"
        path = root / f"state-{item}.aotxccir"
        compact = root / f"small-{item}.aotxccir"
        checkpoint.write_bytes(bytes((item * 73 + x) % 251 for x in range(300 + item)))
        tail.write_bytes(bytes((item * 19 + x) % 253 for x in range(70 + item)))
        lineage = f"{item + 1:032x}"
        run("pack", path, checkpoint, lineage, item + 1, item + 2, 99, tail)
        run("pack", path, checkpoint, lineage, item + 1, item + 1, 99, code=6)
        shown = run("inspect", path)
        check("verified generation 1 sections 3" in shown)
        check("fallback 0 trailing 0" in shown)
        checkpoint.write_bytes(bytes((item * 31 + x) % 247 for x in range(400 + item)))
        run("append", path, checkpoint, item + 2, item + 2, 100)
        check("verified generation 2 sections 2" in run("inspect", path))
        run("compact", path, compact)
        check(compact.stat().st_size < path.stat().st_size)
        check("verified generation 1 sections 2" in run("verify", compact))
        raw = path.read_bytes()
        run("compact", path, compact, code=6)
        check(path.read_bytes() == raw)
        files.append(path)
    verified = run("verify", *files)
    check(verified.count("verified generation 2 sections 2") == count)
    inspected = run("inspect", *files)
    check(inspected.count("lineage ") == count)
    for path in root.iterdir():
        path.unlink()


with tempfile.TemporaryDirectory(prefix="aotx-ccir-cli-") as directory:
    root = pathlib.Path(directory)
    run("--help")
    cases(root, 1)
    cases(root, 64)
    source = root / "source.bin"
    path = root / "invalid.aotxccir"
    source.write_bytes(b"state bytes")
    run("pack", path, source, "0" * 32, 0, 0, 0, code=2)
    run("pack", path, source, "a" * 32, -1, 0, 0, code=2)
    run("pack", path, source, "a" * 32, 0, 2**64, 0, code=2)
    run("pack", path, source, "a" * 32, 2, 1, 0, code=2)
    run("pack", path, source, "a" * 32, 0, 1, 0, code=2)
    check(not path.exists())
    run("pack", path, source, "a" * 32, 0, 0, 0)
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)
    run("inspect", path, code=2)
    run("verify", path, code=2)
    run("append", path, source, 1, 1, 1, code=2)
    run("compact", path, root / "copy.aotxccir", code=2)

print(f"ccir cli: {checks} checks, 0 failed")
