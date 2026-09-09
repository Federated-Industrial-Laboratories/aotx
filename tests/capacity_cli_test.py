#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check typed checkpoint portability between two compiled memory capacities.
# Owns: One new output directory with independent source files and command logs.
# Threading: Serialized file commands with distinct batches of 1 and 64 objects.
# Lifetime: One bounded test; no model files or persistent runtime are used.

# Inputs: large build, small build, source, new output. Output: logs and checks.
# Exit: 0 pass, 1 failed check, 2 bad arguments.
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True


class Check:
    def __init__(self, output):
        self.output, self.checks, self.commands = output, [], 0
        output.mkdir(parents=True, exist_ok=False)

    def check(self, value, label):
        self.checks.append(dict(check=label, passed=bool(value)))
        (self.output / "checks.json").write_text(json.dumps(self.checks, indent=2) + "\n")
        if not value:
            raise AssertionError(label)

    def command(self, args, status=0, no_gpu=False):
        self.commands += 1
        args = list(map(str, args)); env = dict(os.environ)
        if no_gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        start = time.monotonic()
        result = subprocess.run(args, capture_output=True, env=env, timeout=180)
        stem = self.output / f"command-{self.commands:03d}"
        stem.with_suffix(".stdout").write_bytes(result.stdout)
        stem.with_suffix(".stderr").write_bytes(result.stderr)
        with (self.output / "commands.jsonl").open("a") as log:
            log.write(json.dumps(dict(command=args, exit=result.returncode, expected=status,
                no_gpu=no_gpu, seconds=time.monotonic() - start, output=str(stem))) + "\n")
        self.check(result.returncode == status, f"command {self.commands} exit {status}")
        return result.stdout.decode()


def limits(test, build):
    text = test.command([build / "aotx_ccir_state", "--limits"], no_gpu=True)
    match = re.fullmatch(r"objects=(\d+) payload_bytes=(\d+) image_bytes=(\d+)\n", text)
    test.check(match is not None, "exact capacity fields without a GPU")
    objects, payload, image = map(int, match.groups())
    test.check(objects > 0 and payload > 0 and image == 128 + 256 * objects + payload and
               16 + 2 * image <= 0xFFFFFFFF, "capacity arithmetic and transport bound")
    return objects, payload, image


def fixture(f, count, payload_bytes):
    parts = []
    for i in range(count):
        length = payload_bytes // count + (i < payload_bytes % count)
        row = f.object_row(1, 50000 + i, 10000 + i, i + 1)
        text = (f"object {i}: ".encode() + bytes((65 + i % 26,)))
        value = (text * ((length + len(text) - 1) // len(text)))[:length]
        parts.append((row, value))
    return f.image(parts, count, 10)


def exercise(test, large, small, source):
    spec = importlib.util.spec_from_file_location("capacity_bytes", source / "tests/recall_cli_test.py")
    f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    bigger, smaller = limits(test, large), limits(test, small)
    test.check(smaller[0] >= 64 and bigger[0] >= smaller[0] + 64 and bigger[1] >= smaller[1] + 64,
               "builds provide distinct object and payload limits")
    for n in (1, 64):
        cases = (("compact", n, n * 17, True), ("objects", smaller[0] + n, smaller[0] + n, False),
                 ("payload", n, smaller[1] + n, False))
        for name, count, size, fits in cases:
            prefix = test.output / f"{name}-{n}"
            raw, ccir = prefix.with_suffix(".raw"), prefix.with_suffix(".aotxccir")
            image = fixture(f, count, size); raw.write_bytes(image)
            test.command([large / "aotx_recall_cli_fixture", raw, "-", ccir, n])
            digest = hashlib.sha256(ccir.read_bytes()).digest()
            for label, build in (("large", large), ("small", small)):
                output = test.output / f"{name}-{n}-{label}.aotxccir"
                admitted = fits or label == "large"
                test.command([build / "aotx_ccir_state", ccir, output], 0 if admitted else 1)
                test.check(hashlib.sha256(ccir.read_bytes()).digest() == digest, "source remains exact")
                test.check(output.exists() == admitted, "refusal publishes no output")
                if admitted:
                    _, sections = f.container(output)
                    saved = [v[3] for v in sections.values() if v[0] == 2]
                    test.check(saved == [image], "exact schema-1 checkpoint across builds")
                    _, original = f.container(ccir)
                    test.check(sections == original, "optional extents survive export")
    test.check(f.CHECKS > 0, "independent container digest checks executed")
    return f.CHECKS


def main():
    if len(sys.argv) != 5:
        print("usage: capacity_cli_test.py LARGE_BUILD SMALL_BUILD SOURCE OUTPUT", file=sys.stderr); return 2
    large, small, source, output = (Path(v).resolve() for v in sys.argv[1:])
    test, status, hashes = Check(output), 0, 0
    try:
        hashes = exercise(test, large, small, source)
    except Exception as error:
        status = 1; (output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
    result = dict(checks=len(test.checks), hash_checks=hashes,
                  failed=sum(not r["passed"] for r in test.checks), exit=status)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
