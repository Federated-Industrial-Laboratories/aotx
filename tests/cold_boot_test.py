#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check explicit cold storage and repeated recovery through the real console.
# Inputs: build, source, model store, new output directory and batch size.
# Outputs: exact file checks, process logs and a completion receipt. Exit: 0 pass, 1 failure, 2 usage.
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
from runtime_boot_test import RuntimeTest, RuntimeRun, durable, sections
from text_boot_test import setup
from live_boot_test import wait


def control(f, count, mode):
    n = count if mode in (1, 2) else 0
    data = bytearray(64 + 64 * n); data[:8] = b"AOTXTIR1"
    for at, value in ((8, 1), (12, mode), (16, n), (20, 64)):
        f.put(data, at, value, 4)
    data[24:40] = f.LINEAGE; f.put(data, 40, count)
    for i in range(n):
        at = 64 + 64 * i
        data[at:at + 16] = f.identity(700000 + i); f.put(data, at + 16, 1)
        data[at + 24:at + 40] = f.identity(10000 + i)
    return data


def status(run, tiered, count, size):
    before = len(run.console()); run.send("memory")
    pattern = r"memory tier: (\d+) cold objects (\d+) cold bytes (\d+) read pending (\d+)"
    match = wait(lambda: re.search(pattern, run.console()[before:]), run.child)
    run.test.check(tuple(map(int, match.groups())) == (tiered, count, size, 0),
                   "console reports the complete residency state")


def operation(run, f, count, mode):
    path = run.test.output / "tier-request"
    path.write_bytes(control(f, count, mode))
    run.operation("tier", path, 18, count if mode in (1, 2, 3) else 0)
    durable(run)
    path.unlink()


def exercise(test, count):
    f = setup(test); values = []
    for i in range(count):
        text = f"Private source {i}: retain exactly this payload.".encode()
        payload = bytearray(32) + text; payload[:8] = b"AOTXMEM1"
        f.put(payload, 8, 1, 4); f.put(payload, 12, len(text), 4)
        values.append((f.object_row(1, 700000 + i, 10000 + i, i + 1), payload))
    inputs = test.output / "inputs"
    checkpoint, memory = inputs / "checkpoint", inputs / "memory.aotxccir"
    initial = f.image(values, count, 10); checkpoint.write_bytes(initial)
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, "-", memory, 1], "prepared-memory")
    runtime = test.output / "state.aotxccir"
    test.command([test.build / "aotx_ccir_pack", "--memory", memory, "--models", test.store,
                  "--modules", test.output / "modules", "--settings", test.output / "settings",
                  "--roles", "language", "--output", runtime, "--phrases",
                  test.source / "tests/fixtures/quality/refusal-phrases.txt"], "pack-runtime")
    with runtime.open("rb") as stream:
        stream.seek(40); incarnation = stream.read(16)
    shutil.rmtree(inputs); shutil.rmtree(test.output / "modules"); (test.output / "settings").unlink()
    total = sum(len(p) for _, p in values)
    run = RuntimeRun(test, "offload", runtime); run.ready(); durable(run)
    operation(run, f, count, 4); operation(run, f, count, 1)
    status(run, 1, count, total); run.stop(killed=True)
    cold = sections(test, runtime, (8,))
    with runtime.open("rb") as stream:
        stream.seek(40)
        test.check(stream.read(16) == incarnation, "offload appends extents without copying the complete model file")
    test.check(f.get(cold[2], 8, 4) == 3 and not f.get(cold[2], 24), "cold checkpoint has no resident payload")
    test.check(f.get(cold[5], 8, 4) == 1 and f.get(cold[5], 20, 4) & 128,
               "complete runtime declares the required cold-memory feature")
    test.check(f.get(cold[8], 12, 4) == count and f.get(cold[8], 24) == total,
               "the complete file contains every cold extent")
    copied = test.output / "copy.aotxccir"
    test.command([test.build / "aotx_ccir", "compact", runtime, copied], "copy-runtime")
    runtime.unlink(); shutil.rmtree(run.journal)
    previous = run
    for ordinal in (1, 2):
        run = RuntimeRun(test, f"recover-{ordinal}", copied); run.ready(); durable(run)
        status(run, 1, count, total)
        operation(run, f, count, 2 if ordinal == 1 else 3)
        status(run, int(ordinal == 1), 0, 0); run.stop()
        restored = sections(test, copied, (8,))
        test.check(f.get(restored[2], 20, 4) == count and f.get(restored[2], 24) == total,
                   "file-only recovery restores the complete payload batch")
        test.check(restored[2][128:] == initial[128:], "private metadata and payload bytes survive recovery exactly")
        test.check(not f.get(restored[8], 12, 4), "resident objects leave no selected cold extent")
        shutil.rmtree(run.journal)
        if ordinal == 1:
            previous = RuntimeRun(test, "offload-again", copied); previous.ready(); durable(previous)
            operation(previous, f, count, 1); previous.stop(); shutil.rmtree(previous.journal)
    test.check(not runtime.exists() and not inputs.exists(), "repeated recovery uses only the copied complete file")


def main():
    if len(sys.argv) != 6 or sys.argv[5] not in ("1", "64"):
        print("usage: cold_boot_test.py BUILD SOURCE STORE OUTPUT 1|64", file=sys.stderr); return 2
    build, source, store, output = (Path(v).resolve() for v in sys.argv[1:5])
    test = RuntimeTest(build, source, store, output, snapshot_every=64)
    begin, result = time.monotonic(), 0
    try:
        exercise(test, int(sys.argv[5]))
    except Exception as error:
        result = 1; (output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
        print(f"cold boot failed: {error}", file=sys.stderr, flush=True)
    finally:
        for run in reversed(test.active):
            try:
                run.close()
            except Exception as error:
                result = 1; test.record(cleanup_error=str(error))
        test.flush_checks()
    result |= any(not c["passed"] for c in test.checks)
    receipt = dict(status=result, checks=len(test.checks), seconds=time.monotonic() - begin)
    (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True); return result


if __name__ == "__main__":
    sys.exit(main())
