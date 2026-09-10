#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Verify complete multi-pass source embeddings through the actual runtime.
# Owns: New input files, vector comparisons and only the processes started here.
# Threading: One disk test driver; tokenization and embedding run on CUDA.
# Lifetime: Two distinct source batches and a durable memory mirror.

# Inputs: build, source, model store, new output and batch count 1 or 64.
# Output: commands, exact source checks and vectors. Exit: 0 pass, 1 failure, 2 bad arguments.
import json
from pathlib import Path
import struct
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Test, Run
from text_boot_test import setup
from capacity_boot_test import batch
from checkpoint_boot_test import spawn, durable, file_state


def source(i):
    return f"Reply with exactly one word: noted. These are crate labels for row {i}: " + "1 3 5 7 9 " * 110 + "End of labels."


def exercise(test, count):
    f = setup(test); inputs = test.output / "inputs"
    checkpoint, ccir, bind = (inputs / name for name in ("checkpoint", "memory.aotxccir", "bind"))
    checkpoint.write_bytes(f.image([], 0, 10))
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, "-", ccir, 1], "pack-empty")
    bindings = batch(f, b"AOTXBND1", 64, 0, count)
    for i in range(count):
        at = 64 + i * 64; f.put(bindings, at, i, 4)
        bindings[at + 8:at + 24], bindings[at + 40:at + 56] = f.identity(10000 + i), f.identity(8000 + i)
        f.put(bindings, at + 56, 160, 4); f.put(bindings, at + 60, 1, 4)
    bind.write_bytes(bindings)
    mirror = test.output / "memory.aotxccir"
    run = Run(test, "extent", extra=("--memory-mirror", mirror), roles="language,embedding"); run.ready(); spawn(run, count)
    run.operation("load", ccir, 1, 0); durable(run)
    run.operation("bind", bind, 3, count); durable(run)
    cut, vectors = 0, []
    for turn in (1, 2):
        request = batch(f, b"AOTXTXT1", 8256, cut, count)
        expected = []
        for i in range(count):
            at = 64 + i * 8256; f.put(request, at, i, 4)
            request[at + 16:at + 32] = f.identity(8000 + i); f.put(request, at + 32, turn)
            q = at + 64; text = source(i).encode(); text = text[:192] if turn == 1 else text
            expected.append(text)
            request[q:q + 16], request[q + 16:q + 32], request[q + 48:q + 64] = (
                f.identity(300000 + turn * 64 + i), f.identity(10000 + i), f.identity(400000 + turn * 64 + i))
            for offset, value in ((132, 16), (136, 1024), (148, len(text))): f.put(request, q + offset, value, 4)
            request[q + 4640:q + 4640 + len(text)] = text
        path = inputs / f"input-{turn}"; path.write_bytes(request)
        run.operation("text", path, 6, count)
        for i in range(count): run.reply(i, turn, "noted")
        durable(run); state = file_state(test, mirror, f, count, turn); cut = f.get(state[2], 32)
        rows = f.state_rows(state[2]); current = []
        for i in range(count):
            event = f.identity(300000 + turn * 64 + i)
            originals = [p for r, p in rows if r[8:24] == event]
            embedded = [p for r, p in rows if p[:8] == b"AOTXVEC2" and p[88:104] == event]
            test.check(len(originals) == len(embedded) == 1 and originals[0][32:] == expected[i], "complete source has one exact vector dependency")
            p = embedded[0]; width = f.get(p, 12, 4); values = struct.unpack("<" + "f" * width, p[128:])
            test.check(all(abs(v) <= 1 for v in values) and abs(sum(v * v for v in values) - 1) < 0.001,
                       "actual complete-source vector is finite and normalized")
            if turn == 2:
                test.check(p[56:88] != vectors[i][56:88] and p[24:56] == vectors[i][24:56], "long extent has its named processor and the same embedding model")
                short = struct.unpack("<" + "f" * width, vectors[i][128:])
                test.check(sum((x - y) ** 2 for x, y in zip(values, short)) > 0.000001,
                           "actual complete embedding differs from its first 192 bytes", kind="behavior", slot=i)
            current.append(p)
        vectors = current
    run.stop(); test.record(result="complete", count=count)


def main():
    if len(sys.argv) != 6 or sys.argv[5] not in ("1", "64"):
        print("usage: text_extent_boot_test.py BUILD SOURCE STORE OUTPUT 1|64", file=sys.stderr); return 2
    test = Test(*(Path(p).resolve() for p in sys.argv[1:5]), snapshot_every=1024)
    status, start = 0, time.monotonic()
    try: exercise(test, int(sys.argv[5]))
    except (Exception, KeyboardInterrupt) as error:
        status = 1; (test.output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
    finally:
        for run in reversed(test.active):
            try: run.close()
            except Exception as error: status = 1; test.record(cleanup_error=str(error))
        test.flush_checks()
    failed = sum(not row["passed"] for row in test.checks)
    result = dict(checks=len(test.checks), failed=failed, seconds=time.monotonic() - start, exit=int(bool(status or failed)))
    (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)
    return result["exit"]


if __name__ == "__main__": sys.exit(main())
