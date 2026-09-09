#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check contextual recall, corrections and exact cold recovery through real model replies.
# Owns: One new output directory and only the processes started here.
# Threading: One disk driver; CUDA owns encoding, recall, retention and model processing.
# Lifetime: One bounded run with local models and removal of original input files.

# Inputs: build, source, model store, new output and vector-1, vector-64 or text-1.
# Output: portable inputs, logs and exact checks. Exit: 0 pass, 1 failure, 2 bad arguments.
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Test, Run, FIELDS, wait
from text_boot_test import TextRun, setup, PROCESSOR
from retain_boot_test import transfers
from capacity_boot_test import batch, color
from context_boot_bytes import corpus, correction, request, selected


def inspect(test, run, f, count, text, turns, initial_cut):
    records = transfers(test, run, f)
    inputs = [r for r in records if r[0] == (6 if text else 4)]
    choices = [r for r in records if r[0] == 10]
    test.check(len(inputs) == len(choices) == turns, "each contextual input has one complete automatic decision")
    for turn, (raw, out) in enumerate(zip(inputs, choices), 1):
        cut = initial_cut + (turn - 1) * 3 * count + (2 * count if turn > 2 else 0)
        test.check(raw[1] == out[1] and raw[2] == request(f, count, cut, turn, text), "exact contextual source input and decision identity")
        data = out[2]
        test.check(data[:8] == b"AOTXACH1" and f.get(data, 8, 4) == count and f.get(data, 32) == cut and
                   f.get(data, 40, 4) == 9168 and not f.get(data, 44, 4), "complete successful contextual decision")
        tail = data[64 + count * 9168:]
        test.check(len(tail) == f.get(data, 48) and f.get(tail, 20, 4) == 3 * count and
                   f.get(tail, 32) == cut + 1, "automatic tail follows the corrected store cut")
        for i in range(count):
            base = 64 + i * 9168; q = data[base + 64:base + 8256]; choice = data[base + 8256:base + 8784]
            original = raw[2][128 + i * 8256:8320 + i * 8256]; expected = selected(f, i, turn, text)
            test.check(q[4640:] == original[4640:] and q[:64] == original[:64], "exact text, task, participant and authority bytes")
            if text:
                test.check(not any(original[64:132]) and not any(original[160:4256]) and q[96:128] == PROCESSOR and
                           0 < f.get(q, 128, 4) <= 1024, "GPU text preparation supplies the actual encoded query")
            else:
                test.check(q == original, "prepared-vector query survives without a byte change")
            test.check(f.get(choice, 4, 4) == len(expected) and
                       [choice[16 + j * 32:32 + j * 32] for j in range(len(expected))] == expected and
                       all(f.get(choice, 32 + j * 32) == 1 for j in range(len(expected))), "exact required and source-paired selection")
            audit = [r for r in run.events(i) if r.get("kind") == "selection" and r.get("turn") == turn]
            test.check(len(audit) == 1 and audit[0]["text"].endswith(" objects " + ",".join(k.hex() + "@1" for k in expected)),
                       "disk audit contains the exact contextual selection")
            test.check(f"principal {f.identity(10000 + i).hex()} " in audit[0]["text"] and "scope 0 " in audit[0]["text"],
                       "each audit preserves its distinct private principal")
        stored = f.state_rows(tail)
        test.check(len(stored) == 3 * count and all(f.get(r, 2, 2) in (1, 7, 9) and
                   f.get(r, 192, 4) == 0xFFFFFFFF for r, p in stored), "recall does not fabricate or strengthen appraisals")
    return records


def exercise(test, count, text):
    f = setup(test); values = corpus(f, count, text); cut = len(values); inputs = test.output / "inputs"
    checkpoint, ccir, bind = (inputs / p for p in ("checkpoint", "memory.aotxccir", "bind"))
    checkpoint.write_bytes(f.image(values, cut, 10))
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, "-", ccir, 1], "pack")
    data = batch(f, b"AOTXBND1", 64, cut, count)
    for i in range(count):
        base = 64 + i * 64; f.put(data, base, i, 4)
        data[base + 8:base + 24], data[base + 40:base + 56] = f.identity(10000 + i), f.identity(8000 + i)
        f.put(data, base + 56, 160, 4); f.put(data, base + 60, 1, 4)
    bind.write_bytes(data)
    kind, op, cls = ("text", 6, TextRun) if text else ("query", 4, Run)
    run = cls(test, "before"); run.ready()
    for start in range(1, max(count, 2), 8):
        size = min(8, max(count, 2) - start); run.send(f"spawn worker {size}")
        expected = "spawn: worker on slots " + " ".join(str(i) for i in range(start, start + size))
        wait(lambda: expected in run.console(), run.child)
    if count == 1:
        run.send("agent 1 pages 160"); run.send("task 1 Reply with the word ready."); Run.reply(run, 1, 1, "ready")
    run.operation("load", ccir, 1, 0); run.operation("bind", bind, 3, count)
    for turn in (1, 2, 3):
        current = cut + (turn - 1) * 3 * count
        if turn == 3:
            path = inputs / "correction"; path.write_bytes(correction(f, count, current))
            run.operation("apply", path, 2, 0); current += 2 * count
        path = inputs / f"input-{turn}"; path.write_bytes(request(f, count, current, turn, text))
        run.operation(kind, path, op, count)
        for i in range(count):
            Run.reply(run, i, turn, "unknown" if turn == 2 else color(i, turn > 2))
    run.summary(); run.stop(killed=True)
    old, manifests, audits = run.summary(), run.turns(), [run.events(i) for i in range(count)]
    original = inspect(test, run, f, count, text, 3, cut)
    paths = list(inputs.iterdir()); shutil.rmtree(inputs)
    test.check(all(not p.exists() for p in paths), "all original container, memory, correction, binding and input files are removed")
    after = cls(test, "after", True); after.ready()
    report = wait(lambda: re.search(r"restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+) "
                  r"pages (\d+) paced (\d+) rejected (\d+)$", after.path.read_text(), re.M), after.child)
    test.check(int(report[1]) == int(old["replayed"]) and int(report[2], 16) == int(old["state_hash"], 16) and
               int(report[3]) == int(report[6]) == 0, "exact restore count and hash without refused work")
    def durable():
        current = after.summary()
        return current if current["boot"] == after.boot and current.get("restore_hash") != "none" else None
    current = wait(durable, after.child)
    test.check(current.get("restore_of") == old["boot"] and current["restore_hash"] == old["state_hash"], "durable restore parent and state hash")
    wait(lambda: len(after.turns()) >= len(manifests) and all(len(after.events(i)) >= len(audits[i]) for i in range(count)), after.child)
    keys = lambda data: sorted(tuple(r[k] for k in FIELDS) for r in data)
    test.check(keys(after.turns()) == keys(manifests) and [after.events(i) for i in range(count)] == audits,
               "cold recovery preserves exact prompt hashes and complete conversation logs")
    test.check(transfers(test, after, f) == original, "recovery preserves every contextual descriptor, correction and automatic decision")
    path = test.output / "next-input"; path.write_bytes(request(f, count, cut + 11 * count, 4, text))
    after.operation(kind, path, op, count)
    for i in range(count):
        Run.reply(after, i, 4, color(i, True))
    after.stop(); after.summary(); inspect(test, after, f, count, text, 4, cut)
    test.check(all(not p.exists() for p in paths), "source files remain absent after new input recalls the correction")
    test.command([test.build / "aotx_journal", "manifest", after.journal], "manifest-chain")


def main():
    if len(sys.argv) != 6 or sys.argv[5] not in ("vector-1", "vector-64", "text-1"):
        print("usage: context_boot_test.py BUILD SOURCE STORE OUTPUT vector-1|vector-64|text-1", file=sys.stderr); return 2
    text, count = sys.argv[5].startswith("text"), int(sys.argv[5].split("-")[1])
    test = Test(*(Path(v).resolve() for v in sys.argv[1:5]), snapshot_every=1024); status, start = 0, time.monotonic()
    try:
        exercise(test, count, text)
    except (Exception, KeyboardInterrupt) as error:
        status = 1; (test.output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
    finally:
        for run in reversed(test.active):
            try:
                run.close()
            except Exception as error:
                status = 1; test.record(cleanup_error=str(error))
        if any(not r["passed"] for r in test.checks):
            status = 1
        test.flush_checks()
        result = dict(batch=count, text=text, checks=len(test.checks), failed=sum(not r["passed"] for r in test.checks),
                      seconds=time.monotonic() - start, exit=status, claim="contextual recall integration")
        (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
