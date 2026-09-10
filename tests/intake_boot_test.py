#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Verify model-derived memory, corrections and recovery through real runtime files.
# Owns: One new output directory and only the runtime processes started here.
# Threading: One disk driver; all interpretation and memory processing run on CUDA.
# Lifetime: Initial operation, copied-file recovery and exact journal replay.

# Inputs: build, source, model store, new output and batch count 1 or 64.
# Output: commands, model results and exact checks. Exit: 0 pass, 1 failure, 2 bad arguments.
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Test, Run, wait
from text_boot_test import setup
from capacity_boot_test import batch
from checkpoint_boot_test import spawn, durable, file_state
from retain_boot_test import transfers, compiled_limits

FIRST = ("Mira", "Tomas", "Nora", "Jules", "Evan", "Rosa", "Liam", "Sara")
LAST = ("Vale", "March", "Lake", "Stone", "Reed", "Moss", "West", "Hill")


def person(i):
    return FIRST[i % 8] + " " + LAST[i // 8]


def text(i, turn, count):
    name = person(i)
    if turn == 1:
        prefix = ("This message describes a schedule for a small cooking group. The venue has a round table, "
                  "six chairs, and a window beside the door. The cupboard contains bowls, plates, napkins, "
                  "and a blue serving tray. ") if count == 1 else ""
        return prefix + name + " will cook lentils tonight. Reply with exactly one word: noted."
    if turn == 2:
        return "Correction: " + name + " will not cook lentils tonight. The earlier plan changed. Reply with exactly one word: noted."
    if turn == 3:
        return "Will " + name + " cook lentils tonight? Reply with exactly one word: yes or no."
    return name + " may bake bread tomorrow. This is uncertain. Reply with exactly one word: noted."


def request(f, count, cut, turn):
    data = batch(f, b"AOTXTXT1", 8256, cut, count)
    for i in range(count):
        at = 64 + i * 8256
        f.put(data, at, i, 4); data[at + 16:at + 32] = f.identity(8000 + i); f.put(data, at + 32, turn)
        q = at + 64; source = text(i, turn, count).encode()
        data[q:q + 16], data[q + 16:q + 32], data[q + 48:q + 64] = (
            f.identity(300000 + turn * 64 + i), f.identity(10000 + i), f.identity(400000 + turn * 64 + i))
        for offset, value in ((132, 16), (136, 4096), (148, len(source))):
            f.put(data, q + offset, value, 4)
        data[q + 4640:q + 4640 + len(source)] = source
    return data


def interpretations(test, f, state, count, turn):
    found = [[] for _ in range(count)]
    rows = f.state_rows(state)
    sources = {bytes(row[8:24]): (row, payload) for row, payload in rows}
    for row, payload in rows:
        if payload[:8] != b"AOTXMEM3":
            continue
        source = sources[bytes(row[96:112])]
        for i in range(count):
            if row[96:112] != f.identity(300000 + turn * 64 + i):
                continue
            raw = text(i, turn, count).encode(); start, length = f.get(payload, 20, 4), f.get(payload, 12, 4)
            test.check(source[1][32:] == raw and payload[96:] == raw[start:start + length],
                       "model interpretation preserves exact original source and byte span")
            test.check(row[64:80] == f.identity(10000 + i) and f.get(row, 176, 4) == 0 and
                       f.get(row, 180, 4) == 4 and f.get(row, 184, 4) == 0 and row[120:136] == bytes(16),
                       "interpreted content preserves private ownership and grants no identity")
            found[i].append((row, payload, payload[96:].decode()))
    return found


def exercise(test, count):
    f = setup(test); inputs = test.output / "inputs"
    checkpoint, ccir, bind = (inputs / name for name in ("checkpoint", "memory.aotxccir", "bind"))
    checkpoint.write_bytes(f.image([], 0, 10))
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, "-", ccir, 1], "pack-empty-memory")
    data = batch(f, b"AOTXBND1", 64, 0, count)
    for i in range(count):
        at = 64 + i * 64; f.put(data, at, i, 4)
        data[at + 8:at + 24], data[at + 40:at + 56] = f.identity(10000 + i), f.identity(8000 + i)
        f.put(data, at + 56, 160, 4); f.put(data, at + 60, 2, 4)
    bind.write_bytes(data)
    mirror = test.output / "identity.aotxccir"
    args = dict(extra=("--memory-mirror", mirror), roles="language,embedding")
    run = Run(test, "before", **args); run.ready(); spawn(run, count)
    run.operation("load", ccir, 1, 0); durable(run)
    run.operation("bind", bind, 3, count); durable(run)
    cut, old = 0, []
    for turn in (1, 2):
        path = inputs / f"input-{turn}"; path.write_bytes(request(f, count, cut, turn))
        run.operation("text", path, 6, count)
        for i in range(count):
            run.reply(i, turn, "noted")
        durable(run); saved = file_state(test, mirror, f, count, turn, mode=2)
        cut = f.get(saved[2], 32); found = interpretations(test, f, saved[2], count, turn)
        for i, values in enumerate(found):
            if turn == 1:
                assertions = [bytes(row[8:24]) for row, p, quote in values
                              if f.get(p, 16, 4) == 3 and person(i) in quote and "will cook lentils tonight" in quote]
                test.check(bool(assertions), "actual model extracts the stated plan", kind="behavior", slot=i, values=[v[2] for v in values])
                test.check(any(f.get(p, 16, 4) == 1 and person(i) == quote for _, p, quote in values),
                           "actual model extracts the participant mention", kind="behavior", slot=i)
                test.check(any(f.get(p, 16, 4) == 2 and "cook" in quote for _, p, quote in values),
                           "actual model extracts the task mention", kind="behavior", slot=i)
                old.append(assertions)
            else:
                test.check(any(f.get(p, 16, 4) == 4 and bytes(row[136:152]) in old[i] and
                               "will not cook lentils tonight" in quote for row, p, quote in values),
                           "actual model corrects its own prior assertion with negation intact", kind="behavior", slot=i,
                           values=[v[2] for v in values])
        if count == 1 and turn == 1:
            test.check(len(text(0, 1, count).encode()) > 192 and
                       any(f.get(p, 20, 4) >= 192 and "will cook" in quote for _, p, quote in found[0]),
                       "actual intake consumes source bytes beyond the old service limit", kind="behavior")
    limits = compiled_limits(test)
    records = transfers(test, run, f, {14: 64 + 64 * 13920 + limits["image_bytes"]})
    semantic = [p for op, _, p in records if op == 14]
    test.check(len(semantic) == 2 and all(f.get(p, 8, 4) == count for p in semantic), "both actual batches have complete semantic decisions")
    before = saved; run.stop(killed=True)
    portable = test.output / "copied.aotxccir"; shutil.copy2(mirror, portable)
    shutil.rmtree(inputs); shutil.rmtree(run.journal); mirror.unlink()
    test.check(not inputs.exists() and not run.journal.exists() and not mirror.exists(), "only the copied cognitive file remains from the prior runtime")
    args = dict(extra=("--memory-mirror", portable), roles="language,embedding")
    after = Run(test, "after", **args); after.ready(); spawn(after, count)
    after.operation("resume", portable, 11, count); durable(after)
    test.check(file_state(test, portable, f, count, 2, mode=2) == before, "copied file restores every accepted state byte")
    path = test.output / "next-input"; path.write_bytes(request(f, count, cut, 3))
    after.operation("text", path, 6, count)
    for i in range(count):
        after.reply(i, 3, "no")
    durable(after); file_state(test, portable, f, count, 3, mode=2)
    saved = file_state(test, portable, f, count, 3, mode=2)
    path = test.output / "uncertain-input"; path.write_bytes(request(f, count, f.get(saved[2], 32), 4))
    after.operation("text", path, 6, count)
    for i in range(count):
        after.reply(i, 4, "noted")
    durable(after); saved = file_state(test, portable, f, count, 4, mode=2)
    for i, values in enumerate(interpretations(test, f, saved[2], count, 4)):
        test.check(any(f.get(p, 16, 4) == 3 and person(i) in quote and "may bake bread tomorrow" in quote
                       for _, p, quote in values), "actual model retains uncertainty in its assertion",
                   kind="behavior", slot=i, values=[v[2] for v in values])
    after.stop(); summary = after.summary(); audits = [after.events(i) for i in range(count)]
    replay = Run(test, "replay", True, **args); replay.ready()
    report = wait(lambda: re.search(r"restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+) "
                  r"pages (\d+) paced (\d+) rejected (\d+)$", replay.path.read_text(), re.M), replay.child)
    test.check(int(report[1]) == int(summary["replayed"]) and int(report[2], 16) == int(summary["state_hash"], 16) and
               int(report[3]) == int(report[6]) == 0, "semantic journal replay preserves exact accepted state and token hash")
    wait(lambda: all(len(replay.events(i)) >= len(audits[i]) for i in range(count)), replay.child)
    test.check([replay.events(i) for i in range(count)] == audits, "replayed semantic input and response audits match")
    durable(replay); replay.stop()


def main():
    if len(sys.argv) != 6 or sys.argv[5] not in ("1", "64"):
        print("usage: intake_boot_test.py BUILD SOURCE STORE OUTPUT 1|64", file=sys.stderr)
        return 2
    test = Test(*(Path(p).resolve() for p in sys.argv[1:5]), snapshot_every=1024)
    status, start = 0, time.monotonic()
    try:
        exercise(test, int(sys.argv[5]))
    except (Exception, KeyboardInterrupt) as error:
        status = 1; (test.output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
    finally:
        for run in test.active:
            try:
                run.close()
            except Exception as error:
                status = 1; test.record(cleanup_error=str(error))
        test.flush_checks()
    failed = sum(not row["passed"] for row in test.checks)
    result = dict(checks=len(test.checks), failed=failed, seconds=time.monotonic() - start, exit=int(bool(status or failed)))
    (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    test.record(**result)
    print(f"semantic boot: {len(test.checks)} checks, {failed} failures", flush=True)
    return 1 if status or failed else 0


if __name__ == "__main__":
    sys.exit(main())
