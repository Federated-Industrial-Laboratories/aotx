#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Verify GPU reclamation, file shrinking and fresh model input after CCIR recovery.
# Owns: One output directory and only the boot processes started here.
# Threading: One disk driver; CUDA owns memory and model processing.
# Lifetime: Two real runtime instances with the same external model configuration.

# Inputs: build, source, store, new output and vector-1, vector-64 or text-1.
# Output: command logs and exact checks. Exit: 0 pass, 1 failure, 2 bad arguments.
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Test, Run, wait
from text_boot_test import setup
from capacity_boot_test import batch, color, memory
from context_boot_bytes import corpus, correction, request, selected
from retain_boot_test import transfers, compiled_limits


from checkpoint_boot_test import spawn, durable, file_state


def maintain(test, run, f, mirror, count, ordinal, minimum):
    before = file_state(test, mirror, f, count, ordinal)
    state = before[2]; size = mirror.stat().st_size
    policy = bytearray(64); policy[:8] = b"AOTXMNT1"
    f.put(policy, 8, 1, 4); f.put(policy, 20, 1, 4); f.put(policy, 28, 80, 4)
    f.put(policy, 32, f.get(state, 32)); f.put(policy, 40, f.get(state, 96))
    policy[48:64] = state[48:64]
    path = test.output / "maintenance-policy"; path.write_bytes(policy)
    pattern = r"memory: operation (\d+) status (\d+) rows (\d+)"
    start = len(re.findall(pattern, run.console())); begin = time.monotonic()
    run.send(f"memory maintain {path}")
    def completed():
        found = re.findall(pattern, run.console())
        return found[start] if len(found) > start else None
    op, status, removed = map(int, wait(completed, run.child))
    test.check(op == 13 and status == 0 and removed >= minimum, "real runtime releases eligible memory", removed=removed)
    durable(run); after = file_state(test, mirror, f, count, ordinal)
    test.check(f.get(after[2], 8, 4) == 2 and f.get(after[2], 96) == f.get(state, 32) and
               f.get(after[2], 20, 4) + removed == f.get(state, 20, 4), "compacted checkpoint has the exact root and live count")
    if minimum:
        test.check(mirror.stat().st_size < size // 2, "continuous mirror shrinks after GPU reclamation")
    old = {bytes(r[8:24]) + bytes(r[40:48]): (r, p) for r, p in f.state_rows(state)}
    for row, payload in f.state_rows(after[2]):
        prior, original = old[bytes(row[8:24]) + bytes(row[40:48])]
        test.check(row[:160] == prior[:160] and row[168:] == prior[168:] and payload == original,
                   "every retained object keeps exact metadata and payload")
    test.record(maintenance_seconds=time.monotonic() - begin, old_file_bytes=size,
                file_bytes=mirror.stat().st_size, old_state_bytes=len(state), state_bytes=len(after[2]), removed=removed)
    return after


def exercise(test, count, text):
    f = setup(test); values = corpus(f, count, text); limits = compiled_limits(test)
    fill = limits["objects"] // 2
    for r, _ in values:
        f.put(r, 48, f.get(r, 48) + fill); f.put(r, 56, f.get(r, 56) + fill)
    size = limits["payload_bytes"] // limits["objects"] - 32
    prefix = [(f.object_row(1, 9000000 + i, 9100000 + i, i + 1),
               memory(f, (str(i) + ":" + chr(65 + i % 26) * size)[:size])) for i in range(fill)]
    values = prefix + values; cut = len(values)
    inputs = test.output / "inputs"; mirror = test.output / "state.aotxccir"
    checkpoint, ccir, bind = (inputs / name for name in ("checkpoint", "memory.aotxccir", "bind"))
    checkpoint.write_bytes(f.image(values, cut, 10))
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, "-", ccir, 1], "pack")
    data = batch(f, b"AOTXBND1", 64, cut, count)
    for i in range(count):
        at = 64 + i * 64; f.put(data, at, i, 4)
        data[at + 8:at + 24], data[at + 40:at + 56] = f.identity(10000 + i), f.identity(8000 + i)
        f.put(data, at + 56, 160, 4); f.put(data, at + 60, 1, 4)
    bind.write_bytes(data)
    args = dict(extra=("--memory-mirror", mirror), roles="language,embedding" if text else "language")
    kind, op = ("text", 6) if text else ("query", 4)
    run = Run(test, "before", **args); run.ready(); spawn(run, count)
    run.operation("load", ccir, 1, 0); durable(run)
    run.operation("bind", bind, 3, count); durable(run)
    for turn in (1, 2, 3):
        if turn == 3:
            path = inputs / "correction"; path.write_bytes(correction(f, count, cut))
            run.operation("apply", path, 2, 0); cut += 2 * count; durable(run)
        path = inputs / f"input-{turn}"; path.write_bytes(request(f, count, cut, turn, text))
        run.operation(kind, path, op, count); cut += 3 * count
        for i in range(count):
            run.reply(i, turn, "unknown" if turn == 2 else color(i, turn > 2))
        durable(run)
    before = maintain(test, run, f, mirror, count, 3, fill)
    run.stop(killed=True)
    original = list(inputs.iterdir()); shutil.rmtree(inputs); shutil.rmtree(run.journal)
    test.check(not run.journal.exists() and all(not path.exists() for path in original),
               "original inputs and the entire prior runtime journal are removed")
    after = Run(test, "after", **args); after.ready(); spawn(after, count)
    after.operation("resume", mirror, 11, count); durable(after)
    test.check(file_state(test, mirror, f, count, 3) == before, "resume keeps the exact durable file sections")
    test.check(not after.turns(), "fresh runtime contains no replayed conversation log")
    path = test.output / "next-input"; raw = request(f, count, cut, 4, text); path.write_bytes(raw)
    after.operation(kind, path, op, count); cut += 3 * count
    for i in range(count):
        after.reply(i, 4, color(i, True))
        audit = [r for r in after.events(i) if r.get("kind") == "selection" and r.get("turn") == 4]
        expected = selected(f, i, 4, text)
        test.check(len(audit) == 1 and audit[0]["text"].endswith(" objects " + ",".join(k.hex() + "@1" for k in expected)),
                   "new input recalls the exact corrected requirement and source-paired appraisal")
    durable(after)
    state = maintain(test, after, f, mirror, count, 4, 0)
    test.check(f.get(state[2], 32) == cut, "continued retention advances the exact object sequence")
    records = transfers(test, after, f, {11: len(before[4]) + len(before[2])})
    test.check(sum(r[0] == 11 for r in records) == 1 and sum(r[0] == 10 for r in records) == 1 and
               next(r[2] for r in records if r[0] == 11) == before[4] + before[2],
               "new journal records one resume and one fresh automatic decision")
    after.stop()
    old = after.summary(); audits = [after.events(i) for i in range(count)]
    replay = Run(test, "replay", True, **args); replay.ready()
    report = wait(lambda: re.search(r"restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+) "
                  r"pages (\d+) paced (\d+) rejected (\d+)$", replay.path.read_text(), re.M), replay.child)
    test.check(int(report[1]) == int(old["replayed"]) and int(report[2], 16) == int(old["state_hash"], 16) and
               int(report[3]) == int(report[6]) == 0, "journal replay after file resume preserves the exact device hash")
    wait(lambda: all(len(replay.events(i)) >= len(audits[i]) for i in range(count)), replay.child)
    test.check([replay.events(i) for i in range(count)] == audits, "replayed resume restores exact transcript turn counters")
    durable(replay); replay.stop()
    test.check(all(not p.exists() for p in original), "fresh recall needs no original input file")


def main():
    if len(sys.argv) != 6 or sys.argv[5] not in ("vector-1", "vector-64", "text-1"):
        print("usage: checkpoint_boot_test.py BUILD SOURCE STORE OUTPUT vector-1|vector-64|text-1", file=sys.stderr)
        return 2
    count, text = int(sys.argv[5].split("-")[1]), sys.argv[5].startswith("text")
    test = Test(*(Path(p).resolve() for p in sys.argv[1:5]), snapshot_every=1024)
    status, start = 0, time.monotonic()
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
        if any(not row["passed"] for row in test.checks):
            status = 1
        test.flush_checks()
        result = dict(batch=count, text=text, checks=len(test.checks), failed=sum(not r["passed"] for r in test.checks),
                      seconds=time.monotonic() - start, exit=status)
        (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
