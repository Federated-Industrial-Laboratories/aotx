#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check exact source classification through a resident model and native intake.
# Inputs: build, source, model store, new output and batch count 1 or 64.
# Outputs: raw model decisions, commands and checks. Exit: 0 pass, 1 failure, 2 bad arguments.
import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Test, Run, wait
from text_boot_test import setup
from capacity_boot_test import batch
from checkpoint_boot_test import spawn, durable
from retain_boot_test import transfers, compiled_limits
from source_boot_test import decisions
from source_cases import CASE_COUNT, source_case


def exercise(test, count, offset):
    f = setup(test)
    cases = [source_case((offset + i) % CASE_COUNT, i + 1 if count > 1 else 0) for i in range(count)]
    (test.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    inputs = test.output / "inputs"
    checkpoint, ccir, bind, source_path = (inputs / name for name in ("checkpoint", "memory.aotxccir", "bind", "source"))
    checkpoint.write_bytes(f.image([], 0, 10))
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, "-", ccir, 1], "pack-empty-memory")
    data = batch(f, b"AOTXBND1", 64, 0, count)
    for i in range(count):
        at = 64 + i * 64; f.put(data, at, i, 4)
        data[at + 8:at + 24], data[at + 40:at + 56] = f.identity(10000 + i), f.identity(8000 + i)
        f.put(data, at + 56, 512, 4); f.put(data, at + 60, 2, 4)
    bind.write_bytes(data)
    data = batch(f, b"AOTXTXT1", 8256, 0, count)
    for i, (source, _) in enumerate(cases):
        at = 64 + i * 8256; f.put(data, at, i, 4)
        data[at + 16:at + 32] = f.identity(8000 + i); f.put(data, at + 32, 1)
        q = at + 64; raw = source.encode()
        data[q:q + 16], data[q + 16:q + 32], data[q + 48:q + 64] = (
            f.identity(300000 + i), f.identity(10000 + i), f.identity(400000 + i))
        for field, value in ((132, 16), (136, 4096), (148, len(raw))):
            f.put(data, q + field, value, 4)
        data[q + 4640:q + 4640 + len(raw)] = raw
        c = q + 6688; data[c:c + 8] = b"AOTXCTX2"
        f.put(data, c + 8, 2, 4); f.put(data, c + 44, 2, 4)
        data[q + 7760:q + 7776] = f.identity(20000 + i)
    source_path.write_bytes(data)
    run = Run(test, "classify", roles="language,embedding", extra=("--memory-mirror", test.output / "state.aotxccir"))
    run.ready(); spawn(run, count)
    run.operation("load", ccir, 1, 0, seconds=test.operation_seconds); durable(run)
    run.operation("bind", bind, 3, count, seconds=test.operation_seconds); durable(run)
    run.operation("text", source_path, 6, count, seconds=test.operation_seconds)
    for i in range(count):
        turn = wait(lambda: next((r for r in run.turns() if r.get("agent") == i and r.get("turn") == 1), None),
                    run.child, seconds=test.operation_seconds)
        reply = wait(lambda: next((r for r in run.events(i) if r.get("kind") == "reply" and r.get("turn") == 1), None),
                     run.child, seconds=test.operation_seconds)
        test.check(0 < turn.get("tokens", 0) <= 32 and bool(reply.get("text", "").strip()),
                   "each native response completes within its token limit", slot=i, turn=turn, reply=reply)
        test.check(not any(r.get("status") == "prompt_refused" for r in run.events(i)),
                   "the full source prompt fits its page binding", slot=i)
    durable(run)
    limits = compiled_limits(test)
    choices = [p for op, _, p in transfers(test, run, f, {14: 64 + 64 * 18672 + limits["image_bytes"]}) if op == 14]
    test.check(len(choices) == 1 and f.get(choices[0], 8, 4) == count, "the native batch records every source decision")
    decisions(test, f, choices, count, {source: [pair[0] for pair in expected] for source, expected in cases})
    choice = choices[0]
    for i, (_, expected) in enumerate(cases):
        at = 64 + i * 18672 + 14448
        raw = choice[at + 128:at + 128 + f.get(choice, at + 4, 4)].decode()
        test.check(json.loads(raw) == expected, "actual model classifies each exact source sentence",
                   kind="behavior", slot=i, case=(offset + i) % CASE_COUNT, expected=expected, raw=raw)
    run.stop()


def main():
    parser = argparse.ArgumentParser()
    for name in ("build", "source", "store", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("count", type=int, choices=(1, 64))
    parser.add_argument("--case", type=int, default=0, choices=range(CASE_COUNT))
    parser.add_argument("--operation-seconds", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.operation_seconds <= 7200:
        parser.error("operation seconds must be between 1 and 7200")
    test = Test(*(getattr(args, name).resolve() for name in ("build", "source", "store", "output")), snapshot_every=1024)
    test.operation_seconds = args.operation_seconds
    test.record(operation_seconds=args.operation_seconds, batch_count=args.count, first_case=args.case)
    status, start = 0, time.monotonic()
    try:
        exercise(test, args.count, args.case)
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
    print(f"source classification boot: {len(test.checks)} checks, {failed} failures", flush=True)
    return result["exit"]


if __name__ == "__main__":
    sys.exit(main())
