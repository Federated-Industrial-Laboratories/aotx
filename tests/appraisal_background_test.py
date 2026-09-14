#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check actual supplied and native background appraisal with foreground interruption.
# Owns: One new output directory and only the runtime processes started here.
# Threading: One disk driver; CUDA performs all model and cognitive processing.
# Lifetime: Actual source input, background completion, interruption and ordinary recovery.

# Inputs: Build, source, model store, output, batch and creator mode. Output: logs and checks.
# Exit: 0 pass, 1 failed check or cleanup, 2 bad arguments.
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Run, rows as json_rows
from runtime_boot_test import RuntimeTest
from checkpoint_boot_test import spawn, durable, file_state
from text_boot_test import setup
from retain_boot_test import compiled_limits
from appraisal_runtime_test import control, restore_report
from appraisal_runtime_cases import initial, bindings, request, evidence, assess
from appraisal_background_cases import Deadline, appraisal, policy, pages, released, replies, package, records, verify
from turn_manifest import same_turns


def quiet(run, args, expected_pending, label):
    deadline = Deadline(args.work_seconds); first = appraisal(run, deadline); creator = policy(run, args.mode, deadline)
    run.test.check(not first["active"] and first["pending"] == expected_pending, "the observed quiet interval starts with the expected queue")
    if label in ("enabled-empty", "disabled-pending"):
        run.test.check(not first["calls"] and not creator["calls"] and not creator["decision"],
                       "empty or disabled initial work has made no appraisal or creator call")
    tokens = json_rows(run.journal / run.boot / "tokens.jsonl"); begin = time.monotonic()
    end = begin + args.quiet_seconds; last = first
    while time.monotonic() < end:
        time.sleep(max(0, min(0.1, end - time.monotonic())))
        last = appraisal(run, deadline)
        run.test.check(not last["active"] and last["pending"] == expected_pending and
                       all(last[key] == first[key] for key in ("calls", "completed", "refused", "interrupted")),
                       "quiet observations cause no appraisal decoder calls", case=label, appraisal=last)
    after = policy(run, args.mode, deadline)
    run.test.check(after["calls"] == creator["calls"] and after["decision"] == creator["decision"] and
                   json_rows(run.journal / run.boot / "tokens.jsonl") == tokens,
                   "an unchanged empty or disabled queue causes no creator decision or sampled token", case=label)
    released(run, deadline)
    run.test.record(quiet_case=label, observed_seconds=time.monotonic() - begin, appraisal_before=first, appraisal_after=last,
                    policy_before=creator, policy_after=after)
    return last


def typed(run, path, count, deadline):
    start = len(run.console()); run.send("memory text " + str(path))
    return typed_result(run, start, count, deadline)


def typed_result(run, start, count, deadline):
    found = deadline.until(run, lambda: re.search(r"memory: operation 6 status (\d+) rows (\d+)", run.console()[start:]))
    run.test.check(tuple(map(int, found.groups())) == (0, count), "the whole foreground source batch reaches exact typed admission")


def input_batch(run, f, state, args, ordinal, case):
    deadline = Deadline(args.work_seconds); path = run.test.output / "inputs" / ("input-" + str(ordinal))
    data = request(f, args.batch, f.get(state, 32), ordinal, case); path.write_bytes(data)
    run.test.record(input=str(path), sha256=hashlib.sha256(data).hexdigest(), ordinal=ordinal, source_cut=f.get(state, 32))
    typed(run, path, args.batch, deadline); replies(run, args.batch, ordinal, "noted", deadline); released(run, deadline)


def complete(run, args, before):
    deadline = Deadline(args.work_seconds); begin = time.monotonic()
    def ready():
        row = appraisal(run, deadline)
        run.test.check(row["refused"] == before["refused"] and row["interrupted"] == before["interrupted"],
                       "automatic work remains admitted until completion", appraisal=row)
        return row if not row["active"] and not row["pending"] and row["completed"] == before["completed"] + args.batch else None
    final = deadline.until(run, ready)
    run.test.check(final["calls"] == before["calls"] + 1 and not final["status"], "one actual decoder batch completes all background sources")
    released(run, deadline)
    run.test.record(background_completion_seconds=time.monotonic() - begin, appraisal_before=before, appraisal_after=final)
    return final


def interrupt(run, f, args, before):
    deadline = Deadline(args.work_seconds)
    def active():
        row = appraisal(run, deadline)
        if row["completed"] != before["completed"] or row["refused"] != before["refused"] or row["interrupted"] != before["interrupted"]:
            raise AssertionError("Background work ended before an active decoder lease could be interrupted: " + str(row))
        if row["active"] and row["calls"] == before["calls"] + 1:
            pool = pages(run, deadline)
            return (row, pool) if pool[0] < pool[1] else None
        return None
    started, pool = deadline.until(run, active)
    creator = policy(run, args.mode, deadline)
    run.test.check(creator["calls"] == 2 and creator["pending"] == args.batch, "the second background decoder batch follows its selected creator")
    data = request(f, args.batch, creator["source"] + args.batch, 3, "unknown")
    path = run.test.output / "inputs" / "foreground"; path.write_bytes(data)
    run.test.record(input=str(path), sha256=hashlib.sha256(data).hexdigest(), expected_source_cut=creator["source"] + args.batch,
                    active_appraisal=started, active_policy=creator, allocated_pages=pool[1] - pool[0])
    console_start = len(run.console()); begin = time.monotonic(); run.send("memory text " + str(path))
    def ended():
        row = appraisal(run, deadline)
        run.test.check(row["completed"] in (before["completed"], before["completed"] + args.batch) and row["refused"] == before["refused"],
                       "foreground arrival preserves completed work while the new source remains independently eligible", appraisal=row)
        return row if row["interrupted"] == before["interrupted"] + args.batch else None
    final = deadline.until(run, ended); interruption = time.monotonic() - begin
    typed_result(run, console_start, args.batch, deadline)
    replies(run, args.batch, 3, "noted", deadline); completion = time.monotonic() - begin
    run.test.check(final["calls"] in (before["calls"] + 1, before["calls"] + 2),
                   "only interrupted work and the new foreground source can start a decoder batch", appraisal=final)
    run.test.record(foreground_interruption_seconds=interruption, foreground_reply_seconds=completion,
                    caller_work_seconds=args.work_seconds,
                    appraisal_before=started, appraisal_after=final)
    return final


def exercise(test, args):
    selected = test.output / "selected-models"; selected.mkdir()
    entries = [dict(row) for row in json_rows(test.store / "manifest.jsonl") if row.get("role") in ("language", "embedding")]
    test.check(len(entries) == 2 and {row["role"] for row in entries} == {"language", "embedding"}, "one existing language and embedding model are selected")
    for entry in entries:
        original = (test.store / entry["path"]).resolve(); entry["path"] = entry["role"] + ".gguf"
        (selected / entry["path"]).symlink_to(original)
    (selected / "manifest.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries)); test.store = selected
    model = bytes.fromhex(next(row["sha256"] for row in entries if row["role"] == "language"))
    test.check(len(model) == 32, "selected actual language model has an exact digest")
    f = setup(test); limits = compiled_limits(test)
    test.check(limits["objects"] >= args.batch * 64 and limits["payload_bytes"] >= args.batch * 32768,
               "compiled memory fits all source and interruption batches")
    bundle, digest = package(test, args)
    raw, memory, bind = (test.output / "inputs" / name for name in ("initial-checkpoint", "initial.aotxccir", "bind"))
    raw.write_bytes(initial(f, args.batch)); test.command([test.build / "aotx_recall_cli_fixture", raw, "-", memory, 1], "initial-memory")
    mirror = test.output / "memory.aotxccir"
    extra = ["--memory-mirror", mirror, "--policy", bundle]
    if args.mode == "native": extra += ["--policy-trust", digest]
    options = dict(extra=extra, roles="language,embedding")
    run = Run(test, "background", **options); run.ready(args.ready_seconds); spawn(run, args.batch)
    run.operation("load", memory, 1, 0); durable(run)
    bind.write_bytes(bindings(f, args.batch, args.batch, args.pages)); run.operation("bind", bind, 3, args.batch); durable(run)
    control(run, "on", 3); control(run, f"limits {args.pages} {args.tokens} {args.ticks} {args.batch}")
    control(run, "background on", 7); quiet(run, args, 0, "enabled-empty")
    control(run, "background off", 3); durable(run)
    state = file_state(test, mirror, f, args.batch, 0, mode=2)[2]
    input_batch(run, f, state, args, 1, "unknown"); durable(run)
    before = quiet(run, args, args.batch, "disabled-pending")
    audits, turns = [run.events(i) for i in range(args.batch)], run.turns()
    control(run, "background on", 7); complete(run, args, before); durable(run)
    test.check([run.events(i) for i in range(args.batch)] == audits and run.turns() == turns,
               "completed background work creates no external conversation turn or event")
    state = file_state(test, mirror, f, args.batch, 1, mode=2)[2]
    assess(test, f, state, args.batch, 1, "unknown", model)
    quiet(run, args, 0, "completed-empty"); control(run, "background off", 3); durable(run)
    state = file_state(test, mirror, f, args.batch, 1, mode=2)[2]
    input_batch(run, f, state, args, 2, "mixed"); durable(run)
    before = quiet(run, args, args.batch, "second-disabled-pending")
    turns = run.turns(); audits = [run.events(i) for i in range(args.batch)]
    control(run, "background on", 7); interrupt(run, f, args, before)
    follow = dict(before, calls=before["calls"] + 1, interrupted=before["interrupted"] + args.batch)
    complete(run, args, follow); durable(run)
    final = file_state(test, mirror, f, args.batch, 3, mode=2)
    assess(test, f, final[2], args.batch, 3, "unknown", model)
    for i in range(args.batch):
        events = run.events(i)
        test.check(events[:len(audits[i])] == audits[i] and all(row.get("turn") == 3 for row in events[len(audits[i]):]),
                   "interrupted internal work leaves all old conversation events exact", slot=i)
    test.check(run.turns()[:len(turns)] == turns and len(run.turns()) == len(turns) + args.batch,
               "only the new foreground batch creates external turns")
    encoded = verify(test, f, final[2], records(test, run, f, limits["image_bytes"]), args.batch, digest, model)
    accepted = evidence(f, final[2]); counters = quiet(run, args, 0, "interrupted-empty")
    durable(run); final = file_state(test, mirror, f, args.batch, 3, mode=2)
    test.check(evidence(f, final[2]) == accepted, "interrupted work cannot silently retry or strengthen prior exposure")
    audits, turns = [run.events(i) for i in range(args.batch)], run.turns()
    run.stop(killed=True); summary = run.summary()
    after = Run(test, "restored", True, **options); after.ready(args.ready_seconds)
    restore_report(test, after, int(summary["replayed"]), int(summary["state_hash"], 16)); durable(after)
    restored = file_state(test, mirror, f, args.batch, 3, mode=2)
    test.check(restored == final and appraisal(after, Deadline(args.work_seconds))["calls"] == 0,
               "ordinary recovery restores exact memory and bindings without appraisal generation")
    deadline = Deadline(args.ready_seconds)
    deadline.until(after, lambda: len(after.turns()) >= len(turns) and all(len(after.events(i)) >= len(audits[i]) for i in range(args.batch)))
    original_manifest = (run.journal / "manifest" / f"{run.boot}.jsonl").read_bytes()
    recovered_manifest = (after.journal / "manifest" / f"{after.boot}.jsonl").read_bytes()
    test.check(same_turns(original_manifest, recovered_manifest),
               "ordinary replay preserves every turn field and agent order with valid raw manifest chains")
    test.check([after.events(i) for i in range(args.batch)] == audits,
               "ordinary replay preserves exact foreground audit bytes")
    test.check(verify(test, f, restored[2], records(test, after, f, limits["image_bytes"]), args.batch, digest, model) == encoded and
               policy(after, args.mode, deadline)["calls"] == 0, "recovery preserves exact creator decisions and actual model responses without native execution")
    quiet(after, args, 0, "recovered-empty"); after.stop()
    test.record(final_appraisal=counters, model=model.hex(), policy_digest=digest)


def main():
    parser = argparse.ArgumentParser(description="Check actual ABI 2 background appraisal and foreground interruption.")
    for name in ("build", "source", "store", "output"):
        parser.add_argument(name, type=lambda value: Path(value).resolve())
    parser.add_argument("batch", type=int, choices=(1, 64)); parser.add_argument("mode", choices=("supplied", "native"))
    parser.add_argument("--native-image", type=lambda value: Path(value).resolve())
    parser.add_argument("--architecture", type=int)
    parser.add_argument("--pages", type=int, default=160); parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--ticks", type=int, default=16384); parser.add_argument("--quiet-seconds", type=float, default=1)
    parser.add_argument("--ready-seconds", type=float, default=900); parser.add_argument("--work-seconds", type=float, default=900)
    args = parser.parse_args()
    times = (args.quiet_seconds, args.ready_seconds, args.work_seconds)
    if min(args.pages, args.tokens, args.ticks) <= 0 or max(args.pages, args.ticks) > 0xffffffff or args.tokens > 4096 or \
            any(not math.isfinite(value) or value <= 0 for value in times):
        parser.error("Resource counts and caller deadlines must be positive; output tokens must not exceed 4096.")
    if args.mode == "native" and (not args.native_image or not args.native_image.is_file() or not args.architecture or args.architecture < 1):
        parser.error("Native mode requires an existing public maintenance PTX image and its architecture number.")
    test = RuntimeTest(args.build, args.source, args.store, args.output, snapshot_every=1024)
    begin, status_code = time.monotonic(), 0
    try:
        exercise(test, args)
    except (Exception, KeyboardInterrupt) as error:
        status_code = 1; (test.output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
        print(f"appraisal background failed: {error}", file=sys.stderr, flush=True)
    finally:
        for run in reversed(test.active):
            if not run.log.closed:
                try: run.close()
                except Exception as error: status_code = 1; test.record(cleanup_error=str(error))
        test.flush_checks(); status_code |= any(not row["passed"] for row in test.checks)
    result = dict(batch=args.batch, mode=args.mode, checks=len(test.checks), failed=sum(not row["passed"] for row in test.checks),
                  seconds=time.monotonic() - begin, exit=int(status_code))
    (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)
    return status_code


if __name__ == "__main__":
    sys.exit(main())
