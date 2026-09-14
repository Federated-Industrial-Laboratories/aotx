#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check actual appraisal, useful recall and ordinary and complete file recovery.
# Owns: One new output directory and only the runtime processes started here.
# Threading: One disk driver; CUDA executes all model, appraisal and recall work.
# Lifetime: Actual conversation input, journal recovery and complete file activation.

# Inputs: Build, source, model store, output and batch. Output: commands and checks.
# Exit: 0 pass, 1 failed check or cleanup, 2 bad arguments.
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Run, wait
from runtime_boot_test import RuntimeTest, RuntimeRun, durable as runtime_durable, sections
from checkpoint_boot_test import spawn, durable, file_state
from text_boot_test import setup
from retain_boot_test import transfers, compiled_limits
from appraisal_runtime_cases import (PROCESSOR, initial, bindings, request, source, source_id,
    rows, current, evidence, assess, selected, response, FIELDS)


def status(run, command="status"):
    start = len(run.console()); run.send("appraisal " + command)
    def received():
        text = run.console()[start:]
        a = re.search(r"appraisal: flags (\d+) pending (\d+) active (\d+) status (\d+)", text)
        b = re.search(r"appraisal work: calls (\d+) completed (\d+) refused (\d+) interrupted (\d+)", text)
        if a and b:
            return dict(zip(("flags", "pending", "active", "status", "calls", "completed", "refused", "interrupted"),
                            map(int, a.groups() + b.groups())))
        return None
    return wait(received, run.child, 60)


def control(run, command, flags=None):
    start = len(run.console())
    row = status(run, command)
    if command != "status":
        def accepted():
            text = run.console()[start:]
            match = re.search(r"memory: operation 15 status (\d+) rows (\d+)", text)
            if match:
                run.test.check(match.groups() == ("0", "1"), "settings reach atomic memory admission")
                return True
            return False
        wait(accepted, run.child, 60)
        row = status(run)
    run.test.check(not row["status"] and (flags is None or row["flags"] == flags),
                   "appraisal control reports the requested state", command=command, appraisal=row)
    return row


def process(run, count, seconds):
    before = status(run)
    audits = [run.events(i) for i in range(count)]
    turns = run.turns()
    run.test.check(before["pending"] == count and not before["active"], "the complete source batch is pending before explicit work", appraisal=before)
    run.send("appraisal run")
    def completed():
        row = status(run)
        if row["refused"] > before["refused"] or row["interrupted"] > before["interrupted"]:
            raise AssertionError("The actual appraisal batch was refused or interrupted: " + str(row))
        return row if not row["active"] and not row["pending"] and row["completed"] == before["completed"] + count else None
    result = wait(completed, run.child, seconds)
    run.test.check(not result["status"] and result["calls"] > before["calls"],
                   "explicit appraisal uses the resident model and completes every source", appraisal=result)
    run.test.check(run.turns() == turns and [run.events(i) for i in range(count)] == audits,
                   "internal appraisal adds no user turn or transcript event")
    return result


def actual_input(run, f, count, cut, ordinal, case, path, seconds, conversation=8000, generation=0):
    path.write_bytes(request(f, count, cut, ordinal, case, conversation=conversation, generation=generation))
    run.operation("text", path, 6, count)
    for i in range(count):
        wait(lambda: any(row.get("agent") == i and row.get("turn") == ordinal for row in run.turns()), run.child, seconds)
        run.reply(i, ordinal, "noted" if case != "recall" else "no")
    run.test.check(all(len([row for row in run.events(i) if row.get("kind") == "reply" and row.get("turn") == ordinal]) == 1
                       for i in range(count)), "each admitted memory input produces one user reply")


def journal_results(test, run, f, state, count, model_digest):
    limits = compiled_limits(test)
    records = transfers(test, run, f, {14: 64 + 64 * 13920 + limits["image_bytes"],
                                      17: 64 + 64 * 4160 + limits["image_bytes"]})
    results = [data for op, _, data in records if op == 17 and f.get(data, 12, 4)]
    test.check(len(results) == 3 and all(f.get(data, 12, 4) == count for data in results),
               "three complete actual source batches have recorded appraisal results")
    stored = rows(f, state); seen = set()
    for data in results:
        n, tail = f.get(data, 12, 4), f.get(data, 24)
        test.check(data[:8] == b"AOTXAPS1" and f.get(data, 8, 4) == 1 and not f.get(data, 32, 4) and
                   not any(data[36:64]) and len(data) == 64 + n * 4160 + tail,
                   "recorded appraisal result has exact independent framing")
        encoded = data[64 + n * 4160:]
        test.check(encoded[:8] == b"AOTXLOG1" and f.get(encoded, 32) == f.get(data, 16) + 1 and
                   f.get(encoded, 80) == tail, "recorded result includes its canonical typed memory tail")
        for i in range(n):
            row = data[64 + i * 4160:64 + (i + 1) * 4160]
            key = bytes(row[:16]), f.get(row, 16)
            length = f.get(row, 56, 4)
            test.check(key in stored and key not in seen and 0 < length <= 4096 and not f.get(row, 60, 4) and
                       row[24:56] == model_digest and not any(row[64 + length:]),
                       "each actual decoder output has an exact source queue, model and bounded response")
            seen.add(key); qr, qp = stored[key]
            result = response(test, row[64:64 + length])
            matching = [(r, p) for r, p in stored.values() if r[96:112] == qr[96:112] and f.get(r, 112) == f.get(qr, 112)]
            assessments = [(r, p) for r, p in matching if f.get(r, 2, 2) == 3 and len(p) == 128]
            relationships = [(r, p) for r, p in matching if p[:8] == b"AOTXREL1"]
            test.check(len(assessments) == len(relationships) == 1, "recorded output has one persisted assessment and relationship")
            _, ap = assessments[0]; _, rp = relationships[0]
            expected = [f.get(ap, at, 4) for at in (4, 8, 12, 16, 20)] + [f.get(rp, at, 4) for at in (16, 20, 24, 28)]
            test.check([result[key] for key in FIELDS[:9]] == expected, "persisted appraisal values equal actual model output without host interpretation")
            event = stored[(bytes(qr[96:112]), f.get(qr, 112))][1][32:]
            for text, payload, at in ((result["evidence"], ap, 120), (result["task"], rp, 56), (result["commitment"], rp, 64)):
                start, size = f.get(payload, at, 4), f.get(payload, at + 4, 4)
                test.check(isinstance(text, str) and text.encode() == event[start:start + size],
                           "persisted source quote equals the actual recorded response")
    test.record(actual_result_count=len(results), appraisal_sources=len(seen),
                result_sha256=[hashlib.sha256(data).hexdigest() for data in results])
    return results


def restore_report(test, run, count, state_hash):
    match = re.search(r"restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+).* rejected (\d+)", run.path.read_text())
    test.check(match and int(match[1]) == count and int(match[2], 16) == state_hash and not int(match[3]) and not int(match[4]),
               "cold recovery restores the exact accepted record count and state hash")


def recovered_state(test, f, saved, restored):
    test.check(restored[2] == saved[2], "file-only recovery preserves every cognitive memory byte")
    before, after = saved[4], restored[4]
    for state in (saved, restored):
        live, replay = state[4], state[7]
        test.check(len(live) >= 128 and live[:8] == b"AOTXLCP1" and f.get(live, 8, 4) == 1 and
                   f.get(live, 12, 4) == 17416 and f.get(live, 16, 4) <= 64 and
                   len(live) == 128 + f.get(live, 16, 4) * 17416,
                   "saved live state has the complete declared binding batch")
        test.check(len(replay) >= 128 and replay[:8] == b"AOTXRPL1" and f.get(replay, 8, 4) == 1 and
                   f.get(replay, 12, 4) == 2 and f.get(live, 72) == f.get(replay, 24) and f.get(replay, 24) != 0 and
                   f.get(live, 64) == f.get(replay, 64) and f.get(replay, 16) != 0,
                   "each capture tick and memory revision match its complete recorded replay cut")
    test.check(f.get(saved[7], 16) != f.get(restored[7], 16) and f.get(after, 72) >= f.get(before, 72),
               "the recovered capture belongs to a new boot at or after its saved runtime tick")
    test.check(before[:72] == after[:72], "recovery keeps the exact live identity, allocation and accepted memory state")
    cursor, searches = 80, []
    for i in range(f.get(before, 16, 4)):
        at = 128 + i * 17416 + 128 + 8192 + 12
        prior, current = f.get(before, at, 4), f.get(after, at, 4)
        test.check(prior in (0, 1) and current == 0,
                   "recorded live choices recover without an executed vector search", row=i, before=prior, after=current)
        test.check(before[cursor:at] == after[cursor:at], "recovery keeps every recorded live query and result byte", row=i)
        cursor = at + 4; searches.append((prior, current))
    test.check(before[cursor:] == after[cursor:], "recovery keeps the exact final result, context and working focus bytes")
    test.record(recovery_memory_sha256=hashlib.sha256(restored[2]).hexdigest(),
                recovery_live_sha256=[hashlib.sha256(value).hexdigest() for value in (before, after)],
                recovery_capture_ticks=[f.get(value, 72) for value in (before, after)], recovery_executed_searches=searches)


def complete_recovery(test, f, trained, count, corrected, old, args):
    inputs = test.output / "inputs"
    raw, memory = inputs / "trained-checkpoint", inputs / "trained.aotxccir"
    raw.write_bytes(trained)
    test.command([test.build / "aotx_recall_cli_fixture", raw, "-", memory, 1], "trained-memory")
    runtime = test.output / "runtime.aotxccir"
    test.command([test.build / "aotx_ccir_pack", "--memory", memory, "--models", test.store,
        "--modules", test.output / "modules", "--settings", test.output / "settings", "--roles", "language,embedding",
        "--output", runtime, "--phrases", test.source / "tests/fixtures/quality/refusal-phrases.txt"], "runtime-pack")
    created = sections(test, runtime)
    test.check(created[2] == trained, "complete package contains exact actually trained cognitive memory")
    originals = [test.store, test.output / "modules", test.output / "settings", inputs]
    for path in originals:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    test.check(all(not path.exists() for path in originals), "owned model paths, settings, modules and source inputs are removed")
    run = RuntimeRun(test, "complete", runtime); run.ready(args.ready_seconds); spawn(run, count)
    test.check("network: IPv4 and IPv6 sockets are disabled" in run.path.read_text(), "complete activation has no network dependency")
    runtime_durable(run)
    state = sections(test, runtime)[2]
    test.check(state == trained and status(run)["calls"] == 0, "complete activation restores appraisal memory without appraisal generation")
    path = test.output / "complete-bind"
    path.write_bytes(bindings(f, count, f.get(state, 32), args.pages, conversation=18000))
    run.operation("bind", path, 3, count); runtime_durable(run)
    actual_input(run, f, count, f.get(state, 32), 1, "recall", test.output / "complete-input", args.work_seconds,
                 conversation=18000, generation=1)
    runtime_durable(run); saved = sections(test, runtime)
    selected(test, run, f, saved[2], count, 1, corrected, old)
    test.check(evidence(f, saved[2]) == evidence(f, trained), "new file-only recall does not create appraisal exposure")
    run.stop(killed=True); shutil.rmtree(run.journal)
    after = RuntimeRun(test, "file-recovered", runtime); after.ready(args.ready_seconds)
    restore_report(test, after, f.get(saved[7], 40), f.get(saved[7], 32))
    runtime_durable(after); restored = sections(test, runtime)
    recovered_state(test, f, saved, restored)
    test.check(status(after)["calls"] == 0, "file-only replay does not generate new appraisals")
    actual_input(after, f, count, f.get(restored[2], 32), 2, "recall", test.output / "file-recovered-input", args.work_seconds,
                 conversation=18000, generation=1)
    runtime_durable(after); final = sections(test, runtime)
    selected(test, after, f, final[2], count, 2, corrected, old)
    test.check(evidence(f, final[2]) == evidence(f, trained), "continued recovered recall leaves learned exposure unchanged")
    test.check(all(not path.exists() for path in originals), "continued operation needs only the complete runtime file")
    after.stop()


def exercise(test, args):
    external = test.store; local = test.output / "selected-models"; local.mkdir()
    entries = [json.loads(line) for line in (external / "manifest.jsonl").read_text().splitlines() if line.strip()]
    entries = [dict(entry) for entry in entries if entry.get("role") in ("language", "embedding")]
    test.check(len(entries) == 2 and {entry["role"] for entry in entries} == {"language", "embedding"},
               "one language model and one embedding model are selected")
    for entry in entries:
        source_path = (external / entry["path"]).resolve(); entry["path"] = entry["role"] + ".gguf"
        (local / entry["path"]).symlink_to(source_path)
    (local / "manifest.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    test.store = local; f = setup(test); count = args.batch
    model_digest = bytes.fromhex(next(entry["sha256"] for entry in entries if entry["role"] == "language"))
    test.check(len(model_digest) == 32, "actual language model digest has the declared width")
    limits = compiled_limits(test)
    test.check(limits["objects"] >= count * 64 and limits["payload_bytes"] >= count * 32768,
               "configured memory fits complete source, correction and recovery cases")
    raw, memory, bind = (test.output / "inputs" / name for name in ("initial-checkpoint", "initial.aotxccir", "bind"))
    raw.write_bytes(initial(f, count)); test.command([test.build / "aotx_recall_cli_fixture", raw, "-", memory, 1], "initial-memory")
    mirror = test.output / "memory.aotxccir"
    run_args = dict(extra=("--memory-mirror", mirror), roles="language,embedding")
    run = Run(test, "initial", **run_args); run.ready(args.ready_seconds); spawn(run, count)
    run.operation("load", memory, 1, 0); durable(run)
    bind.write_bytes(bindings(f, count, count, args.pages)); run.operation("bind", bind, 3, count); durable(run)
    control(run, "on", 3); control(run, f"limits {args.pages} {args.tokens} {args.ticks} {count}")
    control(run, "priority 0 1000000"); control(run, "background off", 3); durable(run)
    state = file_state(test, mirror, f, count, 0, mode=2)[2]
    previous = corrected = None
    for ordinal, case in enumerate(("unknown", "mixed", "correction"), 1):
        actual_input(run, f, count, f.get(state, 32), ordinal, case, test.output / "inputs" / case, args.work_seconds)
        process(run, count, args.work_seconds); durable(run)
        state = file_state(test, mirror, f, count, ordinal, mode=2)[2]
        found = assess(test, f, state, count, ordinal, case, model_digest, old=previous)
        if case == "mixed": previous = found
        if case == "correction": corrected = found
    accepted = evidence(f, state); counters = status(run)
    run.send("appraisal run"); control(run, "status"); time.sleep(1)
    repeated = status(run); durable(run)
    test.check(repeated["calls"] == counters["calls"] and repeated["completed"] == counters["completed"] and not repeated["pending"],
               "repeated processing of completed sources performs no model work", before=counters, after=repeated)
    test.check(evidence(f, file_state(test, mirror, f, count, 3, mode=2)[2]) == accepted,
               "repeated explicit work cannot strengthen an existing relationship")
    control(run, "writes off", 2)
    for ordinal, enabled in ((4, False), (5, True)):
        control(run, "recall " + ("on" if enabled else "off"), 2 if enabled else 0); durable(run)
        state = file_state(test, mirror, f, count, ordinal - 1, mode=2)[2]
        actual_input(run, f, count, f.get(state, 32), ordinal, "recall", test.output / "inputs" / f"recall-{ordinal}", args.work_seconds)
        durable(run); state = file_state(test, mirror, f, count, ordinal, mode=2)[2]
        selected(test, run, f, state, count, ordinal, corrected, previous, enabled)
        test.check(evidence(f, state) == accepted, "read-only cognitive recall does not add or strengthen learned evidence")
    recorded = journal_results(test, run, f, state, count, model_digest)
    audits = [run.events(i) for i in range(count)]; before = file_state(test, mirror, f, count, 5, mode=2)
    run.stop(killed=True); summary = run.summary()
    recovered = Run(test, "ordinary-recovered", True, **run_args); recovered.ready(args.ready_seconds)
    restore_report(test, recovered, int(summary["replayed"]), int(summary["state_hash"], 16))
    durable(recovered); restored = file_state(test, mirror, f, count, 5, mode=2)
    test.check(restored == before, "ordinary journal recovery preserves exact cognitive memory and bindings")
    wait(lambda: all(len(recovered.events(i)) >= len(audits[i]) for i in range(count)), recovered.child, args.ready_seconds)
    test.check([recovered.events(i) for i in range(count)] == audits, "ordinary replay preserves exact input, selection and response audits")
    test.check(status(recovered)["calls"] == 0, "ordinary replay does not generate appraisals")
    test.check(journal_results(test, recovered, f, restored[2], count, model_digest) == recorded,
               "ordinary replay preserves every actual model appraisal response byte")
    actual_input(recovered, f, count, f.get(restored[2], 32), 6, "recall", test.output / "inputs" / "recovered-recall", args.work_seconds)
    durable(recovered); trained = file_state(test, mirror, f, count, 6, mode=2)[2]
    selected(test, recovered, f, trained, count, 6, corrected, previous)
    test.check(evidence(f, trained) == accepted, "fresh recall after ordinary recovery preserves learned exposure")
    recovered.stop()
    if not args.ordinary_only:
        complete_recovery(test, f, trained, count, corrected, previous, args)


def main():
    parser = argparse.ArgumentParser(description="Check actual model appraisal and cognitive memory recovery.")
    for name in ("build", "source", "store", "output"):
        parser.add_argument(name, type=lambda value: Path(value).resolve())
    parser.add_argument("batch", type=int, choices=(1, 64))
    parser.add_argument("--pages", type=int, default=160)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--ticks", type=int, default=16384)
    parser.add_argument("--ready-seconds", type=int, default=900)
    parser.add_argument("--work-seconds", type=int, default=900)
    parser.add_argument("--ordinary-only", action="store_true", help="Run ordinary memory and journal checks without complete model packaging.")
    args = parser.parse_args()
    if min(args.pages, args.tokens, args.ticks, args.ready_seconds, args.work_seconds) < 1:
        parser.error("Resource counts and caller time limits must be positive.")
    test = RuntimeTest(args.build, args.source, args.store, args.output, snapshot_every=1024)
    begin, status_code = time.monotonic(), 0
    try:
        exercise(test, args)
    except (Exception, KeyboardInterrupt) as error:
        status_code = 1; (test.output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
        print(f"appraisal runtime failed: {error}", file=sys.stderr, flush=True)
    finally:
        for run in reversed(test.active):
            if not run.log.closed:
                try: run.close()
                except Exception as error: status_code = 1; test.record(cleanup_error=str(error))
        test.flush_checks(); status_code |= any(not row["passed"] for row in test.checks)
    result = dict(batch=args.batch, ordinary_only=args.ordinary_only, checks=len(test.checks),
                  failed=sum(not row["passed"] for row in test.checks), seconds=time.monotonic() - begin, exit=int(status_code))
    (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)
    return status_code


if __name__ == "__main__":
    sys.exit(main())
