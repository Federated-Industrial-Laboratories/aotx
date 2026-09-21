#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Verify model-derived memory, corrections and recovery through real runtime files.
# Owns: One new output directory and only the runtime processes started here.
# Threading: One disk driver; all interpretation and memory processing run on CUDA.
# Lifetime: Initial operation, copied-file recovery and exact journal replay.

# Inputs: build, source, model store, new output and batch count 1 or 64.
# Output: commands, model results and exact checks. Exit: 0 pass, 1 failure, 2 bad arguments.
import argparse
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


def target_plan(i):
    return "serve soup tomorrow" if i % 2 else "cook lentils tonight"


def text(i, turn, count):
    name = person(i)
    sources = {
        1: name + " will cook lentils tonight. Their peer is " + person((i + 17) % 64) + ".",
        2: name + " will serve soup tomorrow. This is a separate plan.",
        3: "Correction: " + name + " will not " + target_plan(i) + ". The earlier " +
           ("soup" if i % 2 else "cooking") + " plan changed.",
        4: name + " may bake bread tomorrow. This is uncertain.",
        5: "Will " + name + " " + target_plan(i) + "?",
    }
    return sources[turn] + (" Reply with exactly one word: yes or no." if turn == 5 else
                            " Reply with exactly one word: noted.")


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
        c = q + 6688; data[c:c + 8] = b"AOTXCTX2"
        f.put(data, c + 8, 2, 4); f.put(data, c + 44, 2, 4)
        data[q + 7760:q + 7776] = f.identity(20000 + i)
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


def name_in_quote(i, quote):
    return person(i) in quote


PROFILE_SHA = "af7743df1359d59a72c536c1109927ed2e04fa1ce69e76dc6374efb373c4b946"
EDGE_SPACE = " \t\n\v\f\r\u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"


def decisions(test, f, choices, count, expected_spans=None):
    for choice in choices:
        test.check(choice[:8] == b"AOTXICH2" and f.get(choice, 40, 4) == 18672,
                   "new semantic decisions retain the full versioned row")
        cut = f.get(choice, 32)
        (test.output / f"choice-{cut}.bin").write_bytes(choice)
        values = []
        for i in range(count):
            row = 64 + i * 18672; meta = row + 9168; table = row + 13920
            test.check(f.get(choice, meta, 4) == 2 and f.get(choice, table, 4) == 1 and
                       f.get(choice, table + 4, 4) <= 16, "each source row has a recorded bounded correction table", slot=i)
            length = f.get(choice, meta + 4, 4)
            raw = choice[meta + 128:meta + 128 + length].decode()
            refs = [(choice[table + 16 + j * 32:table + 32 + j * 32].hex(),
                     f.get(choice, table + 32 + j * 32)) for j in range(f.get(choice, table + 4, 4))]
            first = row + 14448; first_length = f.get(choice, first + 4, 4)
            first_raw = choice[first + 128:first + 128 + first_length].decode()
            classified = json.loads(first_raw); statements = [item for item in classified if item[1] == "statement"]; final = json.loads(raw)
            test.check(f.get(choice, first, 4) == 2 and 0 < first_length <= 4096 and
                       choice[first + 8:first + 40] == choice[meta + 8:meta + 40] and
                       f.get(choice, first + 76, 4) == f.get(choice, meta + 76, 4) and
                       f.get(choice, first + 72, 4) == len(statements) and
                       f.get(choice, meta + 80, 4) == bool(statements) and
                       f.get(choice, first + 80, 4) == 1 and f.get(choice, first + 84, 4) == len(classified) and
                       choice[first + 88:first + 120].hex() == PROFILE_SHA and not any(choice[first + 120:first + 128]) and
                       not any(choice[first + 128 + first_length:first + 4224]) and
                       not any(choice[meta + 84:meta + 128]), "both stage outputs have exact model, role and bounded metadata", slot=i)
            source = choice[row + 4704:row + 4704 + f.get(choice, row + 212, 4)].decode()
            end = 0
            for quote, label in classified:
                start = source.find(quote, end)
                test.check(label in ("statement", "request") and bool(quote) and start >= end and
                           not source[end:start].strip(EDGE_SPACE) and quote.strip(EDGE_SPACE) == quote,
                           "first output preserves ordered complete source byte coverage", slot=i)
                end = start + len(quote)
            test.check(not source[end:].strip(EDGE_SPACE), "first output covers the final source fragment", slot=i)
            if expected_spans is not None:
                test.check(source in expected_spans and [item[0] for item in classified] == expected_spans[source],
                           "first quotes equal the independent complete span list", slot=i)
            for j, (quote, _) in enumerate(statements):
                test.check(j < len(final) and final[j][0] in (3, 4) and final[j][1] == quote,
                           "classification preserves each complete accepted statement in order", slot=i)
            for kind, quote, target in final[len(statements):]:
                test.check(kind in (1, 2) and target == 0 and bool(quote) and source.find(quote) >= 0 and
                           source.find(quote, source.find(quote) + 1) < 0 and
                           any(quote in statement[0] for statement in statements),
                           "optional names and tasks stay inside accepted statements", slot=i)
            if not statements:
                test.check(raw == "[]" and not refs, "empty extraction skips classification and target selection", slot=i)
            values.append(dict(slot=i, raw=raw, first_raw=first_raw, accepted=len(statements), rejected=len(classified) - len(statements),
                               first_processor=choice[first + 40:first + 72].hex(), second_call=f.get(choice, meta + 80, 4),
                               targets=refs, model=choice[meta + 8:meta + 40].hex(),
                               processor=choice[meta + 40:meta + 72].hex(), role=f.get(choice, meta + 76, 4)))
        (test.output / f"choice-{cut}.json").write_text(json.dumps(values, indent=2) + "\n")


def exercise(test, count):
    f = setup(test); inputs = test.output / "inputs"
    checkpoint, ccir, bind = (inputs / name for name in ("checkpoint", "memory.aotxccir", "bind"))
    checkpoint.write_bytes(f.image([], 0, 10))
    test.command([test.build / "aotx_recall_cli_fixture", checkpoint, "-", ccir, 1], "pack-empty-memory")
    data = batch(f, b"AOTXBND1", 64, 0, count)
    for i in range(count):
        at = 64 + i * 64; f.put(data, at, i, 4)
        data[at + 8:at + 24], data[at + 40:at + 56] = f.identity(10000 + i), f.identity(8000 + i)
        f.put(data, at + 56, 512, 4); f.put(data, at + 60, 2, 4)
    bind.write_bytes(data)
    mirror = test.output / "identity.aotxccir"
    args = dict(extra=("--memory-mirror", mirror), roles="language,embedding")
    run = Run(test, "before", **args); run.ready(); spawn(run, count)
    run.operation("load", ccir, 1, 0, seconds=test.operation_seconds); durable(run)
    run.operation("bind", bind, 3, count, seconds=test.operation_seconds); durable(run)
    cut, old = 0, []
    for turn in (1, 2, 3, 4):
        path = inputs / f"input-{turn}"; path.write_bytes(request(f, count, cut, turn))
        run.operation("text", path, 6, count, seconds=test.operation_seconds)
        for i in range(count):
            run.reply(i, turn, "noted", seconds=test.operation_seconds)
        durable(run); saved = file_state(test, mirror, f, count, turn, mode=2)
        cut = f.get(saved[2], 32); found = interpretations(test, f, saved[2], count, turn)
        for i, values in enumerate(found):
            if turn == 1:
                assertions = [bytes(row[8:24]) for row, p, quote in values
                              if f.get(p, 16, 4) == 3 and person(i) in quote and "will cook lentils tonight" in quote]
                test.check(bool(assertions), "actual model extracts the stated plan", kind="behavior", slot=i, values=[v[2] for v in values])
                old.append(assertions)
                test.check(any(f.get(p, 16, 4) == 3 and "Their peer is" in quote for _, p, quote in values),
                           "actual model retains the second independent source fact", kind="behavior", slot=i,
                           values=[v[2] for v in values])
            elif turn == 2:
                test.check(any(f.get(p, 16, 4) == 3 and "will serve soup tomorrow" in quote for _, p, quote in values),
                           "actual model retains a separate plan on the same topic", kind="behavior", slot=i)
                test.check(all(not any(row[136:152]) for row, _, _ in values),
                           "unrelated statements do not supersede an existing assertion", kind="behavior", slot=i)
                if i % 2:
                    old[i] = [bytes(row[8:24]) for row, p, quote in values
                              if f.get(p, 16, 4) == 3 and person(i) in quote and "will serve soup tomorrow" in quote]
                    test.check(bool(old[i]), "alternate correction has an exact prior assertion", kind="behavior", slot=i)
            elif turn == 3:
                test.check(any(f.get(p, 16, 4) == 4 and bytes(row[136:152]) in old[i] and
                               "will not " + target_plan(i) in quote for row, p, quote in values),
                           "actual model corrects the exact prior plan with negation intact", kind="behavior", slot=i,
                           values=[v[2] for v in values])
                test.check(all(not any(row[136:152]) or bytes(row[136:152]) in old[i] for row, _, _ in values),
                           "the correction leaves the other plan unchanged", kind="behavior", slot=i)
            else:
                test.check(any(f.get(p, 16, 4) == 3 and name_in_quote(i, quote) and "may bake bread tomorrow" in quote
                               for _, p, quote in values), "actual model preserves uncertainty in a complete assertion",
                           kind="behavior", slot=i, values=[v[2] for v in values])
            test.check(all(not (f.get(p, 16, 4) in (2, 3, 4) and "Reply with" in quote) for _, p, quote in values),
                       "reply format commands do not become stored tasks or facts", kind="behavior", slot=i, turn=turn)
    limits = compiled_limits(test)
    records = transfers(test, run, f, {14: 64 + 64 * 18672 + limits["image_bytes"]})
    semantic = [p for op, _, p in records if op == 14]
    test.check(len(semantic) == 4 and all(f.get(p, 8, 4) == count for p in semantic), "all actual batches have complete source semantic decisions")
    expected_spans = {text(i, turn, count): [part + "." for part in text(i, turn, count).split(". ")[:-1]] +
                      [text(i, turn, count).split(". ")[-1]] for i in range(count) for turn in (1, 2, 3, 4)}
    decisions(test, f, semantic, count, expected_spans)
    before = saved; run.stop(killed=True)
    portable = test.output / "copied.aotxccir"; shutil.copy2(mirror, portable)
    shutil.rmtree(inputs); shutil.rmtree(run.journal); mirror.unlink()
    test.check(not inputs.exists() and not run.journal.exists() and not mirror.exists(), "only the copied cognitive file remains from the prior runtime")
    args = dict(extra=("--memory-mirror", portable), roles="language,embedding")
    after = Run(test, "after", **args); after.ready(); spawn(after, count)
    after.operation("resume", portable, 11, count, seconds=test.operation_seconds); durable(after)
    test.check(file_state(test, portable, f, count, 4, mode=2) == before, "copied file restores every accepted state byte")
    path = test.output / "next-input"; path.write_bytes(request(f, count, cut, 5))
    after.operation("text", path, 6, count, seconds=test.operation_seconds)
    for i in range(count):
        after.reply(i, 5, "no", seconds=test.operation_seconds)
    durable(after); saved = file_state(test, portable, f, count, 5, mode=2)
    rows = f.state_rows(saved[2])
    for i, values in enumerate(interpretations(test, f, saved[2], count, 5)):
        test.check(all(f.get(p, 16, 4) not in (3, 4) for _, p, _ in values),
                   "a question adds no inferred assertion or correction", kind="behavior", slot=i, values=[v[2] for v in values])
        targets = {bytes(row[136:152]) for row, _ in rows if any(row[136:152])}
        test.check(any(ref in targets for ref in old[i]), "the exact original plan remains superseded after copied-file continuation", slot=i)
    actual = [p for op, _, p in transfers(test, after, f, {14: 64 + 64 * 18672 + limits["image_bytes"]}) if op == 14]
    expected_spans = {text(i, 5, count): [text(i, 5, count).split("? ")[0] + "?", text(i, 5, count).split("? ")[1]]
                      for i in range(count)}
    decisions(test, f, actual, count, expected_spans)
    for choice in actual:
        for i in range(count):
            row = 64 + i * 18672; q = row + 64
            test.check(f.get(choice, q + 140, 4) == f.get(choice, q + 144, 4) == 0,
                       "continued reply selects scoped source facts with no required or focus pins", slot=i)
    after.stop(); summary = after.summary(); audits = [after.events(i) for i in range(count)]
    replay = Run(test, "replay", True, **args); replay.ready()
    report = wait(lambda: re.search(r"restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+) "
                  r"pages (\d+) paced (\d+) rejected (\d+)$", replay.path.read_text(), re.M), replay.child, seconds=test.operation_seconds)
    test.check(int(report[1]) == int(summary["replayed"]) and int(report[2], 16) == int(summary["state_hash"], 16) and
               int(report[3]) == int(report[6]) == 0, "semantic journal replay preserves exact accepted state and token hash")
    wait(lambda: all(len(replay.events(i)) >= len(audits[i]) for i in range(count)), replay.child, seconds=test.operation_seconds)
    test.check([replay.events(i) for i in range(count)] == audits, "replayed semantic input and response audits match")
    durable(replay); replay.stop()


def main():
    parser = argparse.ArgumentParser()
    for name in ("build", "source", "store", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("count", type=int, choices=(1, 64))
    parser.add_argument("--operation-seconds", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.operation_seconds <= 7200:
        parser.error("operation seconds must be between 1 and 7200")
    test = Test(*(getattr(args, name).resolve() for name in ("build", "source", "store", "output")), snapshot_every=1024)
    test.operation_seconds = args.operation_seconds
    test.record(operation_seconds=args.operation_seconds, batch_count=args.count)
    status, start = 0, time.monotonic()
    try:
        exercise(test, args.count)
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
    print(f"source memory boot: {len(test.checks)} checks, {failed} failures", flush=True)
    return 1 if status or failed else 0


if __name__ == "__main__":
    sys.exit(main())
