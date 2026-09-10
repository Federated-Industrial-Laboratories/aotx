#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check automatic input retention and exact cold recovery through the live console.
# Owns: One new output directory and only the boot processes started here.
# Threading: One disk driver; CUDA owns encoding, recall, memory and model processing.
# Lifetime: One bounded test with existing local model files.

# Inputs: build, source, model store, new output and vector-1, vector-64 or text-1.
# Output: source files, logs and exact checks. Exit: 0 pass, 1 failure, 2 bad arguments.
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Test, Run, FIELDS, wait
from text_boot_test import TextRun, setup, pack, PROCESSOR
from retain_boot_test import transfers, text_request, NEXT
from capacity_boot_test import prepare, request, color

ROW, PREPARED, RETAINED = 9168, 8192, 8784
FIRST = "The check color is amber. Reply with that color."


def input_bytes(f, cut, turn, count, text):
    if text:
        data = text_request(f, cut, turn, FIRST if turn == 1 else NEXT, turn > 1)
    else:
        data = request(f, cut, turn, count)
    for i in range(count):
        q = 128 + i * 8256
        f.put(data, q + 132, max(1, turn - 1), 4)
        f.put(data, q + 136, 1024, 4)
    return data


def audits(test, run, f, turn, count, text, references):
    for i in range(count):
        audit = [r for r in run.events(i) if r.get("kind") == "selection" and r.get("turn") == turn]
        test.check(len(audit) == 1, "one selection audit per automatic input")
        line = audit[0]["text"]
        expected = [r[i] for r in references[:turn - 1]]
        if turn == 1 and not text:
            expected = [f.identity(60002 + 6 * i)]
        objects = " objects" + (" " + ",".join(k.hex() + "@1" for k in expected) if expected else "")
        test.check(line.endswith(objects), "audit contains exactly the prior selected memories")
        test.check(f"principal {f.identity(10000 + i).hex()} " in line and "scope 0 " in line,
                   "automatic audit preserves each private principal")
        test.check(f"retained {references[turn - 1][i].hex()}@1" in line, "audit names this input's retained working memory")
        accepted = [r for r in run.events(i) if r.get("kind") == "line" and r.get("status") == "accepted" and r.get("turn") == turn]
        test.check(len(accepted) == 1, "automatic retention adds no duplicate input line")


def validate(test, f, records, initial_cut, count, text, turns):
    inputs = [r for r in records if r[0] == (6 if text else 4)]
    choices = [r for r in records if r[0] == 10]
    test.check(len(inputs) == len(choices) == turns and not any(r[0] in (5, 7, 8, 9) for r in records),
               "each input has one combined decision without a second retention request")
    references = []
    for turn, (source, choice) in enumerate(zip(inputs, choices), 1):
        cut = initial_cut + (turn - 1) * 3 * count
        raw, data = source[2], choice[2]
        test.check(source[1] == choice[1] and raw == input_bytes(f, cut, turn, count, text), "exact original input and decision identity")
        test.check(data[:8] == b"AOTXACH1" and f.get(data, 8, 4) == count and f.get(data, 12, 4) == 1 and
                   f.get(data, 32) == cut and f.get(data, 40, 4) == ROW and not any(data[44:48]) and not any(data[56:64]),
                   "combined result carries the original cut and success")
        tail = data[64 + count * ROW:]
        test.check(len(tail) == f.get(data, 48) and tail[:8] == b"AOTXLOG1" and f.get(tail, 20, 4) == 3 * count and
                   f.get(tail, 32) == cut + 1, "three new objects per accepted input in one canonical tail")
        stored = {r[8:24]: (r, p) for r, p in f.state_rows(tail)}
        refs, ids = [], set()
        for i in range(count):
            prefix = raw[64 + i * 8256:128 + i * 8256]
            original = raw[128 + i * 8256:8320 + i * 8256]
            base = 64 + i * ROW; q = data[base + 64:base + 64 + PREPARED]
            selection = data[base + 64 + PREPARED:base + RETAINED]
            r = data[base + RETAINED:base + ROW]
            test.check(data[base:base + 64] == prefix and q[:64] == original[:64] and q[4640:] == original[4640:],
                       "prepared query keeps distinct original authority, IDs and text")
            width = f.get(q, 128, 4)
            if text:
                embedding = next(r for r in test.models if r["role"] == "embedding")
                test.check(q[64:96].hex() == embedding["sha256"] and q[96:128] == PROCESSOR and 0 < width <= 1024,
                           "prepared text identifies the actual resident encoder")
                test.check(not any(original[64:132]) and not any(original[160:4256]), "text caller supplies no encoded vector")
            else:
                test.check(q[64:144] == original[64:144] and q[148:4448] == original[148:4448], "prepared-vector bytes remain exact")
            event, working, component = r[32:48], r[48:64], r[64:80]
            ids.update((event, working, component)); refs.append(working)
            test.check(event == original[:16] and working == b"AOTXGEN1" + (cut + 1 + 2 * i).to_bytes(8, "little") and
                       component == b"AOTXGEN1" + (cut + 2 + 2 * i).to_bytes(8, "little"), "device-generated IDs are distinct and deterministic")
            test.check(f.get(r, 0, 4) == i and f.get(r, 4, 4) == 1 and r[8:24] == prefix[16:32] and f.get(r, 24) == turn,
                       "retained metadata names the correct binding and ordinal")
            test.check(not any(r[80:104]) and f.get(r, 104) == 1 and r[112:128] == original[16:32] and
                       f.get(r, 128, 4) == 0xFFFFFFFF and not any(r[132:160]), "automatic policy does not invent significance or supersession")
            expected_focus = [row[i] for row in references] + [working]
            test.check(f.get(r, 160, 4) == turn and not any(r[164:192]), "working focus grows with accepted input")
            for j, key in enumerate(expected_focus):
                test.check(r[192 + j * 24:208 + j * 24] == key and f.get(r, 208 + j * 24) == 1, "exact ordered retained focus")
            test.check(not any(r[192 + 24 * turn:]), "unused focus bytes are zero")
            for key, kind, source_kind in ((event, 1, 3), (component, 9, 4), (working, 7, 3)):
                row, payload = stored[key]
                test.check(f.get(row, 2, 2) == kind and f.get(row, 40) == 1 and row[64:80] == original[16:32] and
                           f.get(row, 176, 4) == 0 and f.get(row, 180, 4) == source_kind, "automatic objects retain exact type and private authority")
                test.check(f.get(row, 184, 4) == 0 and f.get(row, 188, 4) == 0 and f.get(row, 192, 4) == 0xFFFFFFFF,
                           "stored evidence and importance remain unknown with ordinary retention")
                if kind == 9:
                    test.check(payload[:8] == b"AOTXVEC2" and payload[24:88] == q[64:128] and payload[88:104] == event and
                               payload[128:] == q[160:160 + width * 4] and row[96:112] == event,
                               "automatic component retains exact encoded bytes and source dependency")
                else:
                    length = f.get(original, 148, 4)
                    test.check(payload[:8] == b"AOTXMEM1" and f.get(payload, 12, 4) == length and payload[32:] == original[4640:4640 + length],
                               "stored event and working memory retain exact input text")
            expected = [row[i] for row in references]
            if turn == 1 and not text:
                expected = [f.identity(60002 + 6 * i)]
            test.check(f.get(selection, 0, 4) == 1 and f.get(selection, 4, 4) == len(expected), "selection uses only the pre-write memory")
            for j, key in enumerate(expected):
                test.check(selection[16 + j * 32:32 + j * 32] == key and f.get(selection, 32 + j * 32) == 1,
                           "recorded selection names the exact prior reference")
        test.check(set(stored) == ids and len(ids) == 3 * count, "canonical tail has exactly the declared new objects")
        references.append(refs)
    return references


def exercise(test, count, text):
    if text:
        f = setup(test); ccir, bind = pack(test, f, []); inputs, cut = test.output / "inputs", 0
    else:
        f, inputs, ccir, bind, cut = prepare(test, count)
    test.models = [json.loads(line) for line in (test.store / "manifest.jsonl").read_text().splitlines() if line.strip()]
    data = bytearray(bind.read_bytes())
    for i in range(count):
        f.put(data, 124 + i * 64, 1, 4)
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
    for turn in (1, 2):
        path = inputs / f"input-{turn}"; path.write_bytes(input_bytes(f, cut + (turn - 1) * 3 * count, turn, count, text))
        run.operation(kind, path, op, count)
        for i in range(count):
            Run.reply(run, i, turn, "amber" if text else color(i, turn > 1))
    run.summary(); run.stop(killed=True)
    old, manifests, audit = run.summary(), run.turns(), [run.events(i) for i in range(count)]
    original = transfers(test, run, f); references = validate(test, f, original, cut, count, text, 2)
    for turn in (1, 2):
        audits(test, run, f, turn, count, text, references)
    paths = list(inputs.iterdir()); shutil.rmtree(inputs)
    test.check(all(not p.exists() for p in paths), "original container, checkpoint, bindings and inputs are removed")
    after = cls(test, "after", True); after.ready()
    report = wait(lambda: re.search(r"restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+) "
                  r"pages (\d+) paced (\d+) rejected (\d+)$", after.path.read_text(), re.M), after.child)
    test.check(int(report[1]) == int(old["replayed"]) and int(report[2], 16) == int(old["state_hash"], 16) and
               int(report[3]) == int(report[6]) == 0, "exact restore count and hash without refused work")
    def durable():
        current = after.summary()
        return current if current["boot"] == after.boot and current.get("restore_hash") != "none" else None
    current = wait(durable, after.child)
    test.check(current.get("restore_of") == old["boot"] and current["restore_hash"] == old["state_hash"], "durable restore identity and state hash")
    wait(lambda: len(after.turns()) >= len(manifests) and all(len(after.events(i)) >= len(audit[i]) for i in range(count)), after.child)
    keys = lambda data: sorted(tuple(r[k] for k in FIELDS) for r in data)
    test.check(keys(after.turns()) == keys(manifests) and [after.events(i) for i in range(count)] == audit,
               "cold recovery preserves exact prompt hashes and conversation logs")
    test.check(transfers(test, after, f) == original, "replay preserves every original combined decision without extra retention")
    path = test.output / "next-input"; path.write_bytes(input_bytes(f, cut + 6 * count, 3, count, text))
    after.operation(kind, path, op, count)
    for i in range(count):
        Run.reply(after, i, 3, "amber" if text else color(i, True))
    after.stop(); after.summary()
    references = validate(test, f, transfers(test, after, f), cut, count, text, 3)
    audits(test, after, f, 3, count, text, references)
    test.check(all(not p.exists() for p in paths), "source files remain absent after new input and recall")
    test.command([test.build / "aotx_journal", "manifest", after.journal], "manifest-chain")


def main():
    if len(sys.argv) != 6 or sys.argv[5] not in ("vector-1", "vector-64", "text-1"):
        print("usage: auto_boot_test.py BUILD SOURCE STORE OUTPUT vector-1|vector-64|text-1", file=sys.stderr); return 2
    text, count = sys.argv[5].startswith("text"), int(sys.argv[5].split("-")[1])
    test = Test(*(Path(v).resolve() for v in sys.argv[1:5]), snapshot_every=1024)
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
        if any(not r["passed"] for r in test.checks):
            status = 1
        test.flush_checks()
        result = dict(batch=count, text=text, checks=len(test.checks), failed=sum(not r["passed"] for r in test.checks),
                      seconds=time.monotonic() - start, exit=status, claim="automatic retention integration")
        (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
