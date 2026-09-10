#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Check retention of actual prepared input and focus use after journal restore.
# Owns: One new output directory and the boot processes started by the test.
# Threading: One disk driver; the device owns embedding, retention and recall batches.
# Lifetime: One bounded integration test with existing local model files.

# Inputs: build, source, model store and new output directory. Output: logs and checks.
# Exit: 0 pass, 1 failed check or cleanup, 4 bad arguments.
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
from live_boot_test import Test, Run, FIELDS, envelope, wait
from text_boot_test import TextRun, setup, pack, query, PROCESSOR

FIRST = "The check color is amber. Reply with the word noted."
NEXT = "State the check color in the remembered input. Reply with one word."

def compiled_limits(test):
    if not hasattr(test, "memory_limits"):
        text = test.command([test.build / "aotx_ccir_state", "--limits"], "limits")
        match = re.fullmatch(r"objects=(\d+) payload_bytes=(\d+) image_bytes=(\d+)\n", text)
        test.check(match is not None, "compiled capacity report has exact fields")
        objects, payload, image = map(int, match.groups())
        test.check(objects > 0 and payload > 0 and image == 128 + objects * 256 + payload and
                   16 + 2 * image <= 0xFFFFFFFF, "compiled capacity fits its wire fields")
        test.memory_limits = dict(objects=objects, payload_bytes=payload, image_bytes=image)
    return test.memory_limits


def text_request(f, cut, ordinal, text, focus=False):
    data = query(f, cut, ordinal)
    f.put(data, 68, int(focus), 4)
    q = bytearray(data[128:])
    f.put(q, 148, len(text), 4); q[4640:] = bytes(8192 - 4640)
    q[4640:4640 + len(text)] = text.encode()
    data[128:] = q
    return data


def retain_request(f):
    data = envelope(f, b"AOTXRTN1", 160, 0)
    r = bytearray(160)
    f.put(r, 4, 1, 4); r[8:24] = f.identity(8000); f.put(r, 24, 1)
    r[32:48], r[48:64], r[64:80] = f.identity(3001), f.identity(2000), f.identity(1000)
    f.put(r, 104, 1); r[112:128] = f.identity(10000)
    f.put(r, 128, 750000, 4); f.put(r, 132, 1, 4)
    data[64:] = r
    return data


def transfers(test, run, f):
    records = test.command([test.build / "aotx_journal", "records", run.journal, "--boot", run.boot], "records")
    result, active, identity, total, operation = [], bytearray(), None, 0, 0
    limits = {4: 528448, 5: 37952, 6: 528448, 7: 562240, 8: 10304,
              9: 64 + 64 * 384 + compiled_limits(test)["image_bytes"],
              10: 64 + 64 * 9168 + compiled_limits(test)["image_bytes"]}
    for line in records.splitlines():
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if fields.get("type") != "33" or fields.get("class") != "1":
            continue
        p = bytes.fromhex(fields["body"])
        test.check(32 < len(p) <= 192 and len(p) == int(fields["body_len"]), "journal part bounds")
        op, size, offset = f.get(p, 4, 4), f.get(p, 24, 4), f.get(p, 28, 4)
        if op not in limits:
            continue
        if not offset:
            test.check(not active and 64 <= size <= limits[op], "bounded transfer starts in order")
            total, operation, identity = size, op, p[8:24]
        test.check(f.get(p, 0, 4) == 1 and p[8:24] == identity and op == operation and
                   size == total and offset == len(active), "exact transfer order and identity")
        test.check(len(p) - 32 == min(160, total - offset), "exact part length")
        active.extend(p[32:])
        if len(active) == total:
            result.append((operation, identity.hex(), bytes(active)))
            active.clear(); identity = None
    test.check(not active and bool(result), "complete relevant transfers exist")
    return result


def validate_retained(test, f, records, expected):
    requests = [r for r in records if r[0] == 8]
    results = [r for r in records if r[0] == 9]
    choices = [r[2] for r in records if r[0] == 7]
    test.check(len(requests) == len(results) == 1 and len(choices) == 2, "one retained transaction and two decisions")
    test.check(requests[0][1] == results[0][1] and requests[0][2] == expected, "exact retention source and matching result ID")
    data = results[0][2]
    test.check(data[:8] == b"AOTXRCH1" and f.get(data, 8, 4) == 1 and f.get(data, 40, 4) == 384 and
               f.get(data, 44, 4) == 0 and not any(data[56:64]), "retained result header")
    test.check(data[64:224] == expected[64:] and f.get(data, 224, 4) == 1 and
               not any(data[228:256]), "exact retained row and focus count")
    test.check(data[256:272] == f.identity(2000) and f.get(data, 272) == 1 and
               not any(data[280:448]), "exact retained focus reference")
    tail = data[448:]
    test.check(len(tail) == f.get(data, 48) and tail[:8] == b"AOTXLOG1" and
               f.get(tail, 20, 4) == 3 and f.get(tail, 32) == 1, "canonical three-object mutation")
    objects = {row[8:24]: (row, payload) for row, payload in f.state_rows(tail)}
    test.check(set(objects) == {f.identity(n) for n in (3001, 1000, 2000)}, "exact retained object IDs")
    prepared = choices[0][128:8320]
    width = f.get(prepared, 128, 4)
    test.check(prepared[96:128] == PROCESSOR and 0 < width <= 1024, "actual prepared vector identity")
    for identity, kind, source in ((3001, 1, 3), (1000, 9, 4), (2000, 7, 3)):
        row, payload = objects[f.identity(identity)]
        test.check(f.get(row, 2, 2) == kind and f.get(row, 40) == 1 and row[64:80] == f.identity(10000) and
                   f.get(row, 176, 4) == 0, "retained kind, version, principal and scope")
        test.check(f.get(row, 180, 4) == source and f.get(row, 184, 4) == 0, "source class remains separate from evidence")
        if identity != 1000:
            test.check(payload[:8] == b"AOTXMEM1" and f.get(payload, 12, 4) == len(FIRST) and
                       payload[32:] == FIRST.encode(), "exact retained input text")
        else:
            test.check(payload[:8] == b"AOTXVEC2" and f.get(payload, 8, 4) == 2 and
                       f.get(payload, 12, 4) == width and f.get(payload, 16, 4) == 4 and
                       f.get(payload, 20, 4) == 1, "typed vector header")
            test.check(payload[24:56] == prepared[64:96] and payload[56:88] == prepared[96:128] and
                       payload[128:] == prepared[160:160 + width * 4], "retention keeps the exact resident vector")
            test.check(payload[88:104] == f.identity(3001) and f.get(payload, 104) == 1 and
                       not any(payload[112:128]) and row[96:112] == payload[88:104] and
                       f.get(row, 112) == 1, "vector has the exact typed event source")
    working = objects[f.identity(2000)][0]
    test.check(f.get(working, 192, 4) == 750000 and f.get(working, 188, 4) == 1 and
               f.get(working, 232) == 1, "working importance, retention and policy")
    chosen = choices[1]
    test.check(f.get(chosen, 68, 4) == 1 and f.get(chosen, 128 + 140, 4) == 0 and
               f.get(chosen, 128 + 144, 4) == 1, "device adds focus without required references")
    test.check(chosen[128 + 4448:128 + 4464] == f.identity(2000) and f.get(chosen, 128 + 4464) == 1,
               "prepared query contains the retained focus")
    selection = chosen[8320:]
    test.check(f.get(selection, 4, 4) == 1 and selection[16:32] == f.identity(2000) and
               f.get(selection, 32) == 1, "focus selects retained working memory")
    (test.output / "retained.json").write_text(json.dumps(dict(
        source=prepared[64:128].hex(), width=width, objects=[k.hex() for k in objects],
        tail_bytes=len(tail), focus=f.identity(2000).hex()), indent=2) + "\n")


def accepted(run):
    return [r for r in run.events(0) if r.get("kind") == "line" and r.get("status") == "accepted"]


def exercise(test):
    f = setup(test)
    ccir, bind = pack(test, f, [])
    run = TextRun(test, "before"); run.ready()
    run.send("spawn worker"); wait(lambda: "spawn: worker on slots 1" in run.console(), run.child)
    run.send("agent 1 pages 160"); run.send("task 1 Reply with the word ready.")
    Run.reply(run, 1, 1, "ready")
    run.operation("load", ccir, 1, 0); run.operation("bind", bind, 3, 1)
    inputs = test.output / "inputs"
    first = text_request(f, 0, 1, FIRST)
    (inputs / "first-text").write_bytes(first)
    run.operation("text", inputs / "first-text", 6, 1); run.reply(1)
    before = accepted(run)
    test.check(len(before) == 1 and before[0]["text"] == FIRST, "original accepted input is present")
    retained = retain_request(f); (inputs / "retain").write_bytes(retained)
    run.operation("retain", inputs / "retain", 8, 1)
    test.check(accepted(run) == before, "retention does not add another conversation input")
    second = text_request(f, 3, 2, NEXT, True)
    (inputs / "second-text").write_bytes(second)
    test.check(not any(second[128 + 64:128 + 132]) and not any(second[128 + 160:128 + 4640]),
               "caller supplies no vector or memory reference")
    run.operation("text", inputs / "second-text", 6, 1); run.reply(2)
    test.check(len(accepted(run)) == 2 and accepted(run)[1]["text"] == NEXT, "focus query keeps the original input")
    audit = [r for r in run.events(0) if r.get("kind") == "selection" and r.get("turn") == 2]
    test.check(any(f"objects {f.identity(2000).hex()}@1" in r.get("text", "") for r in audit), "audit names the retained selection")
    run.summary(); run.stop(killed=True)
    old, manifests, transcript = run.summary(), run.turns(), run.events(0)
    original = transfers(test, run, f); validate_retained(test, f, original, retained)
    paths = list(inputs.iterdir()); shutil.rmtree(inputs)
    test.check(all(not p.exists() for p in paths), "original input files removed", paths=list(map(str, paths)))
    after = TextRun(test, "after", True); after.ready()
    report = wait(lambda: re.search(r"^restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+) "
                  r"pages (\d+) paced (\d+) rejected (\d+)$", after.path.read_text(), re.M), after.child)
    test.check(int(report[1]) == int(old["replayed"]) and int(report[2], 16) == int(old["state_hash"], 16), "exact restored record count and state hash")
    test.check(int(report[3]) == 0 and int(report[6]) == 0, "restore has no decode or input refusals")
    def durable():
        current = after.summary()
        return current if current["boot"] == after.boot and current.get("restore_hash") != "none" else None
    current = wait(durable, after.child)
    test.check(current.get("restore_of") == old["boot"] and current["restore_hash"] == old["state_hash"], "durable restore parent and hash")
    wait(lambda: len(after.turns()) >= len(manifests) and len(after.events(0)) >= len(transcript), after.child)
    keys = lambda data: sorted(tuple(r[k] for k in FIELDS) for r in data)
    test.check(keys(after.turns()) == keys(manifests) and after.events(0) == transcript, "exact restored prompt hashes and audit")
    test.check(transfers(test, after, f) == original, "exact restored mutation, focus and choices")
    path = test.output / "next-text"; path.write_bytes(text_request(f, 3, 3, NEXT, True))
    after.operation("text", path, 6, 1); after.reply(3)
    test.check(any(f"objects {f.identity(2000).hex()}@1" in r.get("text", "") for r in after.events(0)
                   if r.get("kind") == "selection" and r.get("turn") == 3), "restored focus remains available")
    test.check(all(not p.exists() for p in paths), "source files remain absent")
    after.stop(); after.summary()
    test.command([test.build / "aotx_journal", "manifest", after.journal], "manifest-chain")


def main():
    if len(sys.argv) != 5:
        print("usage: retain_boot_test.py BUILD SOURCE STORE OUTPUT", file=sys.stderr); return 4
    test = Test(*(Path(v).resolve() for v in sys.argv[1:]))
    status, begin = 0, time.monotonic()
    try:
        exercise(test)
    except Exception as error:
        status = 1; (test.output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
        print(f"retention boot failed: {error}", file=sys.stderr, flush=True)
    finally:
        for run in reversed(test.active):
            try:
                run.close()
            except Exception as error:
                status = 1; test.record(cleanup_error=str(error))
        if any(not r["passed"] for r in test.checks):
            status = 1
        result = dict(checks=len(test.checks), failed=sum(not r["passed"] for r in test.checks),
                      seconds=time.monotonic() - begin, exit=status, claim="integration only")
        (test.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
